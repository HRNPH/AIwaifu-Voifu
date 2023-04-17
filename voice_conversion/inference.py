import matplotlib.pyplot as plt
import os
import json
import math
import wget
from glob import glob
import torch
import torchaudio
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from voice_conversion.Sovits import utils
from voice_conversion.Sovits.data_utils import UnitAudioLoader, UnitAudioCollate
from voice_conversion.Sovits.models import SynthesizerTrn
from scipy.io.wavfile import write
from torchaudio.functional import resample

class vits_vc_inference:
    def __init__(self) -> None:
        file_root = os.path.dirname(os.path.abspath(__file__))
        pretrain_model_name = 'Chtholly.pth'

        # ask if load available checkpoint
        if not os.path.exists(f'{file_root}/models/{pretrain_model_name}'):
            load_checkpoint = input('Load checkpoint? (y/n): ')
            if load_checkpoint == 'y':
                print('Downloading checkpoint...')
                print('https://huggingface.co/openwaifu/SoVits-VC-Chtholly-Nota-Seniorious-0.1/resolve/main/chtholly.pth')
                wget.download('https://huggingface.co/openwaifu/SoVits-VC-Chtholly-Nota-Seniorious-0.1/resolve/main/chtholly.pth', f'{file_root}/models/{pretrain_model_name}')
        
        model_list = glob(f'{file_root}/models/*.pth')
        if len(model_list) > 1:
            print('Multiple models detected. Please select one:')
            for i, model in enumerate(model_list):
                print(f'{i}. {model}')
            model_index = int(input('Model index: '))
            pretrain_model_path = model_list[model_index]
        elif len(model_list) == 1:
            pretrain_model_path = model_list[0]
        else:
            print('No model detected. Please download one from https://huggingface.co/openwaifu/SoVits-VC-Chtholly-Nota-Seniorious-0.1/resolve/main/chtholly.pth')
            exit(0)

        print(f'Using model {pretrain_model_path}')
        # load content encoder
        self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft")

        # load synthesizer
        self.hps = utils.get_hparams_from_file(f"{file_root}/Sovits/configs/sovits.json") # load config
        self.net_g = SynthesizerTrn(
            self.hps.data.filter_length // 2 + 1,
            self.hps.train.segment_size // self.hps.data.hop_length,
            n_speakers=self.hps.data.n_speakers,
            **self.hps.model)
        _ = self.net_g.eval()
        _ = utils.load_checkpoint(f"{pretrain_model_path}", self.net_g, None)
    
    def inference(self, audio, sr):
        resampled_sr = 22050
        resampled = resample(audio, sr, resampled_sr)
        source = resampled.unsqueeze(0)
        with torch.inference_mode():
            # Extract speech units
            unit = self.hubert.units(source)
            unit_lengths = torch.LongTensor([unit.size(1)])
            # for multi-speaker inference
            # sid = torch.LongTensor([4])
            # Synthesize audio
            audio = self.net_g.infer(unit, unit_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.float().numpy()
            # for multi-speaker inference
            # audio = net_g.infer(unit, unit_lengths, sid, noise_scale=.667, noise_scale_w=0.8, length_scale=1.0)[0][0,0].data.float().numpy()
        return audio, resampled_sr