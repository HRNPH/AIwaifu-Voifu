import pyopenjtalk
from scipy.io import wavfile
import numpy as np

def tts(text, out_path):
    wav, sr = pyopenjtalk.tts(text)
    wavfile.write(out_path, sr, wav.astype(np.int16))