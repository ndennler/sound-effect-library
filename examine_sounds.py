#https://github.com/tyiannak/pyAudioAnalysis
import librosa
import soundfile as sf
import os
import numpy as np
import matplotlib.pyplot as plt
import audioread
from tqdm import tqdm
import pandas as pd

# print(audioread.audio_open('audio_files/anxious.mp3').duration)
# lengths = []

names = []
for fname in tqdm(os.listdir('spectrograms')):
    names.append(fname)

pd.DataFrame(names).to_csv('files.csv')
# print(len(lengths))
# print(np.mean(lengths))
# print(np.sum(np.array(lengths) > 6))

# a0 = audioread.audio_open('audio_files/anxious.mp3')
# y, samp_f = librosa.load(a0)
# print(samp_f)

# print(len(os.listdir('audio_files')), len(os.listdir('spectrograms')))
