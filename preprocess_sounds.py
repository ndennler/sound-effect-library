import librosa
import numpy as np
import audioread
import librosa.display
import matplotlib.pyplot as plt
import os
import scipy
# much borrowed from https://www.kaggle.com/code/rftexas/converting-sounds-into-images-a-general-guide


def read_audio(conf, pathname, trim_long_data):
    a0 = audioread.audio_open(pathname)
    y, sr = librosa.load(a0)

    # trim silence
    if 0 < len(y): # workaround: 0 length causes error
        y, _ = librosa.effects.trim(y) # trim, top_db=default(60)
    # make it unified length to conf.samples
    if len(y) > conf.samples: # long enough
        if trim_long_data:
            y = y[0:0+conf.samples]
    else: # pad blank
        padding = conf.samples - len(y)    # add padding at both ends
        offset = padding // 2
        y = np.pad(y, (offset, conf.samples - len(y) - offset), 'constant')
    return y

# Thanks to the librosa library, generating the mel-spectogram from the audio file

def audio_to_melspectrogram(conf, audio):
    spectrogram = librosa.feature.melspectrogram(audio, 
                                                 sr=conf.sampling_rate,
                                                 window=scipy.signal.hamming,
                                                 n_mels=conf.n_mels,
                                                 hop_length=conf.hop_length,
                                                 n_fft=conf.n_fft,
                                                 fmin=conf.fmin,
                                                 fmax=conf.fmax)
    spectrogram = librosa.power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


class conf:
    # Preprocessing settings
    sampling_rate = 22050
    duration = 5
    hop_length = 128
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = 2048
    samples = sampling_rate * duration

config = conf()
for audio_name in os.listdir('audio_files'):
    if f'{audio_name.split(".")[0]}.npy' not in os.listdir('spectrograms'):
        audio = read_audio(config, f'audio_files/{audio_name}', True)
        S = audio_to_melspectrogram(config, audio)
        
        np.save(f'spectrograms/{audio_name.split(".")[0]}.npy', S)
        


