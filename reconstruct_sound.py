import librosa
import audioread
from PIL import Image
import soundfile as sf
import numpy as np

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
    # n_mels = 512
    # n_fft = n_mels * 5
    n_mels = 2048
    n_fft = 2048
    samples = sampling_rate * duration

audio = read_audio(conf, f'audio_files/asmr_girl_ooh.mp3', True)
# S = audio_to_melspectrogram(conf, audio)
# print(S.shape)
import scipy

melspectrogram = librosa.feature.melspectrogram(
    y=audio, sr=conf.sampling_rate, window=scipy.signal.hamming, n_fft=conf.n_fft, hop_length=conf.hop_length)
spectrogram = librosa.power_to_db(melspectrogram)

spectrogram = spectrogram.astype(np.float32) #+ 30 * np.random.random(spectrogram.shape)

print('melspectrogram.shape', spectrogram.shape)
print(np.max(spectrogram), np.min(spectrogram)) 

spectrogram = np.load('spectrograms/important-stuff-song.npy')

print('melspectrogram.shape', spectrogram.shape)
print(np.max(spectrogram), np.min(spectrogram)) 

audio_signal = librosa.feature.inverse.mel_to_audio(
    spectrogram, sr=conf.sampling_rate, n_fft=conf.n_fft, hop_length=conf.hop_length, window=scipy.signal.hamming)
print(audio_signal, audio_signal.shape)

sf.write('test.wav', audio_signal, conf.sampling_rate)


# M = np.asarray(Image.open('audio_images/whatever.png'))[:,:,0].astype('float64')
# print(M.shape)
# y = librosa.feature.inverse.mel_to_audio(S, hop_length=conf.hop_length, n_fft=conf.n_fft, n)

# # Write out audio as 24bit PCM WAV
# sf.write('stereo_file.wav', y, 22050, subtype='PCM_24')