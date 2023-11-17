import pandas as pd
import numpy as np
import os
import librosa
import re
from multiprocessing import Process
import multiprocessing
import math


def load_data(df_sound):
    pattern = r'[.]'
    max_len = 0
    sound_array = []

    # load all the values in
    for index in range(len(df_sound)):
        file_name = df_sound.iloc[index]['fname']
        parts = re.split(pattern, file_name)
        path = ""
        if parts[1] == "wav":
            path = "uniform_audio"
        elif parts[1] == "mp3":
            path = "raw_audio_files"
        array, sampling_rate = librosa.load(os.path.join(path, file_name))
        max_len = max(max_len, len(array))
        sound_array.append(array)
    
     # zero pad array
    feature_np = np.zeros([len(sound_array), max_len])
    for index, value in enumerate(sound_array):
        new_values = []
        if len(value) < max_len:
            new_values = [0 for x in range(max_len - len(value))]
        feature_np[index] = np.append(value, new_values)
    return feature_np

def average_features(features):
    averages = np.zeros([len(features)])
    for i in range(len(features)):
        averages[i] = np.average(features[i])
    return averages


def construct_features(features, all_features):
    for i in range(len(features)):
        print(i)
        # # spectral features
        # chromgram = librosa.feature.chroma_stft(y=features[i])
        # chromagram_cqt = librosa.feature.chroma_cqt(y = features[i])
        # chromagram_vqt = librosa.feature.chroma_vqt(y = features[i], intervals = "ji5")
        # mfcc = librosa.feature.mfcc(y = features[i])
        rms = librosa.feature.rms(y = features[i]) # 1
        spectral_centroid = librosa.feature.spectral_centroid(y = features[i])
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y = features[i])
        spectral_contrast = librosa.feature.spectral_contrast(y = features[i], n_bands = 3)
        spectral_flatness = librosa.feature.spectral_flatness(y = features[i])
        # mel_spectrogram = librosa.feature.melspectrogram(y=features[i])
        root_features = librosa.feature.rms(y=features[i]) # 1
        roll_off = librosa.feature.spectral_rolloff(y=features[i]) # 1

        # rhythm features
        onset_env = librosa.onset.onset_strength(y=features[i])
        # tempogram = librosa.feature.tempogram(onset_envelope=onset_env)
        # tgr = librosa.feature.tempogram_ratio(tg=tempogram)
        tempo = librosa.feature.tempo(onset_envelope= onset_env)

        # inverse
        # S_inv = librosa.feature.inverse.mel_to_stft(mel_spectrogram)

         # concatenate all features together
        concat_features = np.concatenate((root_features, roll_off, rms, spectral_centroid, spectral_contrast, spectral_bandwidth, spectral_flatness), axis = 0)
        averages = average_features(concat_features)
        averages = np.concatenate((averages, tempo))
        all_features.append(averages)

def load_all(features):
    # calculate ranges to use for cores
    num_cores = os.cpu_count()
    ranges = []
    increments = math.floor(len(features)/num_cores)
    i = 1
    while i <= num_cores:
        last_range = i * increments - 1
        if i == num_cores:
            ranges.append([increments * (i - 1), len(features) - 1])
        else:
            ranges.append([increments * (i - 1), last_range])
        i += 1
    all_features = multiprocessing.Manager().list()

    # run parallel proccesses
    running_tasks = [Process(target = construct_features, args = (features[x[0]: x[1]], all_features)) for x in ranges]
    for tasks in running_tasks:
        tasks.start()
    for tasks in running_tasks:
        tasks.join()
    
    # concatenate into one array
    training_features = np.zeros([len(features), len(all_features[0])])
    for index, value in enumerate(all_features):
        training_features[index] = value
    return training_features
    

if __name__ == "__main__":
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    sound_files = pd.read_csv("./data/files.csv")
    feature_np = load_data(sound_files)
    all_features = []
    result = load_all(feature_np)
    print(result)
    np.save("sound_features.npy", result)