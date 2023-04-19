import torch
from model_definitions import AutoEncoder
from torch.utils.data import DataLoader
from utils import SpectrogramDataset
import librosa.display
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import scipy
import soundfile as sf

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


model = AutoEncoder(64)
model.load_state_dict(torch.load('ae_model.pth'))
model.eval()

spectrogram_data = SpectrogramDataset(annotations_file='files.csv', img_dir='../spectrograms', transform=ToTensor())
spectrogram_dataloader = DataLoader(spectrogram_data, batch_size=1, num_workers=4)

for img in spectrogram_dataloader:
    out = model(img)
    reconstruction = spectrogram_data.unscale(out)[0,0]
    reconstruction = reconstruction.detach().numpy()
    original = spectrogram_data.unscale(img).detach().numpy()[0,0]

    fig, ax = plt.subplots(nrows=2, ncols=1, sharex=True)

    librosa.display.specshow(reconstruction, y_axis='log', sr=conf.sampling_rate, hop_length=conf.hop_length,
                         x_axis='time', ax=ax[1])
    librosa.display.specshow(original, y_axis='log', sr=conf.sampling_rate, hop_length=conf.hop_length,
                         x_axis='time', ax=ax[0])

    # audio_signal = librosa.feature.inverse.mel_to_audio(
    #     reconstruction, sr=conf.sampling_rate, n_fft=conf.n_fft, hop_length=conf.hop_length, window=scipy.signal.hamming)
    # print(audio_signal, audio_signal.shape)

    # sf.write('reconstructed.wav', audio_signal, conf.sampling_rate)

    # audio_signal = librosa.feature.inverse.mel_to_audio(
    #     original, sr=conf.sampling_rate, n_fft=conf.n_fft, hop_length=conf.hop_length, window=scipy.signal.hamming)
    # print(audio_signal, audio_signal.shape)

    # sf.write('original.wav', audio_signal, conf.sampling_rate)

    plt.show()
    