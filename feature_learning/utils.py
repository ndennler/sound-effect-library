import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

class SpectrogramDataset(Dataset):
    '''
    Class for loading the spectrogram data.
    '''
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        
        #max and min values in dataset
        self.min_val = -90.41
        self.max_val = 39.82

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 1])
        image = np.load(img_path)
        image = (image - self.min_val) / (self.max_val - self.min_val) #scale to 0 and 1
        if self.transform:
            image = self.transform(image)

        return image

    def unscale(self, img):
        return img * (self.max_val - self.min_val) + self.min_val



if __name__ == '__main__':
    
    data = []
    for fname in os.listdir('../spectrograms'):
        img_path = os.path.join('../spectrograms', fname)
        data.append(np.load(img_path))
    print(np.min(data), np.max(data))

    print('TESTING MAKE DATASET...')
    data = SpectrogramDataset(annotations_file='files.csv', img_dir='../spectrograms', transform=ToTensor())
    print('SUCCESS.')

    print('TESTING GET DATA...')
    for i, data in enumerate(data):
        if i == 0:
            print(f'Shape: {data.shape}, min: {torch.min(data)}, max: {torch.max(data)}')
    print('SUCCESS.')


