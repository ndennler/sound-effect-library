import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import SpectrogramDataset
from model_definitions import AutoEncoder
from torchvision.transforms import ToTensor

# To test your encoder/decoder, let's encode/decode some sample images
# first, make a PyTorch DataLoader object to sample data batches
batch_size = 8
nworkers = 4        # number of wrokers used for efficient data loading
nz = 64          # dimensionality of the learned embedding

spectrogram_data = SpectrogramDataset(annotations_file='files.csv', img_dir='../spectrograms', transform=ToTensor())
data_loader = DataLoader(spectrogram_data, batch_size=batch_size, num_workers=nworkers)

epochs = 50
learning_rate = 1e-3

# build AE model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')   # use GPU if available
ae_model = AutoEncoder(nz).to(device)    # transfer model to GPU if available
ae_model = ae_model.train()   # set model in train mode (eg batchnorm params get updated)

opt = torch.optim.Adam(ae_model.parameters(), learning_rate)          # create optimizer instance
criterion = nn.MSELoss()   # create loss layer instance


train_it = 0
for ep in range(epochs):
  print(f"Run Epoch {ep}")

  for sample_img in data_loader:
    opt.zero_grad()

    sample_img_gpu = sample_img.to(device)

    out = ae_model.forward(sample_img_gpu)
    rec_loss = criterion(out, sample_img_gpu)
    rec_loss.backward()
    opt.step()

    if train_it % 10 == 0:
      print("It {}: Reconstruction Loss: {}".format(train_it, rec_loss))
    train_it += 1
  
print("Done!")

torch.save(ae_model.state_dict(), 'ae_model.pth')
