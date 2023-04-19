import torch
import torch.nn as nn

class VanillaEncoder(nn.Module):
  def __init__(self, nz):
    super().__init__()

    self.net = nn.Sequential(
        nn.Conv2d(1, 32, kernel_size=(4,24), stride=(2, 4)),
        nn.LeakyReLU(),

        nn.Conv2d(32, 64, kernel_size=(4,24), stride=(2, 3)),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),

        nn.Conv2d(64,128, kernel_size=(3,18), stride=(2,3)),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(),

        nn.Conv2d(128,256,kernel_size=(4,5), stride=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(),

        nn.Conv2d(256,512,kernel_size=(3,4), stride=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),

        nn.Conv2d(512,512,kernel_size=(6,6), stride=1),
        nn.BatchNorm2d(512),
        nn.LeakyReLU(),

        nn.Flatten(),
        nn.Linear(8192, nz)
    )
  def forward(self, x):
    return self.net(x)


class VanillaDecoder(nn.Module):
  def __init__(self, nz):
    super().__init__()

    self.map = nn.Linear(nz, 8192)

    self.net = nn.Sequential(
      nn.BatchNorm2d(512),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(512, 512, kernel_size=(6,6)),

      nn.BatchNorm2d(512),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(512, 256, kernel_size=(3,4)),

      nn.BatchNorm2d(256),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(256, 128, kernel_size=(4,5)),

      nn.BatchNorm2d(128),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(128, 64, kernel_size=(3,18), stride=(2,3)),

      
      nn.BatchNorm2d(64),
      nn.LeakyReLU(),
      nn.ConvTranspose2d(64, 32, kernel_size=(6,24), stride=(2, 3)),

      nn.LeakyReLU(),
      nn.ConvTranspose2d(32, 1, kernel_size=(6,26), stride=(2, 4)),
      
      nn.Sigmoid()     
    )
  def forward(self, x):
    return self.net(self.map(x).reshape(-1, 512, 4, 4))


class AutoEncoder(nn.Module):
  def __init__(self, nz):
    super().__init__()
    self.encoder = VanillaEncoder(nz)
    self.decoder = VanillaDecoder(nz)

  def forward(self, x):
    return self.decoder(self.encoder(x))

  def encode(self, x):
    return self.encoder(x)


if __name__ == '__main__':
  #shape of an image
  test_X = torch.zeros([16, 1, 128, 862])

  enc = VanillaEncoder(32)
  print(enc(test_X).shape)

  dec = VanillaDecoder(32)
  print(dec(enc(test_X)).shape)
