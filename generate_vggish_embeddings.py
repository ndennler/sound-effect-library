from towhee import pipeline
import pandas as pd
import numpy as np

import torch

model = torch.hub.load('harritaylor/torchvggish', 'vggish')
model.eval()

# Download an example audio file
# import urllib
# url, filename = ("http://soundbible.com/grab.php?id=1698&type=wav", "bus_chatter.wav")
# try: urllib.URLopener().retrieve(url, filename)
# except: urllib.request.urlretrieve(url, filename)

print(model.forward('./uniform_audio/2.wav').shape)


# embedding_pipeline = pipeline('towhee/audio-embedding-vggish')
# # outs = embedding_pipeline('./uniform_audio/2.wav')
# # embeds = outs[0][0].reshape(-1)

# # print(embeds.shape)
directory = pd.read_csv('data/files.csv')
data = []

for i, row in directory.iterrows():
    print(f"uniform_sounds/{row['fname']}")
    outs = model(f"./uniform_audio/{row['fname']}")
    embeds = outs.cpu().detach().numpy().reshape(-1)
    data.append(embeds)

np.save('data/large_embeddings.npy', np.array(data))