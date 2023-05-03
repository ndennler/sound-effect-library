from towhee import pipeline

pipeline = pipeline('towhee/audio-embedding-vggish')
out = pipeline('./original.wav')

print(out[0][0].shape)