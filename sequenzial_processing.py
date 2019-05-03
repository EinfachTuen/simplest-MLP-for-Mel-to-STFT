import librosa
import numpy as np
import mlpModel as mlp
import torch
import torch.nn as nn
from torch.autograd import Variable

wav, sr = librosa.load("./inWav/arctic_indian_man16.wav")
stft_in = np.abs(librosa.stft(wav))**2
mel_in = librosa.feature.melspectrogram(S=stft_in)
stft_in = np.array(stft_in)
mel_in = np.array(mel_in)

print("-----------------------------------")
print("stft_in Size:", stft_in.shape)


print("-----------------------------------")
print("mel_in Shape:", mel_in.shape)
print("mel_in Size:", mel_in.shape[1])
model = mlp.training(mel_in, stft_in)

#predicted = model(Variable(torch.from_numpy(mel_in[:,0]))).data.numpy()
#print(predicted)

