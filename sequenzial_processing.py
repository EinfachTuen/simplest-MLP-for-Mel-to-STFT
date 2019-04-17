import librosa
import numpy as np
import mlpModel
import torch.nn as nn

wav, sr = librosa.load("./inWav/bbad4n.wav")
stft_in = np.abs(librosa.stft(wav))**2
mel_in = librosa.feature.melspectrogram(S=stft_in)
stft_in = np.array(stft_in)
mel_in = np.array(mel_in)
print("wav:", wav)

print("-----------------------------------")
print("stft_in Size:", stft_in.shape)
print("stft_in:", stft_in)

print("-----------------------------------")
print("mel_in Size:", mel_in.shape)
print("mel_in:", mel_in)



