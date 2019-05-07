import librosa
import numpy as np
import torch
import mlpModel as mlp


def generateMel(filename):
    wav, sr = librosa.load(filename)
    stft_in = np.abs(librosa.stft(wav)) ** 2
    mel_in = librosa.feature.melspectrogram(S=stft_in)
    return mel_in


mel = generateMel("./inWav/arctic_indian_man16.wav")
model = torch.load("MLP1-99",map_location='cpu')

stft_after_model = mlp.generateSTFTFromMel(mel, model)


