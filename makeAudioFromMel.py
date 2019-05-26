import librosa
import librosa.display
import numpy as np
import torch
import mlpModel as mlp
import matplotlib.pyplot as plt

def generateMel(filename):
    wav, sr = librosa.load(filename)
    stft_in = np.abs(librosa.stft(wav)) ** 2
    plotSTFT(stft_in, 'stft_in')
    mel_in = librosa.feature.melspectrogram(S=stft_in)
    plotSTFT(mel_in, 'mel')
    return mel_in,sr

def plotSTFT(stft,title):
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 2, 1)
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
    librosa.display.specshow(D, y_axis='linear')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()


mel,sr = generateMel("./reserveWav/arctic_indian_man16.wav")
stft_out = librosa.feature.inverse.mel_to_stft(mel, sr=sr)

wav = librosa.istft(stft_out)
plotSTFT(stft_out,'stft_out')

librosa.output.write_wav('output_test.wav', wav, sr)