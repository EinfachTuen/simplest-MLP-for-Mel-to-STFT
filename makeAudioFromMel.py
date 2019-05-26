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

def mel_to_stft(mel,sr):
    mel_basis = librosa.filters.mel(sr, 2048, n_mels=mel.shape[0],
                            dtype=mel.dtype)

    stft_again = np.dot(np.swapaxes(mel_basis,0,1),mel)
    # Find the non-negative least squares solution, and apply
    # the inverse exponent.
    # We'll do the exponentiation in-place.
    return np.power(stft_again, 1./2.00, out=stft_again)

mel,sr = generateMel("./reserveWav/arctic_indian_man16.wav")
stft_out = mel_to_stft(mel, sr=sr)

wav = librosa.istft(stft_out)
plotSTFT(stft_out,'stft_out')

librosa.output.write_wav('output_test.wav', wav, sr)