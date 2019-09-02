import librosa
import librosa.display
import numpy as np
import pylab as plt
import os


def calculateError(name,original_file):
    newFileString = "<=============>"+name+"<==============>"
    original_wav, originalsr = librosa.load(original_file)
    pltPrint(original_wav,"Wavefile 16000hz sampling rate")
    spec = librosa.stft(original_wav)
    pltPrintSpectrum(spec,"Magnitude Spectrogram")
    spec = np.abs(spec)
    melspec = librosa.feature.melspectrogram(S=spec,sr=originalsr, n_mels=80,fmax=8000)
    pltPrintMelSpectrum(melspec, "Mel spectrum")

def pltPrint(wave,name):
    plt.figure(figsize=(10, 4))
    plt.plot(wave)
    plt.ylabel("Amplitude")
    plt.xlabel("Timesteps")
    plt.title(name)
    plt.show()

def pltPrintSpectrum(spec,name):
    plt.figure(figsize=(10, 4))
    spec = np.abs(spec)
    #amaxspec =  np.amax(spec) / 4
    plt.imshow(spec, aspect='auto')
    plt.ylim(0, spec.shape[0])
    plt.ylabel("Frequency")
    plt.xlabel("Timesteps")
    plt.title(name)
    plt.show()

def pltPrintMelSpectrum(melspec,name):
    plt.figure(figsize=(10, 4))
    plt.imshow(melspec, aspect='auto')
    plt.ylim(0, 80)
    plt.ylabel("Frequency Bins")
    plt.xlabel("Timesteps")
    plt.title(name)
    plt.show()

def calculateErrorForFile(original_wavs_folder):
    original_wavs_files = os.listdir(original_wavs_folder)
    for i, name in enumerate(original_wavs_files):

        calculateError(name,original_wavs_folder+name)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--original_wav')

    args = parser.parse_args()

    calculateErrorForFile(args.original_wav)



