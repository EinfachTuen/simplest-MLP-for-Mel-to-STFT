import librosa
import numpy as np
import pylab as plt

def calculateError(name,original_file,produced_file):
    print("<=============>",name,"<==============>")
    original_wav, originalsr = librosa.load(original_file)
    original_stft = librosa.stft(original_wav)

    produced_wav, produced_sr = librosa.load(produced_file)
    produced_stft = librosa.stft(produced_wav)

    if original_stft.shape[1] > produced_stft.shape[1]:
        original_stft = original_stft[:,4:-3]

    errorMatrix =original_stft - produced_stft
    errorAbsMatrix = np.abs(errorMatrix)

    absoluteError = np.sum(errorAbsMatrix)
    print("Absolute Error = ", absoluteError)

    meanError = np.mean(errorAbsMatrix)
    print("Mean Error = ", meanError)
    plt.imshow(np.transpose(errorAbsMatrix))
    yString =name," Absolute Error:",absoluteError," Mean Error:",meanError
    plt.xlabel(yString)
    plt.show()

calculateError("Original vs Model", "LJ001-0001.wav","actual-Model_result_audioResult STFT.wav")
calculateError("Norm vs Model", "actual-Model_result_audioNormalized STFT.wav","actual-Model_result_audioResult STFT.wav")
calculateError("Norm vs Original", "LJ001-0001.wav","actual-Model_result_audioNormalized STFT.wav")


calculateError("Original vs Test Model h1", "LJ001-0001.wav","test_19999-h1_result_audioResult STFT.wav")


