import librosa
import numpy as np
import pylab as plt
import os


def calculateError(name,original_file,produced_file,result_name):
    newFileString = "<=============>"+name+"<==============>"
    printToResultFile(newFileString, result_name)
    original_wav, originalsr = librosa.load(original_file)
    produced_wav, produced_sr = librosa.load(produced_file)

    if len(original_wav) < len(produced_wav):
        produced_wav = produced_wav[0:len(original_wav)]

    if len(produced_wav) < len(original_wav):
        original_wav = original_wav[0:len(produced_wav)]

    original_stft = librosa.stft(original_wav)
    produced_stft = librosa.stft(produced_wav)

    if original_stft.shape[1] > produced_stft.shape[1]:
        original_stft = original_stft[:,4:-3]

    errorMatrix = original_stft - produced_stft
    errorAbsMatrix = np.abs(errorMatrix)

    absoluteError = np.sum(errorAbsMatrix)
    absoluteErrorString = "Absolute Error = "+ str(absoluteError)
    printToResultFile(absoluteErrorString,result_name)

    meanError = np.mean(errorAbsMatrix)
    meanErrorString = "Mean Error = "+ str(meanError)
    printToResultFile(meanErrorString,result_name)

    plt.imshow(errorAbsMatrix)
    yString =name," Absolute Error:",absoluteError," Mean Error:",meanError
    plt.xlabel(yString)
    joinFilename= original_file.replace(".", "-")
    joinFilename= joinFilename.replace("/", "-")

    plt.savefig(result_name+'/'+joinFilename+'.png')
    plt.show()
    return absoluteError,meanError

def calculateErrorForFile(model_wavs_folder,original_wavs_folder,result_name):
    original_wavs_files = os.listdir(original_wavs_folder)
    if not os.path.exists(result_name):
        os.mkdir(result_name)
    absErrorSum = 0
    meanErrorSum = 0
    for i, name in enumerate(original_wavs_files):
        absError, meanError = calculateError(name,original_wavs_folder+name, model_wavs_folder+name,result_name)
        absErrorSum += absError
        meanErrorSum += meanError

    if(absErrorSum != 0):
        absErrorSum = absErrorSum / len(original_wavs_files)

    if(meanErrorSum != 0):
        meanErrorSum = meanErrorSum / len(original_wavs_files)
    printSummedResults(result_name, absErrorSum, meanErrorSum)

def printToResultFile(string,result_name):
    print(string)
    string = string +'\n'
    log_file = open(result_name+'/result.txt', 'a')
    log_file.write(string)

def printSummedResults(result_name, absError,meanError):

    completeResult = "<=============>Average result of result_name<==============>\n"
    completeResult += "Absolute Error = "+ str(absError) + "\n"
    completeResult += "Mean Error = "+ str(meanError) + "\n"

    printToResultFile(completeResult, result_name)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-m', '--model_wav')
    parser.add_argument('-o', '--original_wav')
    parser.add_argument('-n', '--name')

    args = parser.parse_args()

    calculateErrorForFile(args.model_wav,args.original_wav,args.name)



