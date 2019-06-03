import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import multiprocessing as mp
import random
import torch
import torch.nn as nn


class subprocess:
    def __init__(self,queue,file_list,start_read,end_read,training_folder):
        self.training_folder = training_folder
        self.queue = queue
        self.file_list = file_list
        self.start_read = start_read
        self.end_read = end_read
        self.readFiles()

    def loadMelAndStft(self, filename):
        wav, sr = librosa.load(filename)
        stft_in = librosa.stft(wav)
        mel_in = librosa.feature.melspectrogram(S=stft_in)
        stft_in = np.array(stft_in)
        mel_in = np.array(mel_in)

        mel_in = np.swapaxes(mel_in, 0, 1)
        stft_in = np.swapaxes(stft_in, 0, 1)

        mel_and_stft = []
        input_overlap_per_side = 1
        for element in range(mel_in.shape[0]):
            if (element > input_overlap_per_side and element < mel_in.shape[0] - input_overlap_per_side):
                mel_in_with_overlap = []
                for number in range(input_overlap_per_side * 2 + 1):
                    actual_mel_index = element - input_overlap_per_side + number
                    mel_in_with_overlap.append(mel_in[actual_mel_index])
                mel_in_with_overlap = np.asarray(mel_in_with_overlap, dtype=np.float32).flatten()
                stft_in = np.asarray(stft_in, dtype=np.float32)
                mel_and_stft.append([mel_in_with_overlap, stft_in[element]])
        return mel_and_stft

    def readFiles(self):
        print("start-read-file")
        print("start ",self.start_read)
        print("end ",self.end_read)
        print("file_list ",str(len(self.file_list)))
        loaded =[]
        for filename in self.file_list[self.start_read:self.end_read]:
            loaded.append(self.loadMelAndStft(self.training_folder+filename))
            #print("Path: " + filename)
        self.queue.put(loaded)
        print("readFiles finished")

class data:
    def __init__(self):
        self.training_folder = './inWav/'
        self.file_list = os.listdir(self.training_folder)
        self.time_before = time.time()
        self.step = 0
        self.file_batch_size = 100
        self.queue = mp.Queue()
        self.next = None
        self.process = None
        self.learning_batch_size = 500

    def startFileLoaderProcess(self):
        self.startNextDataStepProcess()
        self.process.join()
        print("startFileLoaderProcess finished")

    def startNextDataStepProcess(self):
        start_read = self.step * self.file_batch_size
        end_read = self.step * self.file_batch_size + self.file_batch_size

        self.step = self.step + 1
        if len(self.file_list) < end_read:
            end_read = len(self.file_list)
            self.step = 0
            random.shuffle(self.file_list)
        self.process = mp.Process(target=subprocess,
                             args=(self.queue, self.file_list, start_read, end_read, self.training_folder))
        self.process.start()
        self.next = self.queue.get()

    def giveData(self):
        print("start given data")
        if self.process != None:
            self.process.join()
        actual = self.next

        result = []
        for file in actual:
            for element in file:
                result.append(element)
        random.shuffle(result)
        batches = []
        stft_batch = []
        mel_batch= []
        for i in range(len(result)):
            if i != 0 and i % 500 == 0:
                stft_batch = torch.from_numpy(np.asarray(stft_batch))
                mel_batch = torch.from_numpy(np.asarray(mel_batch))
                batches.append((stft_batch,mel_batch))
                stft_batch = []
                mel_batch =[]
            (mel,stft) = result[i]
            stft_batch.append(stft)
            mel_batch.append(mel)
        self.startNextDataStepProcess()
        return batches

if __name__ == '__main__':
    data = data()
    data.startFileLoaderProcess()
    print("actual len 1: ",len(data.giveData()))
    print("actual len 2: ",len(data.giveData()))
    print("actual len 3: ",len(data.giveData()))
    print("actual len 4: ",len(data.giveData()))
    print("actual len 5: ",len(data.giveData()))
    print("actual len 6: ",len(data.giveData()))
    print("actual len 7: ",len(data.giveData()))



