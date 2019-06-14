import os
import threading
import numpy as np
import librosa
import time
from torch.utils.data import Dataset
import multiprocessing
import random

class AudioDataset(Dataset):
    def __init__(self, training_folder):
        self.training_folder = training_folder
        if training_folder != "":
            self.file_list = os.listdir(training_folder)
        else: self.file_list = []
        self.data=[]
        self.threads = []
        self.file_number = 0
        self.random = random.Random()
        self.max_threads = multiprocessing.cpu_count() -1


    def initialize(self):
        for run in range(self.max_threads):
            self.try_update()
        print("initializing dataloader")
        time.sleep(15)

    def threadCount(self):
        return len(self.threads)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.kill_threads()
        self.try_update()
        return self.data[idx]

    def kill_threads(self):
        alive_threads = []
        for thread in self.threads:
            if thread.is_alive() == False:
                thread.join()
            else: alive_threads.append(thread)
        self.threads = alive_threads

    def try_update(self):
        if(len(self.threads) < self.max_threads):
            thread = threading.Thread(target=self.loadDataForFilenumber,kwargs={"file_number": self.file_number})
            self.threads.append(thread)
            thread.start()
            self.file_number = self.random.randint(0,len(self.file_list)-1)

    def loadDataForFilenumber(self, file_number):
        filename = self.training_folder + self.file_list[file_number]
        mel_and_stft = self.loadMelAndStft(filename)

        if(len(self.data) > 2000000):
            del self.data[0: len(mel_and_stft)]
        self.data += mel_and_stft


    def loadMelAndStft(self,filename):
        wav, sr = librosa.load(filename)
        stft_in = librosa.stft(wav)
        mel_in = np.abs(librosa.feature.melspectrogram(S=stft_in))
        stft_in = np.array(stft_in)
        mel_in = np.array(mel_in)

        mel_in = np.swapaxes(mel_in, 0, 1)
        stft_in = np.swapaxes(stft_in, 0, 1)

        mel_and_stft = []
        input_overlap_per_side = 3
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





