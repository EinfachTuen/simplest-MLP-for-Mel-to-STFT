import os
import threading
import numpy as np
import librosa
import sys
import time

class AudioDataset:
    def __init__(self, training_folder):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.training_folder = training_folder
        self.file_list = os.listdir(training_folder)
        self.data=[]
        self.threads = []
        self.file_number = 0

    def initialize(self):
        for run in range(50):
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
        if(len(self.threads) < 6):
            thread = threading.Thread(target=self.loadMelAndStft,kwargs={"file_number": self.file_number})
            self.threads.append(thread)
            thread.start()
            self.file_number += 1
            self.file_number = self.file_number % len(self.file_list)

    def loadMelAndStft(self, file_number):
        filename = self.training_folder + self.file_list[file_number]
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

        if(len(self.data) > 15000):
            del self.data[0: len(mel_and_stft)]
        self.data += mel_and_stft


if __name__ == "__main__":
    audioSet = AudioDataset("./inWav/")
    #audioSet.loadMelAndStft(0)
    audioSet.initialize()

    while(True):
        audioSet.__getitem__(5)
        print(audioSet.__len__())
        print(audioSet.threadCount())




