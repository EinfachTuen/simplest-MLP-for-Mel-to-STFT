import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import multiprocessing as mp
from tempfile import TemporaryFile
from torch.utils.data import DataLoader

class DataSet():
    def __init__(self,training_folder):
        self.training_folder = training_folder
        self.actualLoadedData = []
        self.file_list = os.listdir(self.training_folder)
        self.file_batch_size = 50
        self.steps = int(len(self.file_list) / self.file_batch_size) + 1
        self.steps_per_run = 10
        self.step = 0
        self.shuffle = False

    def loadMelAndStft(self,filename):
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
            if(element > input_overlap_per_side and element <  mel_in.shape[0]-input_overlap_per_side):
                mel_in_with_overlap = []
                for number in range(input_overlap_per_side*2+1):
                    actual_mel_index = element - input_overlap_per_side + number
                    mel_in_with_overlap.append(mel_in[actual_mel_index])
                mel_in_with_overlap = np.asarray(mel_in_with_overlap, dtype=np.float32).flatten()
                stft_in =np.asarray(stft_in, dtype=np.float32)
                mel_and_stft.append([mel_in_with_overlap,stft_in[element]])

        return mel_and_stft

    def readFiles(self,file_list):
        load = []

        for filename in file_list:
            load += self.loadMelAndStft(self.training_folder+filename)

        return load

    def main(self):
        pool_size = mp.cpu_count() * 2

        pool = mp.Pool(processes=pool_size)
        batched_file_list = []

        max_step_this_run = self.step + self.steps_per_run

        for step in range(self.step,max_step_this_run):
            batch_file_list = []
            start_read = step * self.file_batch_size

            end_read = step * self.file_batch_size + self.file_batch_size
            if len(self.file_list) < end_read:
                end_read = len(self.file_list)

            for filename in self.file_list[start_read:end_read]:
                batch_file_list.append(filename)

            batched_file_list.append(batch_file_list)

        self.step = max_step_this_run%self.steps

        result_map = pool.map(self.readFiles, batched_file_list)

        pool.close()  # no more tasks
        pool.join()  # wrap up current tasks
        result_list = []
        for result in result_map:
            result_list += result
        return DataLoader(result_list, batch_size=500, shuffle=self.shuffle)


