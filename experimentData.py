import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import multiprocessing as mp
from tempfile import TemporaryFile

loaded_files = []
training_folder = './inWav/'

class DataSet():
    def __init__(self):
        print("load Data")

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

    def readFiles(self,queue,file_list,start,end):
        print("start-read-file")
        print("start ",start)
        print("end ",end)
        print("file_list ",str(len(file_list)))
        load = []
        for filename in file_list[start:end]:
            load += self.loadMelAndStft(training_folder+filename)
            print("Path: " + filename)
        queue.put(load)
        print("finished")

    def main(self):
        queue = mp.Queue()
        file_list = os.listdir(training_folder)

        time_before = time.time()
        processes = []
        file_batch_size = 50
        steps= int(len(file_list)/file_batch_size)+1
        print(steps)
        for file_batch in range(steps):
            print("run",file_batch)
            start_read = file_batch*file_batch_size

            end_read = file_batch*file_batch_size+file_batch_size
            if len(file_list) < end_read:
                end_read = len(file_list)

            process = mp.Process(target=self.readFiles, args=(queue,file_list,start_read,end_read))
            processes.append(process)

        for process in processes:
            print("start process")
            process.start()
        returns = []
        for process in processes:
            ret = queue.get() # will block
            returns += ret
        for process in processes:
            process.join()
            process.join()
        print(len(returns))
        print("time difference: ", str(time.time()-time_before))
        return returns
