import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import numpy as np
import time
import multiprocessing as mp


class SpectrogramLoader():
    def __init__(self,training_folder):
        self.training_folder = training_folder
        self.mel_folder = "./mel/"
        self.stft_folder = "./stft/"
        print("load Data")

    def loadMelAndStft(self,filename):
        mel = np.load(self.mel_folder+filename)
        stft = np.load(self.stft_folder+filename)
        return [mel,stft]

    def readFiles(self,queue,file_list,start,end):
        print("start-read-file")
        print("start ",start)
        print("end ",end)
        print("file_list ",str(len(file_list)))
        load = []
        for filename in file_list[start:end]:
            load += self.loadMelAndStft(filename)
            print("Path: " + filename)
        queue.put(load)
        print("finished")

    def main(self):
        queue = mp.Queue()
        file_list = os.listdir(self.mel_folder)

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


# if __name__ == '__main__':
#     inputConverter = InputConverter("./inWav/")
#     inputConverter.main()

