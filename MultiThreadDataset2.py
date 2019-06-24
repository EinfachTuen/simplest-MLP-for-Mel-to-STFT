import multiprocessing

from torch.utils.data import Dataset
import time
import numpy as np
import threading
import os
import random

class MultiThreadDataset2(Dataset):
    def __init__(self, model_input_folder, model_output_folder,threads):
        self.model_input_folder = model_input_folder
        self.model_output_folder = model_output_folder
        self.input_file_list = os.listdir(model_input_folder)
        self.random = random.Random()
        self.file_number = 0

        self.threads = []
        self.max_threads = threads
        self.data=[]

    def initialize(self):
        for run in range(self.max_threads):
            self.try_update()
        print("initializing dataloader")
        time.sleep(15)

    def load_data(self,file_number):
        input, output = self.load_files(file_number)
        #print('in',input)
        for i,mel in enumerate(input):
            self.data.append([mel,output[i][0],output[i][1],output[i][2]])
            if(len(self.data) > 50000):
             del self.data[0: 1]

    def load_files(self,file_number):
        input_file_name = self.model_input_folder+self.input_file_list[file_number]
        input = np.load(input_file_name)
        output_file_name = input_file_name.replace(self.model_input_folder,self.model_output_folder)
        output = np.load(output_file_name)
        return input,output

    def threadCount(self):
        return len(self.threads)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        self.kill_threads()
        self.try_update()
        try :
            return self.data[idx]
        except:
            print("list index not found exception")
            return self.data[0]

    def try_update(self):
        if (len(self.threads) < self.max_threads):
            thread = threading.Thread(target=self.load_data, kwargs={"file_number": self.file_number})
            self.threads.append(thread)
            thread.start()
            self.file_number = self.random.randint(0, len(self.input_file_list) - 1)

    def kill_threads(self):
        alive_threads = []
        for thread in self.threads:
            if thread.is_alive() == False:
                thread.join()
            else:
                alive_threads.append(thread)
        self.threads = alive_threads

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-mel', '--model_input_folder', default="./mels-training/")
    parser.add_argument('-o', '--model_output_folder', default="./output-training/")

    args = parser.parse_args()

    multiThreadDataset = MultiThreadDataset2(args.model_input_folder,args.model_output_folder)
    multiThreadDataset.initialize()