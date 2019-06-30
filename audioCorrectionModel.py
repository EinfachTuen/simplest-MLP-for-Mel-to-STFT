from scipy.io.wavfile import write
from scipy.io.wavfile import read
from torch.utils.data import Dataset
import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader

class AudioDataset(Dataset):
    def __init__(self, input_folder,original_folder,wav_vector_size):
        self.data = []
        self.input_folder = input_folder
        self.original_folder = original_folder
        self.wav_vector_size = wav_vector_size

    def initialize(self):
        file_list = os.listdir(self.input_folder)
        for i, file_name in enumerate(file_list):
            sr, wav_data = read(self.input_folder+file_name)
            sr, wav_original_data = read(self.original_folder+self.getOriginalFilename(file_name))
            wav_data = np.asarray(wav_data, dtype='float32')*1000
            wav_original_data = np.asarray(wav_original_data, dtype='float32')

            wav_data = np.split(wav_data, range(self.wav_vector_size, wav_data.shape[0] , self.wav_vector_size))
            wav_original_data = np.split(wav_original_data, range(self.wav_vector_size, wav_original_data.shape[0] , self.wav_vector_size))

            for data_part in range(len(wav_data)):
                if (wav_data[data_part].shape[0] == self.wav_vector_size):
                    self.data.append([wav_data[data_part],wav_original_data[data_part]])

    def getOriginalFilename(self, file_name):
        important_file_name_path = file_name.split('_')[1]
        important_file_name_path = important_file_name_path[:-4]
        return important_file_name_path

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
       return self.data[idx]

class WavRegressionModel(nn.Module):
    def __init__(self, input_dim,hidden_layer_factor):
        super(WavRegressionModel, self).__init__()
        hidden_layer_size = input_dim * hidden_layer_factor

        self.sequencial =nn.Sequential(
            nn.Linear(input_dim, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.ReLU(),
            nn.Linear(hidden_layer_size, input_dim),
        )

    def forward(self, x):
        r = self.sequencial(x)
        return r


class Training():
    def __init__(self):
        self.wav_vector_size = int(22050 / 10)
        self.audioDataset = AudioDataset("./test-result-wav/", "./inWav/",self.wav_vector_size)
        self.epochs = 10000000
        self.model =WavRegressionModel(self.wav_vector_size,3).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        self.criterion = nn.L1Loss()

    def runTraining(self):
        self.audioDataset.initialize()
        dataLoader = DataLoader(self.audioDataset,batch_size=500,shuffle=True)
        loss_list = []
        iteration = 0
        for epoch in range(self.epochs):
            for i, (input,original) in enumerate(dataLoader):
                input = input.cuda()
                original = original.cuda()

                self.optimizer.zero_grad()

                model_output = self.model(input)

                loss = self.criterion(model_output, original)

                loss.backward()
                self.optimizer.step()

                loss_list.append(loss.data.cpu().numpy())
                iteration = iteration+1
                if(iteration%100 == 0):
                    average_loss = np.average(np.array(loss_list))
                    print(average_loss)
                    loss_list = []


t = Training()
t.runTraining()