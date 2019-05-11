import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
import librosa

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        print("input_dim", input_dim)
        print("output_dim", output_dim)
        hidden_layer_size = input_dim*15
        second_hidden_layer_size = input_dim*15
        self.linear1 = nn.Linear(input_dim, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, second_hidden_layer_size)
        self.linear4 = nn.Linear(second_hidden_layer_size, output_dim)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        out = f.relu(self.linear4(x))
        return out

class DataPrep():
    def __init__(self):
        mel_and_stft = self.loadMelAndStft()
        self.dataloader = DataLoader(mel_and_stft,
                                batch_size=1,
                                shuffle=False)

    def loadMelAndStft(self):
        wav, sr = librosa.load("./inWav/arctic_indian_man16.wav")
        stft_in = np.abs(librosa.stft(wav)) ** 2
        mel_in = librosa.feature.melspectrogram(S=stft_in)
        stft_in = np.array(stft_in)
        mel_in = np.array(mel_in)

        mel_in = np.swapaxes(mel_in, 0, 1)
        stft_in = np.swapaxes(stft_in, 0, 1)

        print("-----------------------------------")
        print("stft_in Size:", stft_in.shape)

        print("-----------------------------------")
        print("mel_in Shape:", mel_in.shape)
        print("mel_in Size:", mel_in.shape[1])
        mel_and_stft = []
        input_overlap_per_side = 3
        for element in range(mel_in.shape[0]):
            mel_in_with_overlap = []
            for number in range(input_overlap_per_side*2+1):
                actual_mel_index = element - input_overlap_per_side + number
                if -1 < actual_mel_index < mel_in.shape[0]:
                    mel_in_with_overlap.append(mel_in[actual_mel_index])
                else: mel_in_with_overlap.append(np.zeros(mel_in.shape[1]))
            mel_in_with_overlap = np.asarray(mel_in_with_overlap, dtype=np.float32).flatten()
            stft_in =np.asarray(stft_in,dtype=np.float32)
            mel_and_stft.append([mel_in_with_overlap,stft_in[element]])
        return mel_and_stft

class Training():
    def __init__(self, dataloader):
        self.epochs = 100000

        learning_rate = 0.00001
        model = LinearRegressionModel(7 * 128, 1025).cuda()

        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        last_loss = 999999
        for epoch in range(self.epochs):
            loss_list = []
            for i,(mel,stft) in enumerate(dataloader):
                mel = mel.cuda()
                stft = stft.cuda()
                optimizer.zero_grad()
                output_model = model(mel)
                loss = criterion(output_model, stft)
                loss.backward()
                loss_list.append(loss.data.cpu().numpy())
                optimizer.step()

            loss_np = np.array(loss_list)
            average_loss = np.average(loss_np)
            print('epoch {}, loss {}'.format(epoch,average_loss))

            name= "MLP1-refactored_"
            log_file = open('loss_log.txt', 'a')
            log_file.write(name+str(epoch) + "," + "{:.4f}".format(np.average(loss_np)) + ',\n')
            if (epoch % 1000) == 99:
                torch.save(model, name + str(epoch))
            if (epoch % 200) == 199:
                if average_loss >= last_loss :
                    learning_rate *=-0.5
                    print("learning_rate changed to"+learning_rate)
                last_loss = average_loss
#
# class GenerateAudioFromMel():
#     def __init__(self, dataloader):
#         for i, (mel, stft) in enumerate(dataloader):




Training(DataPrep().dataloader)