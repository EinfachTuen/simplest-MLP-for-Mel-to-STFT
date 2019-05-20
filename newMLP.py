import torch
import torch.nn as nn
import torch.nn.functional as f
from torch.autograd import Variable
import numpy as np
from torch.utils.data import DataLoader
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
from random import shuffle


class StateClass():
    def __init__(self,first_hidden_layer_factor,second_hidden_layer_factor):
        self.epochs = 100000
        self.learning_rate = 0.001
        self.model_input_size = 7 * 128
        self.model_output_size = 1025
        self.last_loss = 999999
        self.dataloaders = None
        self.single_dataloader = None
        self.first_hidden_layer_factor = first_hidden_layer_factor
        self.second_hidden_layer_factor = second_hidden_layer_factor
        self.model = LinearRegressionModel(self.model_input_size, self.model_output_size,first_hidden_layer_factor,second_hidden_layer_factor).cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        self.modelname= "MLP-ADAM-MSE-10Hidden"
        self.model_to_load= ""
        self.model_storage =""
        self.single_file = "./inWav/16kLJ001-0001.wav"
        self.training_folder= "./inWav/"
        self.file_iterations_per_loss = None
        self.epochs_per_save = None
        self.epochs_per_learning_change = 500
        self.result_filename = "result_audio"
        self.normalization_test_filename = "normalization_test_audio"
        self.sampling_rate = 22050
        self.lossfile = 'loss_log.txt'
        self.debug = False


    def do_inference(self):
        self.single_dataloader = DataPrep.loadFile(None,self,False,True,state.single_file)
        generateAudioFromMel = GenerateAudioFromMel()
        GenerateAudioFromMel.load_and_inference_and_convert(generateAudioFromMel,self)
        print("-----------------------------------")
        print("inference finished")
        print("-----------------------------------")


    def run_training(self):
        DataPrep.loadFolder(None,self)
        Training(self)

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim,first_hidden_layer_factor,second_hidden_layer_factor):
        super(LinearRegressionModel, self).__init__()
        print("input_dim", input_dim)
        print("output_dim", output_dim)
        hidden_layer_size = input_dim * first_hidden_layer_factor
        second_hidden_layer_size = input_dim * second_hidden_layer_factor
        self.linear1 = nn.Linear(input_dim, hidden_layer_size)
        self.linear2 = nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = nn.Linear(hidden_layer_size, second_hidden_layer_size)
        self.linear4 = nn.Linear(second_hidden_layer_size, output_dim)

    def forward(self, x):
        x = f.relu(self.linear1(x))
        x = f.relu(self.linear2(x))
        x = f.relu(self.linear3(x))
        out = self.linear4(x)
        return out

class Training():
    def __init__(self,state):
        file_number = 0
        loss_list = []
        for epoch in range(state.epochs):
            for dataloader in state.dataloaders:
                file_number +=1
                for i,(mel,stft) in enumerate(dataloader):
                    mel = mel.cuda()
                    stft = stft.cuda()
                    state.optimizer.zero_grad()
                    output_model = state.model(mel)

                    loss = state.criterion(output_model, stft)
                    loss.backward()
                    loss_list.append(loss.data.cpu().numpy())
                    state.optimizer.step()

                if (file_number % state.epochs_per_save) == (state.epochs_per_save - 1):
                    print("===> model saved", state.model_storage + state.modelname + str(file_number))
                    torch.save(state.model.state_dict(), state.model_storage + state.modelname + str(file_number))
                loss_np = np.array(loss_list)
                average_loss = np.average(loss_np)
                print('file_number {}, loss {}'.format(file_number, average_loss))
                log_file = open(state.lossfile, 'a')
                log_file.write(state.modelname + str(file_number) + "," + "{:.4f}".format(np.average(loss_np)) + ',\n')
                loss_list = []
            print("---------------------------------------------------------")
            print("------------------epoch:"+ str(epoch)+"-----------------------")

            if (epoch % state.epochs_per_learning_change) == state.epochs_per_learning_change-1:
                if average_loss >= state.last_loss :
                    state.learning_rate =int(state.learning_rate *-0.5)
                    print("learning_rate changed to"+str(state.learning_rate))
                    state.last_loss = average_loss

class DataPrep():
    def loadFolder(self, state):

        loaded_files = []
        for filename in os.listdir(state.training_folder):
            loaded_files.append(DataPrep.loadFile(self,state,True,False,""+state.training_folder+filename))
            print("Path: "+state.training_folder+filename)
        state.dataloaders = loaded_files

    def loadFile(self,state,shuffle,should_plot,filename):
        mel_and_stft = DataPrep.loadMelAndStft(self,state,filename,should_plot)
        return  DataLoader(mel_and_stft,
                                batch_size=1,
                                shuffle=shuffle)

    def loadMelAndStft(self,state,filename,should_plot):
        wav, sr = librosa.load(filename)
        state.sampling_rate = sr
        stft_in = librosa.stft(wav)
        mel_in = librosa.feature.melspectrogram(S=stft_in)
        stft_in = np.array(stft_in)
        mel_in = np.array(mel_in)
        if state.debug:
            GenerateAudioFromMel.stft_to_audio(None,state,stft_in,"Original stft",state.normalization_test_filename)
            print("sample rate", sr)
            stft_min = np.min(stft_in)
            stft_max = np.max(stft_in)
            mel_min = np.min(mel_in)
            mel_max = np.max(mel_in)
            print("stft_min",stft_min)
            print("stft_max",stft_max)
            print("mel_min",mel_min)
            print("mel_max",mel_max)
            print("-----------------------------------")
            print("stft_in Size:", stft_in.shape)

            print("-----------------------------------")
            print("mel_in Shape:", mel_in.shape)

        mel_in = np.swapaxes(mel_in, 0, 1)
        stft_in = np.swapaxes(stft_in, 0, 1)

        mel_and_stft = []
        input_overlap_per_side = 3
        for element in range(mel_in.shape[0]):
            if(element > input_overlap_per_side and element <  mel_in.shape[0]-input_overlap_per_side):
                mel_in_with_overlap = []
                for number in range(input_overlap_per_side*2+1):
                    actual_mel_index = element - input_overlap_per_side + number
                    mel_in_with_overlap.append(mel_in[actual_mel_index])
                mel_in_with_overlap = np.asarray(mel_in_with_overlap, dtype=np.float32).flatten()
                stft_in =np.asarray(stft_in,dtype=np.float32)
                mel_and_stft.append([mel_in_with_overlap,stft_in[element]])
        return mel_and_stft

class GenerateAudioFromMel:
    def load_and_inference_and_convert(self, state):
        stft_list = []
        state.model = LinearRegressionModel(state.model_input_size, state.model_output_size,state.first_hidden_layer_factor, state.second_hidden_layer_factor)
        print("load model", state.model_to_load)
        state.model.load_state_dict(torch.load(state.model_to_load))
        state.model.cuda()
        state.model.eval()

        for i, (mel,stft) in enumerate(state.single_dataloader):
            mel = mel.cuda()
            stft_part = state.model(mel).cpu().detach().numpy()
            stft_list.append(stft_part[0])

        stft_list = np.asarray(stft_list)
        stft_list = np.swapaxes(stft_list, 0, 1)
        GenerateAudioFromMel.stft_to_audio(self,state,stft_list,"Result STFT",state.result_filename)
        print(stft_list.shape)

    def stft_to_audio(self,state,stft,diagramm_name,filename):
        wav = librosa.istft(stft)
        librosa.output.write_wav(state.modelname+'_'+filename+".wav",wav,state.sampling_rate)
        GenerateAudioFromMel.plotSTFT(self,stft,diagramm_name)


    def plotSTFT(self,stft,title):
        plt.figure(figsize=(12, 8))
        plt.subplot(4, 2, 1)
        D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-t', "--training", dest='training', action='store_true')
    parser.add_argument('-d', "--debug", dest='debug', action='store_true')
    parser.add_argument('-i', '--inference', dest='inference', action='store_true')
    parser.add_argument('-tf', '--trainingFolder', default="./inWav/")
    parser.add_argument('-m', '--modelname', default="MLP-ADAM-MSE-10H")
    parser.add_argument('-sf', '--single_file', default="./reserveWav/arctic_indian_man16.wav")
    parser.add_argument('-eps', '--epochsPerSave', default=30,type=int)
    parser.add_argument('-lr', '--learningRate', default=0.001,type=float)
    parser.add_argument('-ms', '--modelStorage', default="")
    parser.add_argument('-c', '--modelCheckpoint', default="")
    parser.add_argument('-h1f','--firstHiddenlayer', default=5,type=int)
    parser.add_argument('-h2f','--secondHiddenlayer', default=5,type=int)
    parser.add_argument('-lf','--lossfile', default="log_loss.txt")

    parser.set_defaults(debug=False)
    parser.set_defaults(training=False)
    parser.set_defaults(inference=False)
    args = parser.parse_args()

    state = StateClass(args.firstHiddenlayer,args.secondHiddenlayer)
    print('Model Storage', args.modelStorage)
    state.training_folder = args.trainingFolder
    state.modelname= args.modelname
    state.single_file= args.single_file
    state.epochs_per_save= args.epochsPerSave
    state.learningRate= args.learningRate
    state.lossfile= args.lossfile
    state.model_storage= args.modelStorage
    state.debug= args.debug

    if args.modelCheckpoint != "":
        if args.training:
            state.model.load_state_dict(torch.load(args.modelCheckpoint))
            state.model.cuda()
            state.model.eval()
        if args.inference:
            state.model_to_load = args.modelCheckpoint
    if args.training:
        state.run_training()
    if args.inference:
        state.do_inference()
