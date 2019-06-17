import os
import threading
import numpy as np
import librosa
import time
from torch.utils.data import Dataset
import multiprocessing
import random
import torch
from scipy.io.wavfile import read
import sys
sys.path.insert(0, 'tacotron2')
from tacotron2.layers import TacotronSTFT
import gc

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
        self.max_threads = multiprocessing.cpu_count() - 1
        self.MAX_WAV_VALUE = 32768.0
        self.sampling_rate = 22050

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
        try :
            return self.data[idx]
        except:
            print("list index not found exception")
            return self.data[0]

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

        if(len(self.data) > 50000):
            del self.data[0: len(mel_and_stft)]
        self.data += mel_and_stft

    def loadMelAndStft(self,filename):
        audio = self.readAudio(filename)
        loadedMel = self.loadMel(audio)


       # wav, sr = librosa.load(filename)
       # stft_in = librosa.stft(wav, n_fft=1024, hop_length=256, win_length=1024)
       # mel_in = librosa.feature.melspectrogram(S=stft_in)
       # stft_in = np.abs(stft_in)

        #mel_in = np.array(mel_in)
        imag, real, magnitudes = self.getOuput(audio)

        loadedMel = np.swapaxes(loadedMel, 0, 1)
        imag = np.swapaxes(imag[0], 0, 1)
        real = np.swapaxes(real[0], 0, 1)
        magnitudes = np.swapaxes(magnitudes[0], 0, 1)

        mel_and_stft = []
        input_overlap_per_side = 7
        imag = np.asarray(imag, dtype=np.float32)
        real = np.asarray(real, dtype=np.float32)
        magnitudes = np.asarray(magnitudes, dtype=np.float32)

        for element in range(loadedMel.shape[0]):
            if (element > input_overlap_per_side and element < loadedMel.shape[0] - input_overlap_per_side):
                mel_in_with_overlap = []
                for number in range(input_overlap_per_side * 2 + 1):
                    actual_mel_index = element - input_overlap_per_side + number
                    mel_in_with_overlap.append(loadedMel[actual_mel_index])
                mel_in_with_overlap = np.asarray(mel_in_with_overlap, dtype=np.float32).flatten()
                mel_and_stft.append([mel_in_with_overlap, imag[element], real[element], magnitudes[element]])
        return mel_and_stft

# ================================================================================================
# ================================================================================================
# ================================================================================================
    def getOuput(self,audio):
        stft = TacotronSTFT(filter_length=1024,
                            hop_length=256,
                            win_length=1024,
                            sampling_rate=22050,
                            mel_fmin=0.0, mel_fmax=8000.0)
        return  stft.getOutput(audio)

    def readAudio(self,filename):
        # Read audio
        audio, sampling_rate = self.load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("{} SR doesn't match target {} SR".format(
                sampling_rate, self.sampling_rate))
        audio_norm = audio / self.MAX_WAV_VALUE
        audio_norm = audio_norm.unsqueeze(0)
        audio_norm = torch.autograd.Variable(audio_norm, requires_grad=False)
        return audio_norm

    def loadMel(self, audio):

        mel = self.get_mel(audio)

       # audio = audio / self.MAX_WAV_VALUE

        return mel.numpy()

    def load_wav_to_torch(self,full_path):
        """
        Loads wavdata into torch array
        """
        sampling_rate, data = read(full_path)
        return torch.from_numpy(data).float(), sampling_rate

    def get_mel(self, audio):
        stft = TacotronSTFT(filter_length=1024,
                     hop_length=256,
                     win_length=1024,
                     sampling_rate=22050,
                     mel_fmin=0.0, mel_fmax=8000.0)
        melspec = stft.mel_spectrogram(audio)
        melspec = torch.squeeze(melspec, 0)

        return melspec