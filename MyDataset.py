import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import librosa

class MyDataset(Dataset):
    def __init__(self):
        self.training_folder = './inWav/'
        self.data_files = os.listdir(self.training_folder)
        self.actual_file_index = 0
        self.actual_content =[]

    def __getitem__(self, idx):
        actual_file_index =+ 1


        return

    def load_next_file(self):
        self.actual_content.append(self.load_file(self.training_folder + self.data_files[self.actual_file_index]))


    @staticmethod
    def load_file(filename):
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
        return mel_and_stft

    def __len__(self):
        return len(self.data_files)