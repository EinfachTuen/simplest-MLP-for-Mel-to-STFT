import matplotlib.pyplot as plt
import numpy as np
loss_files =[]
#loss_files.append(open("loss-log.txt", "r"))
#loss_files.append(open("loss-log2.txt", "r"))
#loss_files.append((open("loss-local-new-dataloader.txt", "r"),"loss-relu-new-dataloader"))
#loss_files.append((open("loss-softsign-data.txt", "r"),"loss-softsign-data"))
#loss_files.append((open("loss-newDataloader-10h.txt", "r"),"loss-online-10h-new-Dataloader-Relu"))
#loss_files.append((open("loss-reluMoreData.txt", "r"),"loss-reluMoreData.txt"))
#loss_files.append((open("loss-stft-in-improved.txt", "r"),"loss-stft-in-improved.txt"))
#loss_files.append((open("loss-seq-b500-h5-500files.txt", "r"),"loss-seq-b500-h5-500files.txt"))
#loss_files.append((open("loss-seq-b500-h5-1-500files.txt", "r"),"loss-seq-b500-h5-1-500files.txt"))
#loss_files.append((open("loss-seq-b500-h2-1-695fmix.txt", "r"),"loss-seq-b500-h2-1-695fmix.txt"))
#loss_files.append((open("loss-ip1-seq-b500-h2-1-695fmix.txt", "r"),"loss-ip1-seq-b500-h2-1-695fmix.txt"))
#loss_files.append((open("loss-ip1-f32-seq-b500-h2-1-695fmix.txt", "r"),"loss-ip1-f32-seq-b500-h2-1-695fmix.txt"))
#loss_files.append((open("loss-r2-ip1-f32-seq-b500-h2-1-695fmix.txt", "r"),"loss-r2-ip1-f32-seq-b500-h2-1-695fmix.txt"))
#loss_files.append((open("loss-r1-ip1-seq-b500-h2-1-695fmix.txt", "r"),"loss-r1-ip1-seq-b500-h2-1-695fmix.txt"))
#loss_files.append((open("loss-r1-npabs-ip1-seq-b500-h2-1-695fmix.txt", "r"),"loss-r1-npabs-ip1-seq-b500-h2-1-695fmix.txt"))
#loss_files.append((open("loss-r1-npabs2-ip1-seq-b500-h2-1-695fmix.txt", "r"),"loss-r1-npabs2-ip1-seq-b500-h2-1-695fmix.txt"))
loss_files.append((open("loss_not_shuffle_training_13200files_complex", "r"),"loss_not_shuffle_training_13200files_complex.txt"))
#loss_files.append((open("loss_full-LJ-RELU-full-Abs.txt", "r"),"loss_full-LJ-RELU-full-Abs.txt"))
#loss_files.append((open("loss-h5-medium-training.txt", "r"),"loss-h5-medium-training.txt"))
#loss_files.append((open("loss-elu.txt", "r"),"loss-relu-more-data"))
#loss_files.append((open("loss-relu-larger-data.txt", "r"),"loss-relu-larger-data.txt"))

file_number = 0
for loss_file,name in loss_files:
    lineNumber = 0
    allX = []
    allY = []
    for line in loss_file:
        value = line.split(',')[1]
        lineNumber +=1
        allX.append(lineNumber)
        allY.append(float(value))
        if lineNumber %100 == 0:
            print(lineNumber)
    plt.plot(allX, allY, label = name)
    file_number += 1

print('finished')
plt.legend()
plt.ylabel('Loss Graph')
plt.ylim(bottom=0, top=0.5)
plt.show()