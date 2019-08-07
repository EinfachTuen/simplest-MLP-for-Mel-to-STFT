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
#loss_files.append((open("loss_shuffle_training_10files_complex.txt", "r"),"loss_shuffle_training_10files_complex"))
loss_files.append((open("loss_shuffle_training_13200files_complex.txt", "r"),"loss_shuffle_training_13200files_complex"))
#loss_files.append((open("loss-h5-medium-training.txt", "r"),"loss-h5-medium-training.txt"))
#loss_files.append((open("loss-elu.txt", "r"),"loss-relu-more-data"))
#loss_files.append((open("loss-relu-larger-data.txt", "r"),"loss-relu-larger-data.txt"))

file_number = 0
max_iter=999999
iter_resolution = 1
#plt.xscale('log')
for loss_file,name in loss_files:
    lineNumber = 0
    allImagX = []
    allImagY = []
    allRealX = []
    allRealY = []
    allMagX = []
    allMagY = []
    allAvgX = []
    allAvgY = []
    for line in loss_file:
        if lineNumber % iter_resolution == 0 and lineNumber < max_iter:
            print(lineNumber)
            imag = line.split(',')[1]
            imag = imag.replace("imag: ","")
            real = line.split(',')[2]
            real = real.replace("real: ","")
            magnitudes = line.split(',')[3]
            magnitudes = magnitudes.replace("magnitudes: ","")
            average = line.split(',')[4]
            average = average.replace("average: ","")
            allImagX.append(lineNumber)
            allImagY.append(float(imag))
            allRealX.append(lineNumber)
            allRealY.append(float(real))
            allMagX.append(lineNumber)
            allMagY.append(float(magnitudes))
            allAvgX.append(lineNumber)
            allAvgY.append(float(average))
        lineNumber += 1

    plt.plot(allImagX, allImagY, label = 'imag')
    plt.plot(allRealX, allRealY, label = 'real')
    plt.plot(allMagX, allMagY, label = 'mag')
    plt.plot(allAvgX, allAvgY, label = 'avg')
    file_number += 1

print('finished')
plt.legend()
plt.ylabel('Loss Graph')
plt.ylim(bottom=0, top=2)
plt.savefig('fig1.png', dpi = 600)
plt.show()