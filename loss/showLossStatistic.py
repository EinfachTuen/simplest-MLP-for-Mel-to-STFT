import matplotlib.pyplot as plt
import numpy as np
loss_files =[]
#loss_files.append(open("loss-log.txt", "r"))
#loss_files.append(open("loss-log2.txt", "r"))
loss_files.append(open("loss-local-new-dataloader.txt", "r"))



allLines =[]
lineNumber = 0
allX = []
allY = []

for loss_file in loss_files:
    for line in loss_file:
        value = line.split(',')[1]
        lineNumber +=1
        if (float(value) < 2):
            allX.append(lineNumber)
            allY.append(float(value))
        if lineNumber %100 == 0:
            print(lineNumber)
    plt.plot(allX, allY)

print('finished')

plt.ylabel('Loss Graph')
plt.show()