import os
from os.path import basename
import random
# 1: aeroplane
# 2: bicycle
# 3: bird
# 4: boat
# 5: bottle
# 6: bus
# 7: car
# 8: cat
# 9: chair
# 10: cow
# 11: diningtable
# 12: dog
# 13: horse
# 14: motorbike
# 15: person
# 16: pottedplant
# 17: sheep
# 18: sofa
# 19: train
# 20: tvmonitor

ratio = 0.01

in_file_list = open('train.txt')
lines = in_file_list.readlines()

random.shuffle(lines)
lines = lines[0:int(ratio*len(lines))]

out_file = open("train_01_random.txt",'w')
for i in lines:
    out_file.write(str(i))

out_file.close()
