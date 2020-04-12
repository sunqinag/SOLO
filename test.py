import os
with open('VOC/train/VOCdevkit/VOC2012/ImageSets/Main/trainval.txt','r') as f:
    print('读取成功')
    f.readlines()
    for i in f:
        print(i)