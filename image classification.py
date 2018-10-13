import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

train_data = '/Users/yxk/Documents/Python/601/Miniproject2/Train'
test_data = '/Users/yxk/Documents/Python/601/Miniproject2/Test'

def rename(dir_path, labelname):
    '''
    Actually we don't necessarily need this function,
    but I decide to keep it since it costed me some time to build
    '''
    filelist = os.listdir(dir_path)
    for cnt in range(len(filelist)):

        picnum = "%04d" % cnt
        # set the old file name
        oldname = dir_path + '/' + filelist[cnt]
        # set the new file name
        newname = dir_path + '/' + labelname + '.' + picnum + '.jpg'
        # rename the file
        os.rename(oldname, newname)

def one_hot(labellist, label):

    labelset = [0] * len(labellist)
    labelset[labellist.index(label)] = 1
    ohl = np.array(labelset)
    return ohl

def getlabel(data_path):

    images = []
    foldername = [element for element in os.listdir(data_path) if element != '.DS_Store']

    for folder in foldername:

        img_path = os.path.join(data_path, folder)
        imgnames = os.listdir(img_path)
        imglabel = one_hot(foldername, folder)

        for imgname in imgnames:

            if imgname == '.DS_Store':
                continue

            path = os.path.join(img_path, imgname)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            try:
                img = cv2.resize(img, (128, 128))
            except:
                print(imgname, 'This image may have some problems, please check it')
                quit()
            images.append([np.array(img), imglabel])

    shuffle(images)
    return images

def flatten(data):
    img_data = np.array([i[0] for i in data]).reshape(-1, 128, 128, 1)
    lbl_data = np.array([i[1] for i in data])
    return img_data, lbl_data

def convmodel(training_image, training_label):
    '''
    build and train a 3 convolutional layers neural network
    '''
    dir = [element for element in os.listdir(train_data) if element != '.DS_Store']
    classes = len(dir)

    model = Sequential()

    model.add(InputLayer(input_shape = [128, 128, 1]))
    model.add(Conv2D(filters = 32, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 50, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Conv2D(filters = 80, kernel_size = 5, strides = 1, padding = 'same', activation = 'relu'))
    model.add(MaxPool2D(pool_size = 5, padding = 'same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1024, activation = 'relu'))
    model.add(Dropout(rate = 0.5))
    model.add(Dense(classes, activation = 'softmax'))
    optimizer = Adam(lr = 5e-4)

    model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x = training_image, y = training_label, epochs = 10, batch_size = 64)
    model.summary()
    return model

def main():

    train_img = getlabel(train_data)
    test_img = getlabel(test_data)

    tr_img_data, tr_lbl_data  = flatten(train_img)
    tst_img_data,tst_lbl_data = flatten(test_img)

    model = convmodel(tr_img_data, tr_lbl_data)

    fig = plt.figure(figsize = (10, 10))
    labelname = [element for element in os.listdir(train_data) if element != '.DS_Store']

    for cnt, data in enumerate(test_img[10:40]):

        y = fig.add_subplot(6, 5, cnt+1)
        img = data[0]
        data = img.reshape(-1, 128, 128, 1)
        model_out = model.predict([data])
        str_label = labelname[np.argmax(model_out)]

        y.imshow(img, cmap = 'gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)

    plt.subplots_adjust(wspace=1, hspace=1)
    plt.show()


if __name__ == '__main__':
    main()
