import cv2
import os
import random
import numpy as np
import json

#------- to load the hackathon data from local folder
def load_hack_data():
    #---- training data
    images_tr = [];labels_tr =[]  
    folder='training/background/'
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images_tr.append(img)
            labels_tr.append(0)
    folder='training/hi/'
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images_tr.append(img)
            labels_tr.append(1)
    c = list(zip(images_tr, labels_tr))
    random.shuffle(c)
    a, b = zip(*c)
    train_ims = np.array(a)
    train_labels = np.array(b)

    #---- test data
    images_test = [];labels_test=[]     
    folder='test/'
    for i in range(99):
        filename=(str(i)+'.jpg')
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images_test.append(img)
    test_ims = np.array(images_test)

    labels_test=np.loadtxt('test_labels.txt')
    test_labels = np.array(labels_test.astype(int))

    return train_ims,train_labels,test_ims,test_labels

#----------- from the hackathon drive to json
def write_json(filename, result):
    with open(filename, 'w') as outfile:
        json.dump(result, outfile)

def read_json(filename):
    with open(filename, 'r') as outfile:
        data =  json.load(outfile)
    return data

def generate_sample_file(filename,labels_predicted):
    res = {}
    for i in range(1,99):
        test_set = str(i) + '.png'
        res[test_set] = int(np.argmax(labels_predicted[i-1]))
    write_json(filename, res)
