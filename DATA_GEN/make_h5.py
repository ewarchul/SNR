#https://www.kaggle.com/msripooja/dog-images-classification-using-keras-alexnet

import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import cv2
import h5py
from keras.utils import to_categorical

ile_ras = 120

print("Loaded all libraries")

fpath = "data/"
random_seed = 42

categories = os.listdir(fpath)
categories = categories[:ile_ras]
print("List of categories = ",categories,"\n\nNo. of categories = ", len(categories))

def load_images_and_labels(categories):
    img_lst=[]
    labels=[]
    for index, category in enumerate(categories):
        print("Postep:",index,"\r")
        for image_name in os.listdir(fpath+"/"+category):
            img = cv2.imread(fpath+"/"+category+"/"+image_name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            img_array = Image.fromarray(img, 'RGB')
        
            resized_img = img_array.resize((224, 224))
            
            img_lst.append(np.array(img_array))
            
            labels.append(index)
         
    return img_lst, labels

images, labels = load_images_and_labels(categories)
print("No. of images loaded = ",len(images),"\nNo. of labels loaded = ",len(labels))
print(type(images),type(labels))

images = np.array(images)
labels = np.array(labels)


print("Images shape = ",images.shape,"\nLabels shape = ",labels.shape)
print(type(images),type(labels))

def display_rand_images(images, labels):
    plt.figure(1 , figsize = (19 , 10))
    n = 0 
    for i in range(9):
        n += 1 
        r = np.random.randint(0 , images.shape[0] , 1)
        
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.3 , wspace = 0.3)
        plt.imshow(images[r[0]])
        
        plt.title('Dog breed : {}'.format(labels[r[0]]))
        plt.xticks([])
        plt.yticks([])
        
    plt.show()
    

n = np.arange(images.shape[0])
print("'n' values before shuffling = ",n)


np.random.seed(random_seed)
np.random.shuffle(n)
#print("\n'n' values after shuffling = ",n)


images = images[n]
labels = labels[n]

#print("Images shape after shuffling = ",images.shape,"\nLabels #shape after shuffling = ",labels.shape)

images = images.astype(np.float32)
labels = labels.astype(np.int32)
images = images/255

print("Images shape after normalization = ",images.shape)


x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size = 0.2, random_state = random_seed)

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

hf = h5py.File('XYtrain_test1.h5', 'w')

hf.create_dataset('x_train', data=x_train)
hf.create_dataset('y_train', data=y_train)
hf.create_dataset('x_test', data=x_test)
hf.create_dataset('y_test', data=y_test)

hf.close()


