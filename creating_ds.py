import numpy as np
import pickle
import os 
import cv2
from tqdm import tqdm
import random

datadir = "Datasets/PetImages"
pets = ["Dog","Cat"]

IMG_SIZE = 50

training_data = []
def create_ds():
    for pet in pets:
        path = os.path.join(datadir,pet)
        category = pets.index(pet)
        for img in tqdm(os.listdir(path)):
            try:
                img_array = cv2.imread(os.path.join(path,img) ,cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE)) 
                # print new_array.shape
                training_data.append([new_array, category]) 
            except:  
                pass


create_ds()
print((training_data))
# np.savetxt("traineddata.txt",training_data)

random.shuffle(training_data)

for sample in training_data[:10]:
    print(sample[1])

X = []
y = []

i = 0
for features, label in training_data:
    X.append(features)
    y.append(label)
    i+=1
print(X[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 1)


import pickle

pickle_out = open("X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)
