import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras
from keras.applications import ResNet50
from keras.layers import Layer
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    GlobalAveragePooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam

from collections import Counter

import os

from generators import *
from utilities import *
from metrics import *


# ------------------------------ form the dataset ------------------------------ #

download_file("https://s3.amazonaws.com/google-landmark/metadata/train.csv", "train.csv")
train = pd.read_csv("train.csv")

print(train.head())
print(train.shape)
print("Number of classes {}".format(len(train.landmark_id.unique())))



NUM_THRESHOLD = 20

counts = dict(Counter(train['landmark_id']))
landmarks_dict = {x:[] for x in train.landmark_id.unique() if counts[x] >= NUM_THRESHOLD and x != 138982}
NUM_CLASSES = len(landmarks_dict)
print("Total number of valid classes: {}".format(NUM_CLASSES))

i = 0
landmark_to_idx = {}
idx_to_landmark = []
for k in landmarks_dict:
    landmark_to_idx[k] = i
    idx_to_landmark.append(k)
    i += 1

all_ids = train['id'].tolist()
all_landmarks = train['landmark_id'].tolist()
valid_ids_dict = {x[0].split("/")[-1]:landmark_to_idx[x[1]] for x in zip(all_ids, all_landmarks) if x[1] in landmarks_dict}
valid_ids_list = [x[0] for x in zip(all_ids, all_landmarks) if x[1] in landmarks_dict]

NUM_EXAMPLES = len(valid_ids_list)
print("Total number of valid examples: {}".format(NUM_EXAMPLES))


# --------------------------------------- load model --------------------------------------- #

# loading the model trained in recognition problem
model = load_model("../input/resnet50-0092/resnet50.model", custom_objects={'accuracy_class':accuracy_class})

# removing the softmax layer to keep upto the global pooling layer
model = Model(inputs=[model.input], outputs=[model.layers[-2].output])

# ---------------------------------------- faiss -------------------------------------------- #

import faiss                   # make faiss available
faiss_index = faiss.IndexFlatL2(2048)   # build the index
print(faiss_index.is_trained)

# ------------------------------- add the index images features to faiss -------------------------------- #

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import time

tm = time.time()

index_ids = []

tar_images = []
tar_ids = []


def pickfiles(dirr):
    count = 0
    for f in os.listdir(dirr):
        if os.path.isfile(dirr + "/" + f):
            tar_images.append(dirr + "/" + f)
            tar_ids.append(f[:-4])
            count += 1
        else:
            count += pickfiles(dirr + "/" + f)
    return count


for tar in range(100):
    if tar < 10:
        tar_id = "00" + str(tar)
    else:
        tar_id = "0" + str(tar)

    tar_images = []
    tar_ids = []

    download_file("https://s3.amazonaws.com/google-landmark/index/images_{}.tar".format(tar_id), "images.tar",
                  bar=False)
    tar = tarfile.open('images.tar')
    tar.extractall("imagesfolder")
    tar.close()

    os.unlink("images.tar")

    total = pickfiles("imagesfolder")
    print(tar, total, len(tar_ids))

    N = total
    batchsize = 1000
    validM = N // batchsize + int(N % batchsize > 0)
    for i in range(validM):
        temp = tar_images[i * batchsize:min(N, (i + 1) * batchsize)]
        batch_images = []
        for t in temp:
            im = cv2.imread(t)
            im = cv2.resize(im, (192, 192), interpolation=cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            batch_images.append(im)

        batch_images = np.array(batch_images)

        # print("batch_images", batch_images.shape)
        preds = model.predict(batch_images)
        faiss_index.add(np.array(preds))

    # final_preds.extend(preds_list)
    index_ids.extend(tar_ids)
    shutil.rmtree("imagesfolder")

print("time", time.time() - tm)
print(faiss_index.ntotal)


# ------------------------------------ similarity search from test images --------------------------------- #

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import time

tm = time.time()

final_ids = []
final_preds = []
final_dists = []

tar_images = []
tar_ids = []


def pickfiles(dirr):
    count = 0
    for f in os.listdir(dirr):
        if os.path.isfile(dirr + "/" + f):
            tar_images.append(dirr + "/" + f)
            tar_ids.append(f[:-4])
            count += 1
        else:
            count += pickfiles(dirr + "/" + f)
    return count


for tar in range(20):
    if tar < 10:
        tar_id = "00" + str(tar)
    else:
        tar_id = "0" + str(tar)

    tar_images = []
    tar_ids = []

    download_file("https://s3.amazonaws.com/google-landmark/test/images_{}.tar".format(tar_id), "images.tar", bar=False)
    tar = tarfile.open('images.tar')
    tar.extractall("imagesfolder")
    tar.close()

    os.unlink("images.tar")

    total = pickfiles("imagesfolder")
    print(tar, total, len(tar_ids))

    N = total
    batchsize = 1000
    preds_list = []
    dists_list = []
    validM = N // batchsize + int(N % batchsize > 0)
    for i in range(validM):
        temp = tar_images[i * batchsize:min(N, (i + 1) * batchsize)]
        batch_images = []
        for t in temp:
            im = cv2.imread(t)
            im = cv2.resize(im, (192, 192), interpolation=cv2.INTER_AREA)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            batch_images.append(im)

        batch_images = np.array(batch_images)

        # print("batch_images", batch_images.shape)
        preds = model.predict(batch_images)
        for j in range(preds.shape[0]):
            _, I = faiss_index.search(preds[j:j + 1], 100)
            preds_list.append(I[0])

    final_preds.extend(preds_list)
    final_ids.extend(tar_ids)
    shutil.rmtree("imagesfolder")

print("time", time.time() - tm)

# ---------------------------------------- submission --------------------------------------- #

test_out = []
for i in range(len(final_ids)):
    preds = list(final_preds[i])
    test_out.append(" ".join([index_ids[preds[j]] for j in range(len(preds))]))

outdf = pd.DataFrame({'id':final_ids, 'images':test_out})
outdf.to_csv("submission.csv", index=False)

# ------------------------------------------ the end ---------------------------------------- #