import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import keras
from keras.applications import ResNet50
from keras.layers import Layer
from keras import regularizers
from keras.engine.topology import Input
from keras.layers import Activation, Add, BatchNormalization, Concatenate, Conv2D, Dense, Flatten, GlobalMaxPooling2D, \
    GlobalAveragePooling2D, Lambda, MaxPooling2D, Reshape
from keras.models import Model
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


# ------------------------------------- validation ------------------------------------------------- #

download_file("https://s3.amazonaws.com/google-landmark/train/images_001.tar", "validation.tar", bar=False)
tar = tarfile.open('validation.tar')
tar.extractall("validation")
tar.close()

os.unlink("validation.tar")

print(os.listdir())

validation_images_paths = []
validation_landmarks = []


def pickfiles(dirr):
    count = 0
    for f in os.listdir(dirr):
        if os.path.isfile(dirr + "/" + f):
            if f[:-4] in valid_ids_dict:
                validation_images_paths.append(dirr + "/" + f)
                validation_landmarks.append(valid_ids_dict[f[:-4]])
                count += 1
        else:
            count += pickfiles(dirr + "/" + f)
    return count


total = pickfiles("validation")
print("total:", total)

validation_images = []

for image_path in validation_images_paths:
    im = cv2.imread(image_path)
    im = cv2.resize(im, (192, 192), interpolation=cv2.INTER_AREA)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    validation_images.append(im)

valid_x = np.array(validation_images)
valid_y = np.zeros((len(validation_landmarks), NUM_CLASSES))

for i in range(len(validation_landmarks)):
    valid_y[i, validation_landmarks[i]] = 1.

shutil.rmtree("validation")
del validation_images

# ------------------------------------ model ----------------------------------------- #

res = ResNet50(include_top=False, weights='imagenet', input_shape=(192, 192, 3))

# making all the layers trainable
for layer in res.layers:
    layer.trainable = True

out = GlobalMaxPooling2D()(res.output)
out = Dense(NUM_CLASSES, activation='softmax')(out)
model = Model(res.input, out)
model.summary()

# ---------------------------------- clear block ------------------------------------- #

folder = "/kaggle/working/"
for file in os.listdir(folder):
    file_path = os.path.join(folder, file)
    if os.path.isfile(file_path):
        os.unlink(file_path)
    else:
        import shutil
        shutil.rmtree(file_path)

gc.collect()

# ----------------------------------- training ---------------------------------------- #

EPOCHS = 170
opt = Adam(0.0002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[accuracy_class])
model.fit_generator(generator=DataGen(valid_ids_dict, NUM_CLASSES, start=10, batch_size=64,steps=EPOCHS),
                    epochs=EPOCHS,
                    validation_data = [valid_x, valid_y],
                    use_multiprocessing=True,
                    workers=8,
                    verbose=2)

EPOCHS = 160
opt = Adam(0.0001)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[accuracy_class])
model.fit_generator(generator=DataGen(valid_ids_dict, NUM_CLASSES, start=180, batch_size=48,steps=EPOCHS),
                    epochs=EPOCHS,
                    validation_data = [valid_x, valid_y],
                    use_multiprocessing=True,
                    workers=4,
                    verbose=2)

EPOCHS = 50
opt = Adam(0.00004)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[accuracy_class])
model.fit_generator(generator=DataGen(valid_ids_dict, NUM_CLASSES, start=340, batch_size=48,steps=EPOCHS),
                    epochs=EPOCHS,
                    validation_data = [valid_x, valid_y],
                    use_multiprocessing=True,
                    workers=4,
                    verbose=2)

EPOCHS = 110
opt = Adam(0.00002)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=[accuracy_class])
model.fit_generator(generator=DataGen(valid_ids_dict, NUM_CLASSES, start=390, batch_size=48,steps=EPOCHS),
                    epochs=EPOCHS,
                    validation_data = [valid_x, valid_y],
                    use_multiprocessing=True,
                    workers=4,
                    verbose=2)

# ------------------------------------------- GAP metric validation -------------------------------------- #

gap = validateMAP()
print(gap)

# ------------------------------------------- testset ------------------------------------------------- #

download_file("https://s3.amazonaws.com/google-landmark/metadata/test.csv", "test.csv")
testdf = pd.read_csv("test.csv")
print(testdf.head())

testids = testdf['id'].tolist()
print(len(testids))

# -------------------------------------------- prediction ------------------------------------------------ #

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import time

tm = time.time()

final_ids = []
final_conf = []
final_preds = []

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
    conf_list = []
    y_pred_list = []
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

        preds = model.predict(batch_images)
        conf = list(np.amax(preds, axis=1))
        conf_list.extend(conf)
        y_pred = list(np.argmax(preds, axis=1))
        y_pred_list.extend(y_pred)

    final_preds.extend(y_pred_list)
    final_conf.extend(conf_list)
    final_ids.extend(tar_ids)
    shutil.rmtree("imagesfolder")

print("time", time.time() - tm)
print(len(final_preds))

# --------------------------------------- submission -------------------------------------- #

out = []
for i in range(len(final_preds)):
    idx = final_preds[i]
    out.append(str(idx_to_landmark[idx]) + " " + str(round(final_conf[i], 10)))

print(out[:5])

outdf = pd.DataFrame({"id": final_ids, "landmarks": out})
print(outdf.head())

outdf.to_csv("submissions.csv", index=False)

# ---------------------------------------- the end ----------------------------------------- #
