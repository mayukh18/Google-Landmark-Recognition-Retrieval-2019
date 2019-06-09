import numpy as np
import tensorflow as tf
import keras.backend as K




def accuracy_class(y_true, y_pred):
    true = K.argmax(y_true, axis=1)
    pred = K.argmax(y_pred, axis=1)
    matches = K.equal(true, pred)
    return K.mean(matches)


def accuracy_class_numpy(y_true, y_pred):
    true = np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)
    matches = true == pred
    return np.mean(matches)


def getConfidence(y_pred):
    y_pred_max = np.reshape(np.amax(y_pred, axis=1), (y_pred.shape[0], 1))

    top5 = np.zeros((y_pred.shape[0], 5))
    max_indices = np.argsort(y_pred, axis=1)[:, ::-1][:, :5]
    for i in range(y_pred.shape[0]):
        top5[i, :] = y_pred[i, max_indices[i, :]]
    diff = y_pred_max - top5
    weights = np.array([[0., 0.35, 0.28, 0.22, 0.15]])
    weighted_diffs = diff * weights
    return np.sum(weighted_diffs, axis=1)


def getOrder(y_pred):
    summ = getConfidence(y_pred)
    summ_indices = np.argsort(summ)[::-1]
    return summ_indices


def MAP_numpy(y_true, y_pred):
    true = np.argmax(y_true, axis=1)
    pred = np.argmax(y_pred, axis=1)
    matches = true == pred

    order = getOrder(y_pred)
    orderedMatches = matches[order]

    correct = 0.
    summ = 0.
    for i in range(y_true.shape[0]):
        correct += int(orderedMatches[i])
        summ += (correct / (i + 1)) * int(orderedMatches[i])
    return summ / y_true.shape[0]


def validateMAP(model, valid_x, valid_y):
    """
    :param model: the model to use
    :param valid_x: numpy array of validation images
    :param valid_y: list of landmarks of the validation images
    :return:
    """
    N = valid_x.shape[0]
    batchsize = 1000
    conf_list = []
    y_pred_list = []
    validM = N // batchsize + int(N % batchsize > 0)
    for i in range(validM):
        preds = model.predict(valid_x[i * batchsize:min(N, (i + 1) * batchsize), :, :, :])
        conf = list(np.amax(preds, axis=1))
        conf_list.extend(conf)
        y_pred = list(np.argmax(preds, axis=1))
        y_pred_list.extend(y_pred)

    matches = list(np.array(y_pred_list) == np.array(valid_y))

    order = list(np.argsort(conf_list)[::-1])
    orderedMatches = [matches[o] for o in order]

    correct = 0.
    summ = 0.
    for i in range(len(orderedMatches)):
        correct += int(orderedMatches[i])
        summ += (correct / (i + 1)) * int(orderedMatches[i])

    print(np.sum(matches))
    print(correct)
    print(summ / len(orderedMatches))