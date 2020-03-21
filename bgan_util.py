import os 
import glob
import numpy as np
import pandas as pd
import six

import _pickle as cPickle   #import cPickle
import keras.backend as K

import tensorflow as tf
from scipy.ndimage import imread
from scipy.misc import imresize
import scipy.io as sio

from sklearn.model_selection import train_test_split
import tensorflow.contrib.slim as slim
from skimage.transform import rotate

from PIL import Image
import cv2

import SimpleITK as sitk
import skimage.io as io
from augmenter import *


def show_all_variables():

    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)


def one_hot_encoded(class_numbers, num_classes):

    return np.eye(num_classes, dtype=float)[class_numbers]


class AttributeDict(dict):

    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value
    def __hash__(self):
        return hash(tuple(sorted(self.items())))


def average_gradients(tower_grads):
  """Calculate the average gradient for each shared variable across all towers.
  Note that this function provides a synchronization point across all towers.
  Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list
      is over individual gradients. The inner list is over the gradient
      calculation for each tower.
  Returns:
     List of pairs of (gradient, variable) where the gradient has been averaged
     across all towers.
  """
  average_grads = []
  for grad_and_vars in zip(*tower_grads):
    # Note that each grad_and_vars looks like the following:
    #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
    grads = []
    for g, _ in grad_and_vars:
      # Add 0 dimension to the gradients to represent the tower.
      expanded_g = tf.expand_dims(g, 0)

      # Append on a 'tower' dimension which we will average over below.
      grads.append(expanded_g)

    # Average over the 'tower' dimension.
    grad = tf.concat(axis=0, values=grads)
    grad = tf.reduce_mean(grad, 0)

    # Keep in mind that the Variables are redundant because they are shared
    # across towers. So .. we will just return the first tower's pointer to
    # the Variable.
    v = grad_and_vars[0][1]
    grad_and_var = (grad, v)
    average_grads.append(grad_and_var)
  return average_grads


def dice_coef(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):

    """Soft dice (Sørensen or Jaccard) coefficient for comparing the similarity
    of two batch of data, usually be used for binary image segmentation
    i.e. labels are binary. The coefficient between 0 to 1, 1 means totally match.

    Parameters
    -----------
    output : Tensor
        A distribution with shape: [batch_size, ....], (any dimensions).
    target : Tensor
        The target distribution, format the same with `output`.
    loss_type : str
        ``jaccard`` or ``sorensen``, default is ``jaccard``.
    axis : tuple of int
        All dimensions are reduced, default ``[1,2,3]``.
    smooth : float
        This small value will be added to the numerator and denominator.
            - If both output and target are empty, it makes sure dice is 1.
            - If either output or target are empty (all pixels are background), dice = ```smooth/(small_value + smooth)``, then if smooth is very small, dice close to 0 (even the image values lower than the threshold), so in this case, higher smooth can have a higher dice.

    Examples
    ---------
    >>> outputs = tl.act.pixel_wise_softmax(network.outputs)
    >>> dice_loss = 1 - tl.cost.dice_coe(outputs, y_)
    """

    inse = tf.reduce_sum(output * target, axis=axis)
    if loss_type == 'jaccard':
        l = tf.reduce_sum(output * output, axis=axis)
        r = tf.reduce_sum(target * target, axis=axis)
    elif loss_type == 'sorensen':
        l = tf.reduce_sum(output, axis=axis)
        r = tf.reduce_sum(target, axis=axis)
    else:
        raise Exception("Unknow loss_type")

    dice = (2. * inse + smooth) / (l + r + smooth)
    ##
    dice = tf.reduce_mean(dice, name='dice_coe')

    return dice


def tversky_loss_K(y_true, y_pred):

    alpha = 0.5
    beta = 0.5

    ones = K.ones_like(y_true)
    p0 = y_pred  # proba that voxels are class i
    p1 = ones - y_pred  # proba that voxels are not class i
    g0 = y_true
    g1 = ones - y_true

    num = K.sum(p0 * g0, (0, 1, 2, 3))
    den = num + alpha * K.sum(p0 * g1, (0, 1, 2, 3)) + beta * K.sum(p1 * g0, (0, 1, 2, 3))

    T = K.sum(num / den)  # when summing over classes, T has dynamic range [0 Ncl]

    Ncl = K.cast(K.shape(y_true)[-1], 'float32')
    return Ncl - T


def tversky_loss(y: tf.Tensor, y_pred: tf.Tensor, c_dim=3, alpha=0.5, beta=0.5, smooth=1., name='dice_loss') -> tf.Tensor:
    """
    Implementation of the Tversky loss
    (see Salehi et al., 2017, MLMI, https://arxiv.org/abs/1706.05721)
    which generalizes the Dice and Tanimoto index:
    alpha=beta=0.5  : dice coefficient
    alpha=beta=1    : tanimoto coefficient
    alpha+beta=1    : F_beta score
    The Tversky loss is calculated separately for each class and added together.
    The loss is normalized to [0, 1] using the number of classes.

    Use larger beta to weigh recall higher than precision.


    Args:
        y: The labels as one-hot encoding with the same size as y_pred.
        y_pred: The tensor of the network predictions.
        c_dim: The index of the channel dimension in y.
        alpha: Tversky loss alpha.
        beta: Tversky loss beta.
        smooth: Optional smoothing that is added to the nominator and
            denominator of the Tversky loss.

    Returns:
        loss: The loss tensor.

    """

    # convert to one-hot if shapes are not equal
    sp = np.asarray(y_pred.shape.as_list())

    # reduce all except channel dimension
    red_dim = list(np.setdiff1d(np.arange(0, 5), 4))

    with tf.name_scope(name):
        n_classes = tf.constant(sp[c_dim], dtype=tf.float32, name='num_classes')
        alpha = tf.constant(alpha, dtype=tf.float32, name='alpha')
        beta = tf.constant(beta, dtype=tf.float32, name='beta')
        smooth = tf.constant(smooth, dtype=tf.float32, name='smooth')
        with tf.name_scope('numerator'):
            num = tf.reduce_sum(tf.multiply(y, y_pred), axis=red_dim) + smooth

        with tf.name_scope('denominator'):
            red_sum = lambda x: tf.reduce_sum(x, axis=red_dim)
            denom = num + \
                    alpha * red_sum(tf.multiply(1 - y, y_pred)) + \
                    beta * red_sum(tf.multiply(y, 1 - y_pred))
        loss_t = (1 - tf.reduce_sum(num / denom) / n_classes)
    return loss_t


def dice_coef_K(y_true, y_pred, smooth=1e-5):

    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred, numLabels=3):

    dice=0
    for index in range(numLabels):
        dice -= dice_coef_K(y_true[:,:,:,index], y_pred[:,:,:,index])
    return dice


def generalized_dice_loss_w(y_true, y_pred):

    # Compute weights: "the contribution of each label is corrected by the inverse of its volume"
    Ncl = y_pred.shape[-1]
    w = np.zeros((Ncl,))
    for l in range(0, Ncl): w[l] = np.sum(np.asarray(y_true[:, :, :, l] == 1, np.int8))
    w = 1 / (w ** 2 + 0.00001)

    # Compute gen dice coef:
    numerator = y_true * y_pred
    numerator = w * K.sum(numerator, (0, 1, 2, 3))
    numerator = K.sum(numerator)

    denominator = y_true + y_pred
    denominator = w * K.sum(denominator, (0, 1, 2, 3))
    denominator = K.sum(denominator)

    gen_dice_coef = numerator / denominator

    return 1 - 2 * gen_dice_coef


def stable_nll_loss(output, target, axis=(1, 2, 3), smooth=1e-5):

    inse = tf.reduce_sum(output * target, axis=axis)

    ll = tf.reduce_mean(inse - tf.log(1 + tf.exp(output)))

    return -ll


class PanCT():

    def __init__(self, data_path, fn, img_size=256):

        self.name = "pan"

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/img_2d/'
        self.train_label_dir = data_path + '/label_2d/'
        self.test_img_dir = self.train_img_dir
        self.test_label_dir = self.train_label_dir

        self.train_img_files = os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)
        self.num_train = len(self.train_img_files)
        self.num_test = len(self.test_img_files)
        self.x_dim = [img_size, img_size, 1]  # the grayscale image


    # get_train_batch
    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files, output_size=128, crop_num=3):

        train_img = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.png'

            img = cv2.imread(os.path.join(train_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(train_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

            ############# from label #############################

            img_label, label_label, _ = crop_from_label(img, label, output_size)

            img_label = np.array([img_label])
            label_label = np.array([label_label])
            train_img[indx] = img_label
            train_label[indx] = label_label
            indx += 1

            # ############# from center #########################
            #
            img_center, label_center, start_x_y = crop_from_center(img, label, output_size)
            #
            # img_center = np.array([img_center])
            # label_center = np.array([label_center])
            # train_img[indx] = img_center
            # train_label[indx] = label_center
            # indx += 1

            ##################################################

            for i in range(crop_num):

                x_start_i = max(0, int(np.random.uniform(start_x_y[0] - output_size // 2, start_x_y[0] + output_size // 2)))
                y_start_i = max(0, int(np.random.uniform(start_x_y[1] - output_size // 2, start_x_y[1] + output_size // 2)))

                x_start_i = min(img.shape[0] - output_size - 1, x_start_i + output_size)
                y_start_i = min(img.shape[1] - output_size - 1, y_start_i + output_size)

                crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
                crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

                crop_i_img = np.array([crop_i_img])
                crop_i_label = np.array([crop_i_label])
                train_img[indx] = crop_i_img
                train_label[indx] = crop_i_label
                indx += 1

            ###################################################

        ###############################################################################

        # simple normalization

        train_img = train_img / 127.5 - 1.0

        train_img = train_img[:, :, :, np.newaxis]
        train_label = train_label[:, :, :, np.newaxis]

        #        train_label = tf.one_hot(train_label, depth=n_classes)
        #        train_label = session.run(train_label)

        assert indx == len(train_img)
#        assert np.min(train_label) == 0 and np.max(train_label) == 1

        return train_img, np.round(train_label)


    def get_test_batch(self, test_img_dir, test_label_dir, batch_img_files, output_size=128, crop_num=4):

        test_img = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)
        test_label = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.png'

            img = cv2.imread(os.path.join(test_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(test_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

            ############# from label #############################

            img_label, label_label, start_x_y = crop_from_label(img, label, output_size)

            img_label = np.array([img_label])
            label_label = np.array([label_label])
            test_img[indx] = img_label
            test_label[indx] = label_label
            indx += 1

            ############# from center #########################

            # img_center, label_center, start_x_y = crop_from_center(img, label, output_size)
            #
            # img_center = np.array([img_center])
            # label_center = np.array([label_center])
            # test_img[indx] = img_center
            # test_label[indx] = label_center
            # indx += 1

            ##################################################

            for i in range(crop_num):

                x_start_i = int(np.random.uniform(start_x_y[0] - output_size // 2, start_x_y[0] + output_size // 2))
                y_start_i = int(np.random.uniform(start_x_y[1] - output_size // 2, start_x_y[1] + output_size // 2))

                x_start_i = max(0, x_start_i)
                y_start_i = max(0, y_start_i)

                if x_start_i + output_size >= img.shape[0]:
                    x_start_i = img.shape[0] - output_size - 1

                if y_start_i + output_size >= img.shape[1]:
                    y_start_i = img.shape[1] - output_size - 1

                crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
                crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

                crop_i_img = np.array([crop_i_img])
                crop_i_label = np.array([crop_i_label])
                test_img[indx] = crop_i_img
                test_label[indx] = crop_i_label
                indx += 1

            ###################################################

        ###############################################################################

        test_img = test_img / 127.5 - 1.0

        test_img = test_img[:, :, :, np.newaxis]
        test_label = test_label[:, :, :, np.newaxis]

        assert indx == len(test_img)

        return test_img, test_label # np.round(test_img)


class SpleenCT():

    def __init__(self, data_path, img_size=256, val_ratio=0.1):

        self.name = "spleen"

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/spleen_imgs_2d/'
        self.train_label_dir = data_path + '/spleen_labels_2d/'
        self.test_img_dir = self.train_img_dir
        self.test_label_dir = self.train_label_dir

        self.train_img_files = os.listdir(self.train_img_dir)

        self.val_ratio = val_ratio
        self.num_test = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_test

        self.test_img_files = self.train_img_files[-self.num_test:]
        self.train_img_files = self.train_img_files[:-self.num_test]


        if self.val_ratio == 0:
            self.train_img_files = self.test_img_files
            self.num_test = self.num_train

        self.x_dim = [img_size, img_size, 1]  # the grayscale image


    # get_train_batch
    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files, output_size=128, num_aug=2, crop_num=3):

        train_img = np.zeros((len(batch_img_files) * (num_aug + 1) * (crop_num + 1), output_size, output_size), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files) * (num_aug + 1) * (crop_num + 1), output_size, output_size), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.png'

            img = cv2.imread(os.path.join(train_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(train_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

            ############# from label #############################

            img_label, label_label, start_x_y = crop_from_label(img, label, output_size)

            img_label = np.array([img_label])
            label_label = np.array([label_label])
            train_img[indx] = img_label
            train_label[indx] = label_label
            indx += 1

            ##################################################

            for i in range(crop_num):

                x_start_i = int(np.random.uniform(start_x_y[0] - output_size // 2, start_x_y[0] + output_size // 2))
                y_start_i = int(np.random.uniform(start_x_y[1] - output_size // 2, start_x_y[1] + output_size // 2))

                x_start_i = max(0, x_start_i)
                y_start_i = max(0, y_start_i)

                if x_start_i + output_size >= img.shape[0]:
                    x_start_i = img.shape[0] - output_size - 1

                if y_start_i + output_size >= img.shape[1]:
                    y_start_i = img.shape[1] - output_size - 1

                crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
                crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

                ######## tmp ########
                # aug_img_name = '/home/jtma/PycharmProjects/dataset/' + img_base_name + '_' + str(i) + '.png'
                # aug_label_name = '/home/jtma/PycharmProjects/dataset/' + img_base_name + '_' + str(i) + '_mask.png'
                # cv2.imwrite(aug_img_name, crop_i_img)
                # cv2.imwrite(aug_label_name, crop_i_label)

                crop_i_img = np.array([crop_i_img])
                crop_i_label = np.array([crop_i_label])
                train_img[indx] = crop_i_img
                train_label[indx] = crop_i_label
                indx += 1

            ###################################################

            # data augmentation
            i = 0
            while i < num_aug:

                aug_image, aug_label = augmentation(img_base_name, img, label)
                aug_img_np = np.asarray([aug_image])
                aug_label_np = np.asarray([aug_label])

                print('min / max:', np.min(aug_label_np), np.max(aug_label_np))

                if np.count_nonzero(aug_label_np == 1) >= 500:
                    print("augment %i", i, " data = ", img_base_name, ", number of non-zero pixels = ",
                            np.count_nonzero(np.asarray(sitk.GetArrayFromImage(aug_label), np.int32) == 1))
                    i += 1

                    train_img[indx] = aug_img_np
                    train_label[indx] = aug_label_np
                    indx += 1

                else:
                    print("augment = ", np.count_nonzero(aug_label_np == 1))

        ###############################################################################

        # simple normalization

        train_img = train_img / 127.5 - 1.0

        train_img = train_img[:, :, :, np.newaxis]
        train_label = train_label[:, :, :, np.newaxis]

        assert indx == len(train_img)

        return train_img, np.round(train_label)


    def get_test_batch(self, test_img_dir, test_label_dir, batch_img_files, output_size=128, crop_num=4):

        test_img = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)
        test_label = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.png'

            img = cv2.imread(os.path.join(test_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(test_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

            ############# from label #############################

            img_label, label_label, start_x_y = crop_from_label(img, label, output_size)

            img_label = np.array([img_label])
            label_label = np.array([label_label])
            test_img[indx] = img_label
            test_label[indx] = label_label
            indx += 1

            ##################################################

            for i in range(crop_num):

                x_start_i = int(np.random.uniform(start_x_y[0] - output_size // 2, start_x_y[0] + output_size // 2))
                y_start_i = int(np.random.uniform(start_x_y[1] - output_size // 2, start_x_y[1] + output_size // 2))

                x_start_i = max(0, x_start_i)
                y_start_i = max(0, y_start_i)

                if x_start_i + output_size >= img.shape[0]:
                    x_start_i = img.shape[0] - output_size - 1

                if y_start_i + output_size >= img.shape[1]:
                    y_start_i = img.shape[1] - output_size - 1

                crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
                crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

                crop_i_img = np.array([crop_i_img])
                crop_i_label = np.array([crop_i_label])
                test_img[indx] = crop_i_img
                test_label[indx] = crop_i_label
                indx += 1

            ###################################################

        ###############################################################################

        test_img = test_img / 127.5 - 1.0

        test_img = test_img[:, :, :, np.newaxis]
        test_label = test_label[:, :, :, np.newaxis]

        assert indx == len(test_img)

        return test_img, test_label


class ProstCT():

    def __init__(self, data_path, img_size=224, val_ratio=0.1):

        self.name = "spleen"

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/spleen_imgs_2d/'
        self.train_label_dir = data_path + '/spleen_labels_2d/'
        self.test_img_dir = self.train_img_dir
        self.test_label_dir = self.train_label_dir

        self.train_img_files = os.listdir(self.train_img_dir)

        self.val_ratio = val_ratio
        self.num_test = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_test

        self.test_img_files = self.train_img_files[-self.num_test:]
        self.train_img_files = self.train_img_files[:-self.num_test]


        if self.val_ratio == 0:
            self.train_img_files = self.test_img_files
            self.num_test = self.num_train

        self.x_dim = [img_size, img_size, 1]  # the grayscale image


    # get_train_batch
    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files, output_size=128, num_aug=2, crop_num=3):

        train_img = np.zeros((len(batch_img_files) * (num_aug + 1) * (crop_num + 1), output_size, output_size), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files) * (num_aug + 1) * (crop_num + 1), output_size, output_size), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.png'

            img = cv2.imread(os.path.join(train_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(train_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

            ############# from label #############################

            img_label, label_label, start_x_y = crop_from_label(img, label, output_size)

            img_label = np.array([img_label])
            label_label = np.array([label_label])
            train_img[indx] = img_label
            train_label[indx] = label_label
            indx += 1

            ##################################################

            for i in range(crop_num):

                x_start_i = int(np.random.uniform(start_x_y[0] - output_size // 2, start_x_y[0] + output_size // 2))
                y_start_i = int(np.random.uniform(start_x_y[1] - output_size // 2, start_x_y[1] + output_size // 2))

                x_start_i = max(0, x_start_i)
                y_start_i = max(0, y_start_i)

                if x_start_i + output_size >= img.shape[0]:
                    x_start_i = img.shape[0] - output_size - 1

                if y_start_i + output_size >= img.shape[1]:
                    y_start_i = img.shape[1] - output_size - 1

                crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
                crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

                ######## tmp ########
                # aug_img_name = '/home/jtma/PycharmProjects/dataset/' + img_base_name + '_' + str(i) + '.png'
                # aug_label_name = '/home/jtma/PycharmProjects/dataset/' + img_base_name + '_' + str(i) + '_mask.png'
                # cv2.imwrite(aug_img_name, crop_i_img)
                # cv2.imwrite(aug_label_name, crop_i_label)

                crop_i_img = np.array([crop_i_img])
                crop_i_label = np.array([crop_i_label])
                train_img[indx] = crop_i_img
                train_label[indx] = crop_i_label
                indx += 1

            ###################################################

            # data augmentation
            i = 0
            while i < num_aug:

                aug_image, aug_label = augmentation(img_base_name, img, label)
                aug_img_np = np.asarray([aug_image])
                aug_label_np = np.asarray([aug_label])

                print('min / max:', np.min(aug_label_np), np.max(aug_label_np))

                if np.count_nonzero(aug_label_np == 1) >= 500:
                    print("augment %i", i, " data = ", img_base_name, ", number of non-zero pixels = ",
                            np.count_nonzero(np.asarray(sitk.GetArrayFromImage(aug_label), np.int32) == 1))
                    i += 1

                    train_img[indx] = aug_img_np
                    train_label[indx] = aug_label_np
                    indx += 1

                else:
                    print("augment = ", np.count_nonzero(aug_label_np == 1))

        ###############################################################################

        # simple normalization

        train_img = train_img / 127.5 - 1.0

        train_img = train_img[:, :, :, np.newaxis]
        train_label = train_label[:, :, :, np.newaxis]

        assert indx == len(train_img)

        return train_img, np.round(train_label)


    def get_test_batch(self, test_img_dir, test_label_dir, batch_img_files, output_size=128, crop_num=4):

        test_img = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)
        test_label = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.png'

            img = cv2.imread(os.path.join(test_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
            label = cv2.imread(os.path.join(test_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

            img = cv2.resize(img, (img.shape[0] // 2, img.shape[1] // 2), interpolation=cv2.INTER_CUBIC)
            label = cv2.resize(label, (label.shape[0] // 2, label.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

            ############# from label #############################

            img_label, label_label, start_x_y = crop_from_label(img, label, output_size)

            img_label = np.array([img_label])
            label_label = np.array([label_label])
            test_img[indx] = img_label
            test_label[indx] = label_label
            indx += 1

            ##################################################

            for i in range(crop_num):

                x_start_i = int(np.random.uniform(start_x_y[0] - output_size // 2, start_x_y[0] + output_size // 2))
                y_start_i = int(np.random.uniform(start_x_y[1] - output_size // 2, start_x_y[1] + output_size // 2))

                x_start_i = max(0, x_start_i)
                y_start_i = max(0, y_start_i)

                if x_start_i + output_size >= img.shape[0]:
                    x_start_i = img.shape[0] - output_size - 1

                if y_start_i + output_size >= img.shape[1]:
                    y_start_i = img.shape[1] - output_size - 1

                crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
                crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

                crop_i_img = np.array([crop_i_img])
                crop_i_label = np.array([crop_i_label])
                test_img[indx] = crop_i_img
                test_label[indx] = crop_i_label
                indx += 1

            ###################################################

        ###############################################################################

        test_img = test_img / 127.5 - 1.0

        test_img = test_img[:, :, :, np.newaxis]
        test_label = test_label[:, :, :, np.newaxis]

        assert indx == len(test_img)

        return test_img, test_label


################## Utilities ############################


def find_bounding_box(label):

    x_min, y_min = label.shape[0], label.shape[1]
    x_max, y_max = 0, 0

    im2, ctrs, hier = cv2.findContours(label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(label)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        x_min = x if x_min > x else x_min
        y_min = y if y_min > y else y_min

        x_max = x + w if x_max < w + w else x_max
        y_max = y + h if y_max < y + h else y_max

    return [x_min, x_max, y_min, y_max]

    ####################################################################################################################


def crop_from_label(img, label, output_size=128):

    x_min, y_min = img.shape[0], img.shape[1]
    x_max, y_max = 0, 0

    im2, ctrs, hier = cv2.findContours(label.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(label)[0])

    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)
        x_min = x if x_min > x else x_min
        y_min = y if y_min > y else y_min

        x_max = x + w if x_max < w + w else x_max
        y_max = y + h if y_max < y + h else y_max

    ####################################################################################################################

    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    x_start = x_center - int(output_size / 2) if x_center - int(output_size / 2) >= 0 else 0
    y_start = y_center - int(output_size / 2) if y_center - int(output_size / 2) >= 0 else 0

    if x_start + output_size > img.shape[0]:
        x_start = img.shape[0] - output_size

    if y_start + output_size > img.shape[1]:
        y_start = img.shape[1] - output_size

    img = img[y_start:y_start + output_size, x_start:x_start + output_size]
    label = label[y_start:y_start + output_size, x_start:x_start + output_size]

    return img, label, [x_start, y_start]


def crop_from_center(img, label, output_size=128):

    x_min, y_min = img.shape[0], img.shape[1]
    x_max, y_max = 0, 0

    ####################################################################################################################

    x_center = int((x_min + x_max) / 2)
    y_center = int((y_min + y_max) / 2)

    x_start = x_center - int(output_size / 2) if x_center - int(output_size / 2) >= 0 else 0
    y_start = y_center - int(output_size / 2) if y_center - int(output_size / 2) >= 0 else 0

    if x_start + output_size > img.shape[0]:
        x_start = img.shape[0] - output_size

    if y_start + output_size > img.shape[1]:
        y_start = img.shape[1] - output_size

    img = img[y_start:y_start + output_size, x_start:x_start + output_size]
    label = label[y_start:y_start + output_size, x_start:x_start + output_size]

    return img, label, [x_start, y_start]


def augmentation(img_name, image, imageB, org_width=256, org_height=256, width=278, height=280):

    max_angle = 20
    image = cv2.resize(image, (height, width))
    imageB = cv2.resize(imageB, (height, width))
    #
    angle = np.random.randint(max_angle)
    if np.random.randint(2):
        angle = -angle
    image = rotate(image, angle, resize=True)
    imageB = rotate(imageB, angle, resize=True)

    xstart = np.random.randint(width - org_width)
    ystart = np.random.randint(height - org_height)
    image = image[xstart:xstart + org_width, ystart:ystart + org_height]
    imageB = imageB[xstart:xstart + org_width, ystart:ystart + org_height]

    if np.random.randint(2):
        image = cv2.flip(image, 1)
        imageB = cv2.flip(imageB, 1)

    if np.random.randint(2):
        image = cv2.flip(image, 0)
        imageB = cv2.flip(imageB, 0)

    image = cv2.resize(image, (org_height, org_width))
    imageB = cv2.resize(imageB, (org_height, org_width))

    return image, imageB


def GenerateSlices(img_dir, label_dir, save_img_path, save_label_path, num_pixels, thres=[50, 500]):

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    if not os.path.exists(save_label_path):
        os.makedirs(save_label_path)


    for file_name in os.listdir(img_dir):

        img_base_name = file_name.split('.')[0]
        label_base_name = img_base_name + '_mask'

        ext = '.png'

#        image = readItkData(img_dir + file_name)
        image = readItkData_multiclass(img_dir + file_name)
        label = readItkLabelData(label_dir + label_base_name + '.nii.gz')

        img_np = sitk.GetArrayFromImage(image)
        img_np = np.asarray(img_np, np.float32)
#        img_np = np.transpose(img_np, (3, 2, 1, 0))

        sitk.WriteImage(sitk.GetImageFromArray(img_np), '/home/jtma/PycharmProjects/dataset/' + img_base_name + '.nii.gz')


        label_np = sitk.GetArrayFromImage(label)
        label_np = np.asarray(label_np, np.int32)
        label_np = np.transpose(label_np, (2, 1, 0))

        # threshold img_np
        img_np[img_np > thres[1]] = thres[1]
        img_np[img_np < thres[0]] = thres[0]

            # label_np[label_np > 1] = 1

        for z in range(img_np.shape[2]):

            img_slice = img_np[:, :, z, :]
            label_slice = label_np[:, :, z]

            if np.count_nonzero(label_slice == 1) > num_pixels:   # num_pixels

                # print("total pixels = ", np.count_nonzero(label_slice == 1))

                write_name_img = save_img_path + img_base_name + "_%03d" % z + ext
                write_name_label = save_label_path + img_base_name + "_%03d" % z + '_mask' + ext

                cv2.imwrite(write_name_img, img_slice)
                cv2.imwrite(write_name_label, label_slice)


def Generate_negative_slices(img_dir, save_img_path, thres=[50, 500]):

    if not os.path.exists(save_img_path):
        os.makedirs(save_img_path)

    for file_name in os.listdir(img_dir):

        img_base_name = file_name.split('.')[0]
        ext = '.png'

        print('processing image - ', img_base_name)
        image = readItkData(img_dir + file_name)
#        image = readItkData_multiclass(img_dir + file_name)

        img_np = sitk.GetArrayFromImage(image)
        img_np = np.asarray(img_np, np.float32)
        img_np = np.transpose(img_np, (2, 1, 0))

        # # threshold img_np
        # img_np[img_np > thres[1]] = thres[1]
        # img_np[img_np < thres[0]] = thres[0]

        for z in range(img_np.shape[2]):

            img_slice = img_np[:, :, z]

            if np.max(img_slice > 0):

                write_name_img = save_img_path + img_base_name + "_%03d" % z + ext
                cv2.imwrite(write_name_img, img_slice)


def readItkData(filename):

    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    img = reader.Execute()

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkInt16)
    img = castImageFilter.Execute(img)

    return img


def readItkData_multiclass(filename):

    img = sitk.ReadImage(filename, sitk.sitkVectorFloat32)

    return img


def readItkLabelData(filename):

    # open the shape with sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    label = reader.Execute()

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
    label = castImageFilter.Execute(label)

    return label


def saveItkData(img_arr, save_path, save_name, ext=".nii.gz"):

    """
    save the array img_arr to .nii.gz image
    :param img_arr: note that transpose the array back
    :return: the itk image from img_arr
    """
    img_arr = np.asarray(img_arr, np.float32)
    img_arr = np.transpose(img_arr, (2, 1, 0))

    out_image = sitk.GetImageFromArray(img_arr)
    out_image.SetSpacing()

    # save image as nii.gz
    sitk.WriteImage(sitk.GetImageFromArray(img_arr), save_path + save_name + ext)


def findMaskBoundingBox(img_arr):

    mask = np.where(img_arr > 0)

    boundingAxes = [np.min(mask[0]), np.max(mask[0]),
                    np.min(mask[1]), np.max(mask[1])]

    x_dim = boundingAxes[1] - boundingAxes[0] + 1
    y_dim = boundingAxes[3] - boundingAxes[2] + 1

    mask = np.zeros((x_dim, y_dim), dtype=np.float32)
    mask = img_arr[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3]]  # merge_to_3d_volume

    # xmin = img_dim[0] - 1 - np.max(mask[0])
    # xmax = img_dim[0] - 1 - np.min(mask[0])
    # ymin = img_dim[1] - 1 - np.max(mask[1])
    # ymax = img_dim[1] - 1 - np.min(mask[1])

    return boundingAxes, mask


def getPaddingBox(array, out_size=128):

    """
    pad the array with zeros if smaller than the out size
    :param array: input array - 3d
    :param out_size: the target size
    :return: the padded array with the output size
    """

    dims = array.shape

    out_arr = np.zeros((out_size, out_size))
    out_arr[:dims[0], :dims[1]] = array

    return out_arr


def binaryThresholdImage(img, lowerThreshold):

    maxFilter = sitk.StatisticsImageFilter()
    maxFilter.Execute(img)
    maxValue = maxFilter.GetMaximum()
    thresholded = sitk.BinaryThreshold(img, lowerThreshold, maxValue, 1, 0)

    return thresholded


def getLargestConnectedComponents(img):

    connectedFilter = sitk.ConnectedComponentImageFilter()
    connectedComponents = connectedFilter.Execute(img)

    labelStatistics = sitk.LabelShapeStatisticsImageFilter()
    labelStatistics.Execute(connectedComponents)
    nrLabels = labelStatistics.GetNumberOfLabels()

    biggestLabelSize = 0
    biggestLabelIndex = 1
    for i in range(1, nrLabels+1):
        curr_size = labelStatistics.GetNumberOfPixels(i)
        if curr_size > biggestLabelSize:
            biggestLabelSize = curr_size
            biggestLabelIndex = i

    largestComponent = sitk.BinaryThreshold(connectedComponents, biggestLabelIndex, biggestLabelIndex)

    return largestComponent


##############################################################################################


def get_test_batch_offline(test_img_dir, test_label_dir, batch_img_files, output_size=128, crop_num=4):

    test_img = np.zeros((len(batch_img_files) * (crop_num + 1), output_size, output_size), dtype=np.float32)
    test_label = np.zeros((len(batch_img_files) * (crop_num + 1),  output_size, output_size), dtype=np.float32)
    test_crop_stores = np.zeros((len(batch_img_files) * (crop_num + 1), 2), dtype=np.float32)

    indx = 0
    for file_index in range(len(batch_img_files)):

        img_base_name = batch_img_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.png'

        img = cv2.imread(os.path.join(test_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(test_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

        img_dim = img.shape

        ## resample it first
        img = cv2.resize(img, (img_dim[0] // 2, img_dim[0] // 2), interpolation=cv2.INTER_CUBIC)
        label = cv2.resize(label, (img_dim[0] // 2, img_dim[0] // 2), interpolation=cv2.INTER_CUBIC)

        ############################## crop it ####################################

        #img_center, label_center, start_x_y = crop_from_center(img, label, output_size)   #(img.shape[0] / 2)
        img_center, label_center, start_x_y = crop_from_label(img, label, output_size)

        # save_path = '/home/jtma/PycharmProjects/dataset/resample_test/'
        # aug_img_name = save_path + img_base_name + '_' + 'center.png'
        # aug_label_name = save_path + img_base_name + '_' + 'center_mask.png'
        # cv2.imwrite(aug_img_name, img_center)
        # cv2.imwrite(aug_label_name, label_center)

        img_center = np.array([img_center])
        label_center = np.array([label_center])
        test_img[indx] = img_center
        test_label[indx] = label_center
        test_crop_stores[indx] = start_x_y
        indx += 1

        ##################################################

        for i in range(crop_num):

            x_start_i = int(np.random.uniform(start_x_y[0] - output_size // 4, start_x_y[0] + output_size // 4))   # int(img.shape[0] / 4
            y_start_i = int(np.random.uniform(start_x_y[1] - output_size // 4, start_x_y[1] + output_size // 4))

            x_start_i = max(0, x_start_i)
            y_start_i = max(0, y_start_i)

            if x_start_i + output_size >= img.shape[0]:
                x_start_i = img.shape[0] - output_size - 1

            if y_start_i + output_size >= img.shape[1]:
                y_start_i = img.shape[1] - output_size - 1

            crop_i_img = img[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]
            crop_i_label = label[y_start_i:y_start_i + output_size, x_start_i:x_start_i + output_size]

            # aug_img_name = save_path + img_base_name + '_' + str(i) + '.png'
            # aug_label_name = save_path + img_base_name + '_' + str(i) + '_mask.png'
            # cv2.imwrite(aug_img_name, crop_i_img)
            # cv2.imwrite(aug_label_name, crop_i_label)


            # crop_i_img = cv2.resize(crop_i_img, (output_size, output_size), interpolation=cv2.INTER_CUBIC)
            # crop_i_label = cv2.resize(crop_i_label, (output_size, output_size), interpolation=cv2.INTER_CUBIC)

            crop_i_img = np.array([crop_i_img])
            crop_i_label = np.array([crop_i_label])
            test_img[indx] = crop_i_img
            test_label[indx] = crop_i_label
            test_crop_stores[indx] = [x_start_i, y_start_i]
            indx += 1

        ###################################################

    ###############################################################################

    test_img = test_img / 127.5 - 1.0
        # test_label[test_label > 1] = 1
        # test_label[test_label < 0 ] = 0

    test_img = test_img[:, :, :, np.newaxis]
    test_label = test_label[:, :, :, np.newaxis]


    assert indx == len(test_img)
#    assert np.min(test_label) == 0 or np.max(test_label) == 1

    return test_img, np.round(test_label), test_crop_stores


def get_test_img_label_offline(test_img_dir, test_label_dir, batch_img_files, img_size=(512, 512)):

    test_img = np.zeros((len(batch_img_files), img_size[0], img_size[1]), dtype=np.float32)
    test_label = np.zeros((len(batch_img_files),  img_size[0], img_size[1]), dtype=np.float32)

    indx = 0
    for file_index in range(len(batch_img_files)):

        img_base_name = batch_img_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.png'

        img = cv2.imread(os.path.join(test_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(os.path.join(test_label_dir, label_base_name + ext), cv2.IMREAD_GRAYSCALE)

        img = np.array([img])
        label = np.array([label])
        test_img[indx] = img
        test_label[indx] = label
        indx += 1

    ###############################################################################

    test_img = test_img / 127.5 - 1.0
        # test_label[test_label > 1] = 1
        # test_label[test_label < 0 ] = 0

    test_img = test_img[:, :, :, np.newaxis]
    test_label = test_label[:, :, :, np.newaxis]


    assert indx == len(test_img)
#    assert np.min(test_label) == 0 or np.max(test_label) == 1

    return test_img, np.round(test_label)


def get_test_img_unlabel(test_img_dir, batch_img_files, output_size=128):

    test_img = np.zeros((len(batch_img_files), output_size, output_size), dtype=np.float32)
    test_crop_stores = np.zeros((len(batch_img_files), 4), dtype=np.int32)

    indx = 0
    for file_index in range(len(batch_img_files)):

        img_base_name = batch_img_files[file_index].split('.')[0]
        ext = '.png'

        img = cv2.imread(os.path.join(test_img_dir, img_base_name + ext), cv2.IMREAD_GRAYSCALE)

        ## crop from the slice first! , record the [x_start, y_start] ##

        boundingAxes, mask = findMaskBoundingBox(img)
#        print("img_base_name, ", boundingAxes, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2])

        test_crop_stores[indx] = boundingAxes

        # resample or not ??

        ## resample it first
        mask = cv2.resize(mask, (mask.shape[0] // 2, mask.shape[1] // 2), interpolation=cv2.INTER_CUBIC)

        ############################## crop it ####################################

        if mask.shape[0] >= output_size and mask.shape[1] >= output_size:
            print("crop from border ! ")
            mask = mask[:output_size, :output_size]

        else:
            mask = getPaddingBox(mask, out_size=output_size)

        test_img[indx] = np.array([mask])
        test_crop_stores[indx] = boundingAxes
        indx = indx + 1

        # save_path = '/home/jtma/PycharmProjects/dataset/resample_test/'
        # aug_img_name = save_path + img_base_name + '_' + 'center.png'
        # cv2.imwrite(aug_img_name, img_center)


    ###############################################################################

    test_img = test_img / 127.5 - 1.0
    test_img = test_img[:, :, :, np.newaxis]

    assert indx == len(test_img)

    return test_img, test_crop_stores


def merge_to_3d_volume(slice_dir, origin_img_dir, save_dir, img_base_name, strip='_fuse', binary=False):

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    image = readItkData(origin_img_dir + img_base_name + ".nii.gz")
    image_arr = sitk.GetArrayFromImage(image)
    image_arr = np.asarray(image_arr, np.int32)
    image_arr = np.transpose(image_arr, (2, 1, 0))

    output_arr = np.zeros(image_arr.shape, dtype=np.float32)

    for file_name in os.listdir(slice_dir):

        image_base_name = file_name.split(".")[0].strip(strip)
        slice_num = image_base_name[len(image_base_name) - 3: len(image_base_name)]

        img = cv2.imread(os.path.join(slice_dir, file_name), cv2.IMREAD_GRAYSCALE)
        img_np = np.array([img])

        output_arr[:, :, int(slice_num)] = img_np

    origin = image.GetOrigin()
    direction = image.GetDirection()
    spacing = image.GetSpacing()

    output_arr = np.asarray(output_arr, np.float32)
    output_arr = np.transpose(output_arr, (2, 1, 0))

    if binary:

        output_arr[output_arr > 0.5] = 1
        output_arr[output_arr <= 0.5] = 0

    out_image = sitk.GetImageFromArray(output_arr)
    out_image.SetSpacing(spacing)
    out_image.SetDirection(direction)
    out_image.SetOrigin(origin)

    ### post-processing ####

    out_image = binaryThresholdImage(out_image, 0.5)
    out_image = getLargestConnectedComponents(out_image)
    out_image = sitk.BinaryClosingByReconstruction(out_image, [10, 10, 10])

    sitk.WriteImage(out_image, save_dir + img_base_name + ".nii.gz")






