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


from PIL import Image
import cv2

import SimpleITK as sitk
import skimage.io as io
from augmenter import *
from utilities import *


def dice_coef(output, target, loss_type='jaccard', axis=(1, 2, 3), smooth=1e-5):

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


class Pancreas_3D():


    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'Pancreas'
        self.path = data_path
        self.reference_img_name = self.train_img_dir + 'pancreas_129.nii.gz'   # 235

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(
                data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_pancreas(self.train_img_dir, self.train_label_dir, reference_img_dir=self.reference_img_name)

            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1


    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        train_img[train_img < -500] = -500
        train_img[train_img > 400] = 400
        train_img = (train_img - np.min(train_img)) / (np.max(train_img) - np.min(train_img))


        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        val_img[val_img < -500] = -500
        val_img[val_img > 400] = 400
        val_img = (val_img - np.min(val_img)) / (np.max(val_img) - np.min(val_img))

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)

        test_img = test_img[:, :, :, :, np.newaxis]

        test_img[test_img < -500] = -500
        test_img[test_img > 400] = 400
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))

        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):


        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            img = resampleCTData(img, new_spacing=self.new_spacing, is_label=False)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            img_np = histogram_matching(img_np, bins=self.bins, cdf=self.cdf)

            img_save_for_aug = sitk.GetImageFromArray(np.asarray(np.transpose(img_np, (2, 1, 0)), np.float32))
            img_save_for_aug.SetSpacing(img.GetSpacing())
            img_save_for_aug.SetOrigin(img.GetOrigin())
            img_save_for_aug.SetDirection(img.GetDirection())

            ############################################################################################################

            boundingAxes = getBoundingAxes(label_np)

            print(img_base_name, "[", boundingAxes[1]-boundingAxes[0], boundingAxes[3]-boundingAxes[2], boundingAxes[5]-boundingAxes[4], ")")
            #
            left_top_corner = 0
            right_bottom_corner = 0
            step_size = 2
            #
            # ################## crop patches from [X_start, Y_start, Z_start] ###########################################
            #
            num_patches = 0

            x = max(0, boundingAxes[0] - left_top_corner)
            while x < boundingAxes[1]:

                x_end = min(img_np.shape[0], x + self.patch_size[0])
                y = max(0, boundingAxes[2] - left_top_corner)
                while y < boundingAxes[3]:

                    y_end = min(img_np.shape[1], y + self.patch_size[1])
                    z = max(0, boundingAxes[4] - left_top_corner)
                    while z < boundingAxes[5]:

                        z_end = min(img_np.shape[2], z + self.patch_size[2])

                        img_patch = img_np[x:x_end, y:y_end, z:z_end]
                        label_patch = label_np[x:x_end, y:y_end, z:z_end]
                        img_patch, label_patch = flip_rotate(img_patch, label_patch)

                        if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:

                            img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                            label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                        img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                        label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                        sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(num_patches) + ".nii.gz")
                        sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(num_patches) + "_mask.nii.gz")

                        num_patches += 1

                        if z_end >= boundingAxes[5] + right_bottom_corner:
                            break
                        else:
                            z += self.patch_size[2] // step_size   ## step size = 2

                    if y_end >= boundingAxes[3] + right_bottom_corner:
                        break
                    else:
                        y += self.patch_size[1] // step_size

                if x_end >= boundingAxes[1] + right_bottom_corner:
                    break
                else:
                    x += self.patch_size[0] // step_size

            print(" complete writing this image with number of patches = ", num_patches)
            #
            # #############################################################################################################
            #
            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.7)
                s_y = np.random.uniform(0.5, 1.7)
                s_z = np.random.uniform(0.5, 1.7)

                r_d = np.random.uniform(10, 80)

                img_aug, label_aug = augmentCTData(img_save_for_aug, label, scale=(s_x, s_y, s_z), rotate=r_d)

                img_aug_np = np.transpose(sitk.GetArrayFromImage(img_aug), (2, 1, 0)).astype(np.float32)
                label_aug_np = np.transpose(sitk.GetArrayFromImage(label_aug), (2, 1, 0)).astype(np.float32)

                if np.count_nonzero(label_aug_np.flatten()) < 50 or np.max(label_aug_np.flatten()) < 2:
                    print(" !!! not write this augmented image !!! ")
                    continue

                ###################### this could make mistake ! the histogram matching ####################################

                boundingAxes = getBoundingAxes(label_aug_np)

                num_patches = 0
                step_size = 2

                x = max(0, boundingAxes[0])
                while x < boundingAxes[1]:

                    x_end = min(img_np.shape[0], x + self.patch_size[0])
                    y = max(0, boundingAxes[2])
                    while y < boundingAxes[3]:

                        y_end = min(img_np.shape[1], y + self.patch_size[1])
                        z = max(0, boundingAxes[4])
                        while z < boundingAxes[5]:

                            z_end = min(img_np.shape[2], z + self.patch_size[2])

                            img_patch = img_aug_np[x:x_end, y:y_end, z:z_end]
                            label_patch = label_aug_np[x:x_end, y:y_end, z:z_end]
                            img_patch, label_patch = flip_rotate(img_patch, label_patch)

                            if np.max(label_patch) > 1 and np.count_nonzero(label_patch) >= 50:

                                if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:
                                    img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                                    label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                                if write_to_write_out_ct(img_patch):

                                    img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                                    label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                                    sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + "_" + str(num_patches) + ".nii.gz")
                                    sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_" + str(num_patches) + "_mask.nii.gz")

                                    num_patches += 1

                            if z_end >= boundingAxes[5]:
                                break
                            else:
                                z += self.patch_size[2] // step_size  ## step size = 2

                        if y_end >= boundingAxes[3]:
                            break
                        else:
                            y += self.patch_size[1] // step_size

                    if x_end >= boundingAxes[1]:
                        break
                    else:
                        x += self.patch_size[0] // step_size

                print(" complete augmenting this augmented image with number of patches = ", num_patches)

        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleCTData(test_img, new_spacing=self.new_spacing, is_label=False)

        test_origin_img = readItkData(self.test_img_origin_dir + test_name)
        test_origin_img = resampleCTData(test_origin_img, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        test_origin_img_np = np.transpose(sitk.GetArrayFromImage(test_origin_img), (2, 1, 0)).astype(np.float32)
        test_origin_img_np = histogram_matching(test_origin_img_np, bins=self.bins, cdf=self.cdf)

        patch_array, coords_array = generate_offline_test_batch_CT(origin_img_np=test_origin_img_np, img_np=test_img_np,
                                                                   patch_size=self.patch_size, step_size=step_size,
                                                                      test_name=test_img_base_name)  ## half of the patch size

        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ").nii.gz")

        return patch_array, coords_array


class Prostate_3D():

    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, channel=2, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.name = 'Prostate'
        self.path = data_path


        self.bins, self.cdfs = [], []
        for c in range(channel):

            if os.path.exists(data_path + '/imhist' + str(c) + '.txt') and \
                    os.path.exists(data_path + '/bins' + str(c) + '.txt') and os.path.exists(
                    data_path + '/cdf' + str(c) + '.txt'):

                print(" loading histogram information from path ... ")

                self.imhist = np.loadtxt(data_path + '/imhist' + str(c) + '.txt')
                self.bins.append(np.loadtxt(data_path + '/bins' + str(c) + '.txt'))
                self.cdfs.append(np.loadtxt(data_path + '/cdf' + str(c) + '.txt'))

            else:

                imhist, bins, cdf = compute_histogram_of_train_images_multi(self.train_img_dir,
                                                                                          self.train_label_dir,
                                                                                          channel=c,
                                                                                          num_bins=num_bins)

                np.savetxt(data_path + '/imhist' + str(c) + '.txt', imhist, fmt="%s")
                np.savetxt(data_path + '/bins' + str(c) + '.txt', bins, fmt="%s")
                np.savetxt(data_path + '/cdf' + str(c) + '.txt', cdf, fmt="%s")

                self.bins.append(bins)
                self.cdfs.append(cdf)

        print("the histogram information within this class : ", len(self.bins), len(self.cdfs))

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)   # os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)
        self.test_img_origin_files = os.listdir(self.test_img_origin_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(os.listdir(self.test_img_dir)))

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train
        #
        self.patch_size = patch_size
        self.channel = channel
        self.new_spacing = resampled_spacing

    # get_train_batch
    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData_multiclass(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            ############################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        ################################################################################

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData_multiclass(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ############################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            ############################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        ################################################################################################################

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img = readItkData_multiclass(self.test_img_dir + batch_img_files[file_index])
            img = resampleItkDataMulti(img, new_spacing=self.new_spacing)

            ############################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            boundingAxes, mask_np = findMaskBoundingBox_multi(img_np, thresIntens=10)  # mri mode

            if mask_np.shape[0] > self.patch_size[0] or mask_np.shape[1] > self.patch_size[0] or mask_np.shape[2] > self.patch_size[2]:

                mask_np = cropCenter_multi(mask_np, out_size=self.patch_size)

            if mask_np.shape[0] < self.patch_size[0] and mask_np.shape[1] < self.patch_size[1] and mask_np.shape[2] < self.patch_size[2]:

                mask_np = getPaddingBox_multi(mask_np, out_size=self.patch_size, channel=mask_np.shape[3])

            patch_img_stack = np.array([mask_np])
            test_img[indx] = patch_img_stack
            indx += 1

        ################################################################################

        assert indx == len(test_img)
        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=10):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData_multiclass(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            ## resample and cropping ####

            img_r = resampleItkDataMulti(img, new_spacing=self.new_spacing)
            label_r = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)

            img_np = sitk.GetArrayFromImage(img_r)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            label_np = sitk.GetArrayFromImage(label_r)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            print(img_base_name, " : ")
            center_patch_img, center_patch_label, boundingAxes = crop_center_from_label_multi(img_np, label_np,
                                                                                              patch_size=self.patch_size,
                                                                                              channel=self.channel)

            print("start augmenting image = ", img_base_name, boundingAxes, "(", boundingAxes[1] - boundingAxes[0],
                  boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4], ")")

            ## histogram matching ##

            img_slices = []
            for c in range(self.channel):

                img_t = center_patch_img[:, :, :, c]

                img_t = histogram_matching(img_t, bins=self.bins[c], cdf=self.cdfs[c])

                img_slices.append(sitk.GetImageFromArray(np.transpose(img_t, (2, 1, 0))).astype(np.float32))

            img_patch_out = sitk.JoinSeries(img_slices)
            label_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_label, (2, 1, 0)).astype(np.float32))

            sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug.nii.gz")
            sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_mask.nii.gz")

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.8)
                s_y = np.random.uniform(0.5, 1.8)
                s_z = np.random.uniform(0.5, 1.8)

                r_d = np.random.uniform(30, 70)

                img_aug, label_aug = augmentMRIData(img_patch_out, label_patch_out, (s_x, s_y, s_z), r_d)

                if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 50:  # if img_aug and label_aug:

                    sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + ".nii.gz")
                    sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_mask.nii.gz")

                else:
                    print("not writing this batch ")

        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData_multiclass(test_img_dir + test_name)
        test_img = resampleItkDataMulti(test_img, new_spacing=self.new_spacing)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (3, 2, 1, 0)).astype(np.float32)

        patch_array, coords_array = generate_offline_test_batch_multi(img_cp_np=test_img_np,
                                                                      patch_size=self.patch_size, step_size=step_size,
                                                                      cdfs=self.cdfs, bins=self.bins,
                                                                      test_name=test_img_base_name)  ## half of the patch size

        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):
                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                patch_img = write_multi_channel_image(patch)
                sitk.WriteImage(patch_img, save_dir + test_img_base_name + "_" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ".nii.gz")

        return patch_array, coords_array


class BrainTumor_3D():

    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, channel=4, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'BrainTumor'
        self.path = data_path

        self.bins, self.cdfs = [], []
        for c in range(channel):

            if os.path.exists(data_path + '/imhist' + str(c) + '.txt') and \
                    os.path.exists(data_path + '/bins' + str(c) + '.txt') and os.path.exists(
                    data_path + '/cdf' + str(c) + '.txt'):

                print(" loading histogram information from path ... ")

                self.imhist = np.loadtxt(data_path + '/imhist' + str(c) + '.txt')
                self.bins.append(np.loadtxt(data_path + '/bins' + str(c) + '.txt'))
                self.cdfs.append(np.loadtxt(data_path + '/cdf' + str(c) + '.txt'))

            else:

                imhist, bins, cdf = compute_histogram_of_train_images_multi(self.train_img_dir, self.train_label_dir,
                                                                                           channel=c, num_bins=num_bins)

                np.savetxt(data_path + '/imhist' + str(c) + '.txt', imhist, fmt="%s")
                np.savetxt(data_path + '/bins' + str(c) + '.txt', bins, fmt="%s")
                np.savetxt(data_path + '/cdf' + str(c) + '.txt', cdf, fmt="%s")

                self.bins.append(bins)
                self.cdfs.append(cdf)


        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)
        #
        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(self.test_img_files))
        #
        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]
        #
        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.channel = channel
        self.new_spacing = resampled_spacing

    # get_train_batch
    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData_multiclass(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            # center_patch_img, center_patch_label, boundingAxes = crop_center_from_label_multi(img_np, label_np, patch_size=self.patch_size, channel=self.channel)
            # center_patch_img = normalization_multi(center_patch_img)

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

 #       train_img = train_img[:, :, :, :, np.newaxis]
 #       train_label = train_label[:, :, :, :, np.newaxis]

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData_multiclass(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ############################## resample it ################################

            # img_np = sitk.GetArrayFromImage(img)
            # img_np = np.asarray(img_np, np.float32)
            # img_np = np.transpose(img_np, (3, 2, 1, 0))
            #
            # t_slices = []
            # for t in range(img_np.shape[3]):
            #     img_t = img_np[:, :, :, t]
            #     img_t = np.transpose(img_t, (2, 1, 0))
            #     img_t = sitk.GetImageFromArray(img_t)
            #     img_t = resampleItkData(img_t, new_spacing=self.new_spacing, is_label=False)
            #
            #     t_slices.append(img_t)
            #
            # img_out = sitk.JoinSeries(t_slices)
            # img_out.SetOrigin(img.GetOrigin())
            # img_out.SetDirection(img.GetDirection())
            #
            # ### prepare label ####
            # label_np = sitk.GetArrayFromImage(label)
            # label_np = np.asarray(label_np, np.int32)
            # label_out = resampleItkData(sitk.GetImageFromArray(label_np), new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)
            #
            # label_out.SetOrigin(label.GetOrigin())
            # label_out.SetDirection(label.GetDirection())

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################
            #
            # center_patch_img, center_patch_label = crop_center_multi(img_np, label_np, patch_size=self.patch_size, channel=self.channel)
            # center_patch_img, center_patch_label, _ = crop_center_from_label_multi(img_np, label_np, patch_size=self.patch_size, channel=self.channel)
            #
            # #center_patch_img[center_patch_img < 500] = 0
            # center_patch_img = normalization_multi(center_patch_img)

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1


        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2], self.channel), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData_multiclass(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            val_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(val_img)
        return val_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData_multiclass(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            ## resample and cropping ####

            img = resampleItkDataMulti(img, new_spacing=self.new_spacing)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (3, 2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            center_patch_img, center_patch_label, boundingAxes = crop_center_from_label_multi(img_np, label_np,
                                                                                   patch_size=self.patch_size,
                                                                                   channel=self.channel)

            print("start augmenting image = ", img_base_name, boundingAxes, "(", boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4], ")")

            ## histogram matching ##

            img_slices = []
            for c in range(self.channel):

                img_t = center_patch_img[:, :, :, c]
                img_t = histogram_matching(img_t, bins=self.bins[c], cdf=self.cdfs[c])

                img_slices.append(sitk.GetImageFromArray(np.transpose(img_t, (2, 1, 0))).astype(np.float32))

            img_patch_out = sitk.JoinSeries(img_slices)
            label_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_label, (2, 1, 0)).astype(np.float32))

            sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug.nii.gz")
            sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_mask.nii.gz")

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.8)
                s_y = np.random.uniform(0.5, 1.8)
                s_z = np.random.uniform(0.5, 1.8)

                r_d = np.random.uniform(30, 70)

                img_aug, label_aug = augmentMRIData(img_patch_out, label_patch_out, (s_x, s_y, s_z), r_d)

                if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 50:   #   if img_aug and label_aug:

                    sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + ".nii.gz")
                    sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_mask.nii.gz")

                else: print("not writing this batch ")

        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData_multiclass(test_img_dir + test_name)
        test_img = resampleItkDataMulti(test_img, new_spacing=self.new_spacing)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (3, 2, 1, 0)).astype(np.float32)

        patch_array, coords_array = generate_offline_test_batch_multi(img_cp_np=test_img_np,
                                                                      patch_size=self.patch_size, step_size=step_size,
                                                                      cdfs=self.cdfs, bins=self.bins,
                                                                      test_name=test_img_base_name)  ## half of the patch size

        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                patch_img = write_multi_channel_image(patch)
                sitk.WriteImage(patch_img, save_dir + test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ").nii.gz")

        return patch_array, coords_array


class Heart_3D():

    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.name = 'Heart'
        self.path = data_path

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_train_images(self.train_img_dir, self.train_label_dir, num_bins=num_bins)
            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(os.listdir(self.test_img_dir)))

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]
        self.test_img_files = os.listdir(self.test_img_dir)

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1


    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, test_img_dir, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        test_names = []

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            test_names.append(img_base_name)
            # print(img_base_name)

            img = readItkData(test_img_dir + batch_img_files[file_index])
            img_r = resampleMRIData(img, new_spacing=self.new_spacing, is_label=False)

            # sitk.WriteImage(img_r, "/home/jtma/PycharmProjects/dataset/test/" + img_base_name + "_r.nii.gz")

            # ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img_r)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            #boundingAxes = findROI(img_np, thresIntens=10)
            boundingAxes = findMaskBoundingBox_MRI(img_np, thresIntens=0)
            mask_np = img_np[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3], boundingAxes[4]:boundingAxes[5]]

            print(img_base_name, "bounding axes = ", boundingAxes, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])

            ################## histogram matching ##########################

            mask_shape = mask_np.shape

            s_values, bin_indx, s_counts = np.unique(mask_np.flatten(), return_inverse=True, return_counts=True)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]

            interp_t_values = np.interp(s_quantiles, self.cdf, self.bins)   # t_values in origin
            mask_np = interp_t_values[bin_indx]                     # mask_np = np.interp(mask_np.flatten(), self.bins[:-1], self.cdf)
            mask_np = np.reshape(mask_np, mask_shape)

            if mask_np.shape[0] > self.patch_size[0] or mask_np.shape[1] > self.patch_size[1] or mask_np.shape[2] > self.patch_size[2]:

                print("crop from center")
                mask_np = cropCenter_MRI(mask_np, out_size=self.patch_size)

            if mask_np.shape[0] < self.patch_size[0] or mask_np.shape[1] < self.patch_size[1] or mask_np.shape[2] < self.patch_size[2]:

                print("get padding box")
                mask_np = getPaddingBox_MRI(mask_np, out_size=self.patch_size)

            #######################################################################################################################################

            patch_img_stack = np.array([mask_np])
            test_img[indx] = patch_img_stack
            indx += 1

        #################################################################################

        test_img = test_img[:, :, :, :, np.newaxis]  # [batch, h, w, d, channel]

        assert indx == len(test_img)
        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            print(self.name, ": start augmenting image = ", img_base_name)

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            ## resample and cropping ####

            img = resampleMRIData(img, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            center_patch_img, center_patch_label, boundingAxes = crop_center_from_label_MRI(img_np, label_np, patch_size=self.patch_size)
            print(" after resampling - ", boundingAxes, "(", boundingAxes[1] - boundingAxes[0],
                  boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4], ")")

            s_values, bin_indx, s_counts = np.unique(center_patch_img.flatten(), return_inverse=True, return_counts=True)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]

            interp_t_values = np.interp(s_quantiles, self.cdf, self.bins)  # t_values in origin
            center_patch_img = interp_t_values[bin_indx]  # center_patch_img = np.interp(center_patch_img.flatten(), bins[:-1], cdf)

            center_patch_img = np.reshape(center_patch_img, self.patch_size)

            ###########################################################################

            img_patch_out = np.transpose(center_patch_img, (2, 1, 0)).astype(np.float32)
            img_patch_out = sitk.GetImageFromArray(img_patch_out)
            label_patch_out = np.transpose(center_patch_label, (2, 1, 0)).astype(np.float32)
            label_patch_out = sitk.GetImageFromArray(label_patch_out)

            sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug.nii.gz")   # img_origin
            sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_mask.nii.gz")

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.8)
                s_y = np.random.uniform(0.5, 1.8)
                s_z = np.random.uniform(0.5, 1.8)

                r_d = np.random.uniform(30, 70)

                img_aug, label_aug = augmentMRIData(img_patch_out, label_patch_out, (s_x, s_y, s_z), r_d)

                if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 20:   #   if img_aug and label_aug:

                    sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + ".nii.gz")
                    sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_mask.nii.gz")

                else:

                    print("not writing this batch ")

        print( self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleMRIData(test_img, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        patch_array, coords_array = generate_offline_test_batch_MRI(img_np=test_img_np, patch_size=self.patch_size, step_size=step_size,
                                                                      cdf=self.cdf, bins=self.bins, test_name=test_img_base_name)  ## half of the patch size


        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "_" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ".nii.gz")

        return patch_array, coords_array


class Spleen_3D():


    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.name = 'Spleen'
        self.path = data_path
        self.reference_img_dir = self.train_img_dir + 'spleen_14.nii.gz'

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_reference(self.train_img_dir, self.train_label_dir,
                                                                              reference_img_dir=self.reference_img_dir)

            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1

    # get_train_batch

    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        train_img[train_img < -200] = -200
        train_img[train_img > 300] = 300
        train_img = (train_img - np.min(train_img)) / (np.max(train_img) - np.min(train_img))

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        val_img[val_img < -200] = -200
        val_img[val_img > 300] = 300
        val_img = (val_img - np.min(val_img)) / (np.max(val_img) - np.min(val_img))

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)
        test_img = test_img[:, :, :, :, np.newaxis]

        test_img[test_img < -200] = -200
        test_img[test_img > 300] = 300
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))

        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):


        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            img = resampleCTData(img, new_spacing=self.new_spacing, is_label=False)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            img_np = histogram_matching(img_np, bins=self.bins, cdf=self.cdf)

            img_save_for_aug = sitk.GetImageFromArray(np.asarray(np.transpose(img_np, (2, 1, 0)), np.float32))
            img_save_for_aug.SetSpacing(img.GetSpacing())
            img_save_for_aug.SetOrigin(img.GetOrigin())
            img_save_for_aug.SetDirection(img.GetDirection())

            ############################################################################################################

            boundingAxes = getBoundingAxes(label_np)

            print(img_base_name, "[", boundingAxes[1]-boundingAxes[0], boundingAxes[3]-boundingAxes[2], boundingAxes[5]-boundingAxes[4], ")")
            #
            left_top_corner = 5
            right_bottom_corner = 5
            step_size = 2
            #
            # ################## crop patches from [X_start, Y_start, Z_start] ###########################################
            #
            num_patches = 0

            x = max(0, boundingAxes[0] - left_top_corner)
            while x < boundingAxes[1]:

                x_end = min(img_np.shape[0], x + self.patch_size[0])
                y = max(0, boundingAxes[2] - left_top_corner)
                while y < boundingAxes[3]:

                    y_end = min(img_np.shape[1], y + self.patch_size[1])
                    z = max(0, boundingAxes[4] - left_top_corner)
                    while z < boundingAxes[5]:

                        z_end = min(img_np.shape[2], z + self.patch_size[2])

                        img_patch = img_np[x:x_end, y:y_end, z:z_end]
                        label_patch = label_np[x:x_end, y:y_end, z:z_end]
                        img_patch, label_patch = flip_rotate(img_patch, label_patch)

                        if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:

                            img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                            label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                        img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                        label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                        sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(num_patches) + ".nii.gz")
                        sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(num_patches) + "_mask.nii.gz")

                        num_patches += 1

                        if z_end >= boundingAxes[5] + right_bottom_corner:
                            break
                        else:
                            z += self.patch_size[2] // step_size   ## step size = 2

                    if y_end >= boundingAxes[3] + right_bottom_corner:
                        break
                    else:
                        y += self.patch_size[1] // step_size

                if x_end >= boundingAxes[1] + right_bottom_corner:
                    break
                else:
                    x += self.patch_size[0] // step_size

            print(" complete writing this image with number of patches = ", num_patches)
            #
            # #############################################################################################################
            #
            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.7)
                s_y = np.random.uniform(0.5, 1.7)
                s_z = np.random.uniform(0.5, 1.7)

                r_d = np.random.uniform(10, 80)

                img_aug, label_aug = augmentCTData(img_save_for_aug, label, scale=(s_x, s_y, s_z), rotate=r_d)

                img_aug_np = np.transpose(sitk.GetArrayFromImage(img_aug), (2, 1, 0)).astype(np.float32)
                label_aug_np = np.transpose(sitk.GetArrayFromImage(label_aug), (2, 1, 0)).astype(np.float32)

                if np.count_nonzero(label_aug_np.flatten()) < 50:
                    print(" !!! not write this augmented image !!! ")
                    continue

                ###################### this could make mistake ! the histogram matching ####################################

                boundingAxes = getBoundingAxes(label_aug_np)

                num_patches = 0
                step_size = 2

                x = max(0, boundingAxes[0])
                while x < boundingAxes[1]:

                    x_end = min(img_np.shape[0], x + self.patch_size[0])
                    y = max(0, boundingAxes[2])
                    while y < boundingAxes[3]:

                        y_end = min(img_np.shape[1], y + self.patch_size[1])
                        z = max(0, boundingAxes[4])
                        while z < boundingAxes[5]:

                            z_end = min(img_np.shape[2], z + self.patch_size[2])

                            img_patch = img_aug_np[x:x_end, y:y_end, z:z_end]
                            label_patch = label_aug_np[x:x_end, y:y_end, z:z_end]
                            img_patch, label_patch = flip_rotate(img_patch, label_patch)

                            if np.count_nonzero(label_patch) >= 50:

                                if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:
                                    img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                                    label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                                if write_to_write_out_ct(img_patch):

                                    img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                                    label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                                    sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + "_" + str(num_patches) + ".nii.gz")
                                    sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_" + str(num_patches) + "_mask.nii.gz")

                                    num_patches += 1

                            if z_end >= boundingAxes[5]:
                                break
                            else:
                                z += self.patch_size[2] // step_size  ## step size = 2

                        if y_end >= boundingAxes[3]:
                            break
                        else:
                            y += self.patch_size[1] // step_size

                    if x_end >= boundingAxes[1]:
                        break
                    else:
                        x += self.patch_size[0] // step_size

                print(" complete augmenting this augmented image with number of patches = ", num_patches)

        print(self.name, " : augmentation is done .. ")


    #####################################################################################################################################


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_origin_img = readItkData(self.test_img_origin_dir + test_name)

        test_img = resampleCTData(test_img, new_spacing=self.new_spacing, is_label=False)
        test_origin_img = resampleCTData(test_origin_img, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0))

        test_origin_img_np = np.transpose(sitk.GetArrayFromImage(test_origin_img), (2, 1, 0))
        test_origin_img_np = histogram_matching(test_origin_img_np, bins=self.bins, cdf=self.cdf)

        patch_array, coords_array = generate_offline_test_batch_spleen(origin_img_np=test_origin_img_np, img_np=test_img_np,
                                                                       patch_size=self.patch_size, step_size=step_size,
                                                                      test_name=test_img_base_name)  ## half of the patch size

        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ").nii.gz")

        return patch_array, coords_array


class Liver_3D():

    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'Liver'
        self.path = data_path

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(
                data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_reference(self.train_img_dir, self.train_label_dir)

            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(self.test_img_files))

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1

    # get_train_batch

    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)

        test_img = test_img[:, :, :, :, np.newaxis]
        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            img = resampleCTData(img, new_spacing=self.new_spacing, is_label=False)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            img_np = histogram_matching(img_np, cdf=self.cdf, bins=self.bins)

            ############################################################################################################

            boundingAxes = getBoundingAxes(label_np)

            print(img_base_name, "[", boundingAxes[1]-boundingAxes[0], boundingAxes[3]-boundingAxes[2], boundingAxes[5]-boundingAxes[4], ")")

            left_top_corner = 0
            right_bottom_corner = 0
            step_size = 2

            ################## crop patches from [X_start, Y_start, Z_start] ###########################################

            num_patches = 0

            x = max(0, boundingAxes[0] - left_top_corner)
            while x < boundingAxes[1]:

                x_end = min(img_np.shape[0], x + self.patch_size[0])
                y = max(0, boundingAxes[2] - left_top_corner)
                while y < boundingAxes[3]:

                    y_end = min(img_np.shape[1], y + self.patch_size[1])
                    z = max(0, boundingAxes[4] - left_top_corner)
                    while z < boundingAxes[5]:

                        z_end = min(img_np.shape[2], z + self.patch_size[2])

                        img_patch = img_np[x:x_end, y:y_end, z:z_end]
                        label_patch = label_np[x:x_end, y:y_end, z:z_end]

                        if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:

                            img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                            label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                        img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                        label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                        sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(num_patches) + ".nii.gz")
                        sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(num_patches) + "_mask.nii.gz")

                        num_patches += 1

                        if z_end >= boundingAxes[5] + right_bottom_corner:
                            break
                        else:
                            z += self.patch_size[2] // step_size   ## step size = 2

                    if y_end >= boundingAxes[3] + right_bottom_corner:
                        break
                    else:
                        y += self.patch_size[1] // step_size

                if x_end >= boundingAxes[1] + right_bottom_corner:
                    break
                else:
                    x += self.patch_size[0] // step_size

            print(" complete augmenting this image with number of patches = ", num_patches)

            #############################################################################################################

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.7)
                s_y = np.random.uniform(0.5, 1.7)
                s_z = np.random.uniform(0.5, 1.7)

                # rotate_angle = [90, 180, 270]
                # r_d = np.random.choice(rotate_angle, 1)  # r_d = np.random.uniform(30, 70)

                img_aug, label_aug = augmentCTData(img, label, scale=(s_x, s_y, s_z), rotate=None)

                img_aug_np = np.transpose(sitk.GetArrayFromImage(img_aug), (2, 1, 0)).astype(np.float32)
                label_aug_np = np.transpose(sitk.GetArrayFromImage(label_aug), (2, 1, 0)).astype(np.float32)

                ### random flipping and rotate 90 ##

                img_aug_np, label_aug_np = flip_rotate(img_aug_np, label_aug_np)
                img_aug_np = histogram_matching(img_aug_np, bins=self.bins, cdf=self.cdf)

                if np.count_nonzero(label_aug_np) < 50:
                    print(" !!! not write this augmented image !!! ")
                    continue

                boundingAxes = getBoundingAxes(label_aug_np)

                ################## crop patches from [X_start, Y_start, Z_start] ###########################################

                num_patches = 0
                step_size = 2

                x = max(0, boundingAxes[0])
                while x < boundingAxes[1]:

                    x_end = min(img_np.shape[0], x + self.patch_size[0])
                    y = max(0, boundingAxes[2])
                    while y < boundingAxes[3]:

                        y_end = min(img_np.shape[1], y + self.patch_size[1])
                        z = max(0, boundingAxes[4])
                        while z < boundingAxes[5]:

                            z_end = min(img_np.shape[2], z + self.patch_size[2])

                            img_patch = img_aug_np[x:x_end, y:y_end, z:z_end]
                            label_patch = label_aug_np[x:x_end, y:y_end, z:z_end]

                            if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:

                                img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                                label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                            img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                            label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                            if np.max(label_patch) > 1:

                                sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + "_" + str(num_patches) + ".nii.gz")
                                sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_" + str(num_patches) + "_mask.nii.gz")

                            num_patches += 1

                            if z_end >= boundingAxes[5]:
                                break
                            else:
                                z += self.patch_size[2] // step_size  ## step size = 2

                        if y_end >= boundingAxes[3]:
                            break
                        else:
                            y += self.patch_size[1] // step_size

                    if x_end >= boundingAxes[1]:
                        break
                    else:
                        x += self.patch_size[0] // step_size

                print("complete augmenting this augmented image with number of patches = ", num_patches)

        print(self.name, " : augmentation is done .. ")

        # for file_index in range(len(batch_img_files)):
        #
        #     img_base_name = batch_img_files[file_index].split('.')[0]
        #     label_base_name = img_base_name + "_mask"
        #     ext = '.nii.gz'
        #
        #     img = readItkData(train_img_dir + batch_img_files[file_index])
        #     label = readItkLabelData(train_label_dir + label_base_name + ext)
        #
        #     img = resampleCTData(img, new_spacing=self.new_spacing, is_label=False)
        #     label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)
        #
        #     img_np = sitk.GetArrayFromImage(img)
        #     img_np = np.asarray(img_np, np.float32)
        #     img_np = np.transpose(img_np, (2, 1, 0))
        #
        #     label_np = sitk.GetArrayFromImage(label)
        #     label_np = np.asarray(label_np, np.int32)
        #     label_np = np.transpose(label_np, (2, 1, 0))
        #
        #     ############################################################################################################
        #
        #     boundingAxes = getBoundingAxes(label_np)
        #
        #     print(img_base_name, "[", boundingAxes[1]-boundingAxes[0], boundingAxes[3]-boundingAxes[2], boundingAxes[5]-boundingAxes[4], ")")
        #
        #     left_top_corner = 10
        #     step_size = 2
        #
        #     ################## crop patches from [X_start, Y_start, Z_start] ###########################################
        #
        #     num_patches = 0
        #
        #     x = max(0, boundingAxes[0] - left_top_corner)
        #     while x < boundingAxes[1]:
        #
        #         x_end = min(img_np.shape[0], x + self.patch_size[0])
        #         y = max(0, boundingAxes[2] - left_top_corner)
        #         while y < boundingAxes[3]:
        #
        #             y_end = min(img_np.shape[1], y + self.patch_size[1])
        #             z = max(0, boundingAxes[4] - left_top_corner)
        #             while z < boundingAxes[5]:
        #
        #                 z_end = min(img_np.shape[2], z + self.patch_size[2])
        #
        #                 img_patch = img_np[x:x_end, y:y_end, z:z_end]
        #                 label_patch = label_np[x:x_end, y:y_end, z:z_end]
        #
        #                 img_patch = histogram_matching(img_patch, bins=self.bins, cdf=self.cdf)
        #
        #                 if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:
        #
        #                     img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
        #                     label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)
        #
        #                 img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)))
        #                 label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)))
        #
        #                 sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(num_patches) + ".nii.gz")
        #                 sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(num_patches) + "_mask.nii.gz")
        #
        #                 num_patches += 1
        #
        #                 if np.max(label_patch) > 1:
        #
        #                     for indx in range(num_aug):
        #
        #                         s_x = np.random.uniform(0.5, 1.7)
        #                         s_y = np.random.uniform(0.5, 1.7)
        #                         s_z = np.random.uniform(0.5, 1.7)
        #                         r_d = np.random.uniform(30, 70)
        #
        #                         img_aug, label_aug = augmentCTData(img_patch_out, label_patch_out, scale=(s_x, s_y, s_z), rotate=r_d)
        #
        #                         if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 10:
        #
        #                             sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_" + str(num_patches) + "_" + str(indx) + ".nii.gz")
        #                             sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_" + str(num_patches) + "_" + str(indx) + "_mask.nii.gz")
        #
        #                 print("complete augmenting this patch")
        #
        #
        #                 if z_end >= boundingAxes[5]:
        #                     break
        #                 else:
        #                     z += self.patch_size[2] // step_size   ## step size = 2
        #
        #             if y_end >= boundingAxes[3]:
        #                 break
        #             else:
        #                 y += self.patch_size[1] // step_size
        #
        #         if x_end >= boundingAxes[1]:
        #             break
        #         else:
        #             x += self.patch_size[0] // step_size
        #
        #     print("complete augmenting this image with number of patches = ", num_patches)
        #
        # print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleCTData(test_img, new_spacing=self.new_spacing, is_label=False)

        test_img_origin = readItkData(self.test_img_origin_dir + test_name)
        test_img_origin = resampleCTData(test_img_origin, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        test_img_origin_np = np.transpose(sitk.GetArrayFromImage(test_img_origin), (2, 1, 0)).astype(np.float32)
        test_img_origin_np = histogram_matching(test_img_origin_np, bins=self.bins, cdf=self.cdf)

        patch_array, coords_array = generate_offline_test_batch_CT(origin_img_np=test_img_origin_np, img_np=test_img_np,
                                                                   patch_size=self.patch_size, step_size=step_size,
                                                                      cdf=self.cdf, bins=self.bins, test_name=test_img_base_name)  ## half of the patch size


        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ").nii.gz")

        return patch_array, coords_array


class HepaticVessel_3D():


    def __init__(self, data_path, patch_size=(64, 64, 64), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'HepaticVessel'
        self.path = data_path
        self.reference_img = self.train_img_dir + '/hepaticvessel_007.nii.gz'

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(self.test_img_files))

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(
                data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_reference(self.train_img_dir, self.train_label_dir,
                                                                              reference_img_dir=self.reference_img)

            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

    # get_train_batch

    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        train_img[train_img < -500] = -500
        train_img[train_img > 400] = 400
        train_img = (train_img - np.min(train_img)) / (np.max(train_img) - np.min(train_img))

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        val_img[val_img < -500] = -500
        val_img[val_img > 400] = 400
        val_img = (val_img - np.min(val_img)) / (np.max(val_img) - np.min(val_img))

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)

        test_img = test_img[:, :, :, :, np.newaxis]

        test_img[test_img < -500] = -500
        test_img[test_img > 400] = 400
        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))

        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            img = resampleCTData(img, new_spacing=self.new_spacing, is_label=False)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            img_np = histogram_matching(img_np, bins=self.bins, cdf=self.cdf)

            img_save_for_aug = sitk.GetImageFromArray(np.asarray(np.transpose(img_np, (2, 1, 0)), np.float32))
            img_save_for_aug.SetSpacing(img.GetSpacing())
            img_save_for_aug.SetOrigin(img.GetOrigin())
            img_save_for_aug.SetDirection(img.GetDirection())

            ############################################################################################################

            boundingAxes = getBoundingAxes(label_np)

            print(img_base_name, boundingAxes, "[", boundingAxes[1]-boundingAxes[0], boundingAxes[3]-boundingAxes[2], boundingAxes[5]-boundingAxes[4], "]")

            left_top_corner = 0
            step_size = 2

            ################## crop patches from [X_start, Y_start, Z_start] ###########################################

            num_patches = 0

            x = max(0, boundingAxes[0] - left_top_corner)
            while x < boundingAxes[1]:

                x_end = min(img_np.shape[0], x + self.patch_size[0])
                y = max(0, boundingAxes[2] - left_top_corner)
                while y < boundingAxes[3]:

                    y_end = min(img_np.shape[1], y + self.patch_size[1])
                    z = max(0, boundingAxes[4] - left_top_corner)
                    while z < boundingAxes[5]:

                        z_end = min(img_np.shape[2], z + self.patch_size[2])

                        img_patch = img_np[x:x_end, y:y_end, z:z_end]
                        label_patch = label_np[x:x_end, y:y_end, z:z_end]

                        if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:

                            img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                            label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                        img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                        label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                        sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(num_patches) + ".nii.gz")
                        sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(num_patches) + "_mask.nii.gz")

                        num_patches += 1

                        if z_end >= boundingAxes[5]:
                            break
                        else:
                            z += self.patch_size[2] // step_size   ## step size = 2

                    if y_end >= boundingAxes[3]:
                        break
                    else:
                        y += self.patch_size[1] // step_size

                if x_end >= boundingAxes[1]:
                    break
                else:
                    x += self.patch_size[0] // step_size

            print("complete writing this image with number of patches = ", num_patches)

            #############################################################################################################

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.7)
                s_y = np.random.uniform(0.5, 1.7)
                s_z = np.random.uniform(0.5, 1.7)
                r_d = np.random.uniform(10, 80)

                img_aug, label_aug = augmentCTData(img_save_for_aug, label, scale=(s_x, s_y, s_z), rotate=r_d)   # img

                img_aug_np = np.transpose(sitk.GetArrayFromImage(img_aug), (2, 1, 0)).astype(np.float32)
                label_aug_np = np.transpose(sitk.GetArrayFromImage(label_aug), (2, 1, 0)).astype(np.float32)

                ### random flipping and rotate 90 ##

                img_aug_np, label_aug_np = flip_rotate(img_aug_np, label_aug_np)
                # img_aug_np = histogram_matching(img_aug_np, bins=self.bins, cdf=self.cdf)

                if np.count_nonzero(label_aug_np) < 50:
                    print(" !!! not write this augmented image !!! ")
                    continue

                boundingAxes = getBoundingAxes(label_aug_np)

                ################## crop patches from [X_start, Y_start, Z_start] ###########################################

                num_patches = 0
                step_size = 1

                x = max(0, boundingAxes[0])
                while x < boundingAxes[1]:

                    x_end = min(img_np.shape[0], x + self.patch_size[0])
                    y = max(0, boundingAxes[2])
                    while y < boundingAxes[3]:

                        y_end = min(img_np.shape[1], y + self.patch_size[1])
                        z = max(0, boundingAxes[4])
                        while z < boundingAxes[5]:

                            z_end = min(img_np.shape[2], z + self.patch_size[2])

                            img_patch = img_aug_np[x:x_end, y:y_end, z:z_end]
                            label_patch = label_aug_np[x:x_end, y:y_end, z:z_end]

                            if np.max(label_patch) > 1 and np.count_nonzero(label_patch) > 70:

                                if img_patch.shape[0] < self.patch_size[0] or img_patch.shape[1] < self.patch_size[1] or img_patch.shape[2] < self.patch_size[2]:

                                    img_patch = getPaddingBox_CT(img_patch, out_size=self.patch_size)
                                    label_patch = getPaddingBox_MRI(label_patch, out_size=self.patch_size)

                                if write_to_write_out_ct(img_patch):

                                    img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                                    label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                                    sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + ".nii.gz")
                                    sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_mask.nii.gz")

                                    num_patches += 1

                            if z_end >= boundingAxes[5]:
                                break
                            else:
                                z += self.patch_size[2] // step_size  ## step size = 2

                        if y_end >= boundingAxes[3]:
                            break
                        else:
                            y += self.patch_size[1] // step_size

                    if x_end >= boundingAxes[1]:
                        break
                    else:
                        x += self.patch_size[0] // step_size

                print("complete augmenting this augmented image with number of patches = ", num_patches)

        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleCTData(test_img, new_spacing=self.new_spacing, is_label=False)

        test_origin_img = readItkData(self.test_img_origin_dir + test_name)
        test_origin_img = resampleCTData(test_origin_img, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        test_origin_img_np = np.transpose(sitk.GetArrayFromImage(test_origin_img), (2, 1, 0)).astype(np.float32)
        test_origin_img_np = histogram_matching(test_origin_img_np, bins=self.bins, cdf=self.cdf)

        patch_array, coords_array = generate_offline_test_batch_CT(origin_img_np=test_origin_img_np, img_np=test_img_np,
                                                                   patch_size=self.patch_size, step_size=step_size,
                                                                   test_name=test_img_base_name)  ## half of the patch size

        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):
                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(
                                    coords[2]) + ").nii.gz")

        return patch_array, coords_array


class Hippocampus_3D():

    def __init__(self, data_path, patch_size=(128, 128, 30), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'Hippocampus'
        self.path = data_path

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)
        #self.test_img_files = os.listdir(self.test_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.hardIntens = 1000
        self.channel = 1

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_hippocampus(self.train_img_dir, self.train_label_dir,
                                                                                 num_bins=num_bins, hardIntens=1000)
            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

    # get_train_batch
    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)

        test_img = test_img[:, :, :, :, np.newaxis]
        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            print(self.name, ": start augmenting image = ", img_base_name)

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            image_res = resampleMRIData(img, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)
            label_res = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)

            img_np = sitk.GetArrayFromImage(image_res)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label_res)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            ################################################################################################################

            center_patch_img, center_patch_label, boundingAxes = crop_center_from_label_MRI(img_np, label_np, patch_size=self.patch_size)
            img_shape = center_patch_img.shape

            print(" after resampling - ", img_base_name, "(", boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4], ")")

            ############### intensity normalization + histogram matching ##############################

            center_patch_img_arr = center_patch_img.flatten()  # shape = (x_dim * y_dim * z_dim, )

            arr_min, arr_max = np.min(center_patch_img_arr), np.max(center_patch_img_arr)
            arr_thres = [center_patch_img_arr[i] * self.hardIntens / (arr_max - arr_min) for i in range(center_patch_img_arr.shape[0])]   # just stretch to [0, self.hardIntens]

            s_values, bin_indx, s_counts = np.unique(arr_thres, return_inverse=True, return_counts=True)
            s_quantiles = np.cumsum(s_counts).astype(np.float64)
            s_quantiles /= s_quantiles[-1]

            interp_t_values = np.interp(s_quantiles, self.cdf, self.bins)  # t_values in origin
            arr_thres = interp_t_values[bin_indx]  # mask_np = np.interp(mask_np.flatten(), self.bins[:-1], self.cdf)

            center_patch_img = np.reshape(arr_thres, img_shape)

            ##########################################################################################

            img_patch_out = np.transpose(center_patch_img, (2, 1, 0)).astype(np.float32)
            img_patch_out = sitk.GetImageFromArray(img_patch_out)
            label_patch_out = np.transpose(center_patch_label, (2, 1, 0)).astype(np.float32)
            label_patch_out = sitk.GetImageFromArray(label_patch_out)

            sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug.nii.gz")
            sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_mask.nii.gz")

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.7)
                s_y = np.random.uniform(0.5, 1.7)
                s_z = np.random.uniform(0.5, 1.7)
                r_d = np.random.uniform(30, 70)

                img_aug, label_aug = augmentMRIData(img_patch_out, label_patch_out, scale=(s_x, s_y, s_z), rotate=r_d)

                ############################################################################################################

                if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 50:

                    sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + ".nii.gz")
                    sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_mask.nii.gz")

                    # img_np = sitk.GetArrayFromImage(img_aug)
                    # img_np = np.asarray(img_np, np.float32)
                    # img_np = np.transpose(img_np, (2, 1, 0))
                    #
                    # label_np = sitk.GetArrayFromImage(label_aug)
                    # label_np = np.asarray(label_np, np.int32)
                    # label_np = np.transpose(label_np, (2, 1, 0))
                    #
                    # ################################################################################################################
                    #
                    # center_patch_img, center_patch_label, boundingAxes = crop_center_from_label(img_np, label_np, patch_size=self.patch_size)
                    # img_shape = center_patch_img.shape
                    #
                    # center_patch_img_arr = center_patch_img.flatten()  # shape = (x_dim * y_dim * z_dim, )
                    #
                    # ##### thresholding to [0, self.hardIntes] #######
                    #
                    # arr_min, arr_max = np.min(center_patch_img_arr), np.max(center_patch_img_arr)
                    # arr_thres = [center_patch_img_arr[i] * self.hardIntens / (arr_max - arr_min) for i in range(center_patch_img_arr.shape[0])]
                    #
                    # s_values, bin_indx, s_counts = np.unique(arr_thres, return_inverse=True, return_counts=True)
                    # s_quantiles = np.cumsum(s_counts).astype(np.float64)
                    # s_quantiles /= s_quantiles[-1]
                    #
                    # interp_t_values = np.interp(s_quantiles, self.cdf, self.bins)  # t_values in origin
                    # arr_thres = interp_t_values[bin_indx]  # mask_np = np.interp(mask_np.flatten(), self.bins[:-1], self.cdf)
                    #
                    # center_patch_img = np.reshape(arr_thres, img_shape)
                    #
                    # print(" aug center_patch_img : ", np.min(center_patch_img), np.max(center_patch_img))

                    ###########################################################################

                    # img_patch_out = np.transpose(center_patch_img, (2, 1, 0))
                    # img_patch_out = sitk.GetImageFromArray(img_patch_out)
                    # label_patch_out = np.transpose(center_patch_label, (2, 1, 0))
                    # label_patch_out = sitk.GetImageFromArray(label_patch_out)

                else:

                    print("not writing this batch ")

        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleMRIData(test_img, new_spacing=self.new_spacing)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        patch_array, coords_array = generate_offline_test_batch_MRI(img_np=test_img_np,
                                                                      patch_size=self.patch_size, step_size=step_size,
                                                                      cdf=self.cdf, bins=self.bins,
                                                                      test_name=test_img_base_name)  ## half of the patch size

        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):
                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]
                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "_" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ".nii.gz")

        return patch_array, coords_array


class Lung_3D():


    def __init__(self, data_path, patch_size=(48, 48, 48), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'Lung'
        self.path = data_path

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(os.listdir(self.test_img_dir)))

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]
        self.test_img_files = os.listdir(self.test_img_dir)

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_lung(self.train_img_dir, self.train_label_dir)
            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")


    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            # img_np[img_np < -500] = -500
            # img_np[img_np > 400] = 400
            #
            # img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        train_img[train_img < -500] = -500
        train_img[train_img > 400] = 400

        train_img = (train_img - np.min(train_img)) / (np.max(train_img) - np.min(train_img))

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            # img_np[img_np < -500] = -500
            # img_np[img_np > 400] = 400
            #
            # img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        val_img[val_img < -500] = -500
        val_img[val_img > 400] = 400

        val_img = (val_img - np.min(val_img)) / (np.max(val_img) - np.min(val_img))

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            # img_np[img_np < -500] = -500
            # img_np[img_np > 400] = 400
            #
            # img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)

        test_img = test_img[:, :, :, :, np.newaxis]

        test_img[test_img < -500] = -500
        test_img[test_img > 400] = 400

        test_img = (test_img - np.min(test_img)) / (np.max(test_img) - np.min(test_img))

        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            print(self.name, ": start augmenting image = ", img_base_name)

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            img = resampleCTData(img, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)  # get bounding box: getBoundingBox(label)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            img_np = histogram_matching(img_np, bins=self.bins, cdf=self.cdf)

            ############################################################################################################

            center_patch_img, center_patch_label, maskBounds = crop_center_from_label_CT(img_np, label_np, patch_size=self.patch_size)
            print(img_base_name, "[", maskBounds[1] - maskBounds[0], maskBounds[3] - maskBounds[2], maskBounds[5] - maskBounds[4], ")")

            center_patch_img, center_patch_label = flip_rotate(center_patch_img, center_patch_label)

            img_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_img, (2, 1, 0)).astype(np.float32))
            label_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_label, (2, 1, 0)).astype(np.float32))

            sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_center.nii.gz")  # img_origin
            sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_center_mask.nii.gz")

            # boundingAxes = getBoundingAxes(label_np)
            # patches_img, patches_label = generate_patches_around_label(img_np, label_np, patch_size=self.patch_size, patch_num=5, boundingAxes=boundingAxes)

            # for p_indx in range(len(patches_img)):
            #
            #     patch_img_indx = patches_img[p_indx]
            #     patch_label_indx = patches_label[p_indx]
            #
            #     patch_img_indx, patch_label_indx = flip_rotate(patch_img_indx, patch_label_indx)
            #
            #     img_patch_out = sitk.GetImageFromArray(np.transpose(patch_img_indx, (2, 1, 0)))
            #     label_patch_out = sitk.GetImageFromArray(np.transpose(patch_label_indx, (2, 1, 0)))
            #
            #     sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(p_indx) + "_center.nii.gz")   # img_origin
            #     sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(p_indx) + "_center_mask.nii.gz")


            num_patches = 0
            for indx in range(num_aug):

                s_x = np.random.uniform(0.7, 1.5)
                s_y = np.random.uniform(0.7, 1.5)
                s_z = np.random.uniform(0.7, 1.5)

                r_d = np.random.uniform(10, 90)

                img_aug, label_aug = augmentCTData(img, label, scale=(s_x, s_y, s_z), rotate=r_d)

                img_aug_np = np.transpose(sitk.GetArrayFromImage(img_aug), (2, 1, 0)).astype(np.float32)
                label_aug_np = np.transpose(sitk.GetArrayFromImage(label_aug), (2, 1, 0)).astype(np.float32)

                img_aug_np = histogram_matching(img_aug_np, bins=self.bins, cdf=self.cdf)

                if np.count_nonzero(label_aug_np) > 50:

                    # center_patch_img, center_patch_label, maskBounds = crop_center_from_label_CT(img_aug_np, label_aug_np, patch_size=self.patch_size)

                    boundingAxes = getBoundingAxes(label_aug_np)
                    patches_img, patches_label = generate_patches_around_label(img_aug_np, label_aug_np, patch_size=self.patch_size,
                                                                          patch_num=5, boundingAxes=boundingAxes)


                    for p_indx in range(len(patches_img)):

                        patch_img_indx = patches_img[p_indx]
                        patch_label_indx = patches_label[p_indx]

                        patch_img_indx, patch_label_indx = flip_rotate(patch_img_indx, patch_label_indx)

                        img_patch_out = sitk.GetImageFromArray(np.transpose(patch_img_indx, (2, 1, 0)).astype(np.float32))
                        label_patch_out = sitk.GetImageFromArray(np.transpose(patch_label_indx, (2, 1, 0)).astype(np.float32))

                        sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(
                            p_indx) + "_" + str(indx) + "_center.nii.gz")  # img_origin
                        sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(
                            p_indx) + "_" + str(indx) + "_center_mask.nii.gz")

                    num_patches += len(patches_img)

                    # img_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_img, (2, 1, 0)))
                    # label_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_label, (2, 1, 0)))
                    #
                    # sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_aug_" + str(indx) + "_center.nii.gz")  # img_origin
                    # sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_aug_" + str(indx) + "_center_mask.nii.gz")

            print(" => writing ", num_patches)


            ################## crop patches from [X_start, Y_start, Z_start] ###########################################

            num_patches = 14
            patch_img_arr, patch_label_arr = generate_random_patches(img_np, label_np, patch_size=self.patch_size, num_patches=num_patches, bins=self.bins, cdf=self.cdf)

            for indx in range(patch_img_arr.shape[0]):

                img_patch = patch_img_arr[indx, :, :, :]
                label_patch = patch_label_arr[indx, :, :, :]

                img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(indx) + ".nii.gz")
                sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(indx) + "_mask.nii.gz")


        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleCTData(test_img, new_spacing=self.new_spacing, is_label=False)

        test_origin_img = readItkData(self.test_img_origin_dir + test_name)
        test_origin_img = resampleCTData(test_origin_img, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        test_origin_img_np = np.transpose(sitk.GetArrayFromImage(test_origin_img), (2, 1, 0)).astype(np.float32)
        test_origin_img_np = histogram_matching(test_origin_img_np, bins=self.bins, cdf=self.cdf)  # this step will generate full-size estimation!

        patch_array, coords_array = generate_offline_test_batch_CT(origin_img_np=test_origin_img_np, img_np=test_img_np,
                                                                   patch_size=self.patch_size, step_size=step_size,
                                                                   test_name=test_img_base_name)  ## half of the patch size


        patch_array[patch_array > 400] = 400
        patch_array[patch_array < -500] = -500

        patch_array = (patch_array - np.min(patch_array)) / (np.max(patch_array) - np.min(patch_array))


        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "_" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ".nii.gz")

        return patch_array, coords_array


class Colon_3D():


    def __init__(self, data_path, patch_size=(64, 64, 64), resampled_spacing=(2.0, 2.0, 2.0), val_ratio=0.1, num_bins=256):

        # train_img with train_label are used in supervised training batch
        self.train_img_dir = data_path + '/imagesTr/'
        self.train_label_dir = data_path + '/labelsTr/'
        self.test_img_dir = data_path + '/imagesTs_cropped/'
        self.test_img_origin_dir = data_path + '/imagesTs/'
        self.test_patch_dir = data_path + '/imagesTs_patches/'
        self.name = 'Colon'
        self.path = data_path

        if os.path.exists(data_path + '/imhist.txt') and os.path.exists(data_path + '/bins.txt') and os.path.exists(
                data_path + '/cdf.txt'):

            print(" loading histogram information from path ... ")

            self.imhist = np.loadtxt(data_path + '/imhist.txt')
            self.bins = np.loadtxt(data_path + '/bins.txt')
            self.cdf = np.loadtxt(data_path + '/cdf.txt')

        else:

            self.imhist, self.bins, self.cdf = compute_histogram_of_reference(self.train_img_dir, self.train_label_dir)

            np.savetxt(data_path + "/imhist.txt", self.imhist, fmt="%s")
            np.savetxt(data_path + "/bins.txt", self.bins, fmt="%s")
            np.savetxt(data_path + "/cdf.txt", self.cdf, fmt="%s")

        self.train_img_aug_dir = data_path + '/imagesTr_aug/'
        self.train_label_aug_dir = data_path + '/labelsTr_aug/'

        if not os.path.exists(self.train_img_aug_dir):
            os.makedirs(self.train_img_aug_dir)

        if not os.path.exists(self.train_label_aug_dir):
            os.makedirs(self.train_label_aug_dir)

        self.aug_img_files = os.listdir(self.train_img_dir)
        self.train_img_files = os.listdir(self.train_img_aug_dir)    # os.listdir(self.train_img_dir)
        self.test_img_files = os.listdir(self.test_img_dir)

        self.val_ratio = val_ratio
        self.num_val = int(np.floor(self.val_ratio * int(len(self.train_img_files))))
        self.num_train = int(len(self.train_img_files)) - self.num_val
        self.num_test = int(len(self.test_img_files))

        self.val_img_files = self.train_img_files[-self.num_val:]
        self.train_img_files = self.train_img_files[:-self.num_val]

        if self.val_ratio == 0:
            self.train_img_files = self.val_img_files
            self.num_val = self.num_train

        self.patch_size = patch_size
        self.new_spacing = resampled_spacing
        self.channel = 1


    def get_train_batch(self, train_img_dir, train_label_dir, batch_img_files):

        train_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        train_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)
#
# ##################################################################################################################################
#
            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

#######################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            train_img[indx] = patch_img_stack
            train_label[indx] = patch_label_stack
            indx += 1

        # ###############################################################################

        train_img = train_img[:, :, :, :, np.newaxis]   # [batch, h, w, d, channel]

        assert indx == len(train_img)
        return train_img, np.round(train_label)


    def get_val_batch(self, val_img_dir, val_label_dir, batch_img_files):

        val_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)
        val_label = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(val_img_dir + batch_img_files[file_index])
            label = readItkLabelData(val_label_dir + label_base_name + ext)

            ##################################################################################################################################

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))
            #
            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            patch_label_stack = np.array([label_np])
            val_img[indx] = patch_img_stack
            val_label[indx] = patch_label_stack
            indx += 1

        val_img = val_img[:, :, :, :, np.newaxis]   # [batch_size, h, w, d, channel]

        assert indx == len(val_img)
        return val_img, np.round(val_label)


    def get_test_batch(self, batch_img_files):

        test_img = np.zeros((len(batch_img_files), self.patch_size[0], self.patch_size[1], self.patch_size[2]), dtype=np.float32)

        indx = 0
        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            img = readItkData(self.test_patch_dir + batch_img_files[file_index])

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            ####################################################################################################################################

            patch_img_stack = np.array([img_np])
            test_img[indx] = patch_img_stack
            indx += 1

        assert indx == len(test_img)

        test_img = test_img[:, :, :, :, np.newaxis]
        return test_img


    def augment_data(self, train_img_dir, train_label_dir, batch_img_files, num_aug=2):

        for file_index in range(len(batch_img_files)):

            img_base_name = batch_img_files[file_index].split('.')[0]
            label_base_name = img_base_name + "_mask"
            ext = '.nii.gz'

            img = readItkData(train_img_dir + batch_img_files[file_index])
            label = readItkLabelData(train_label_dir + label_base_name + ext)

            img = resampleCTData(img, new_spacing=self.new_spacing, is_label=False)
            label = resampleMRIData(label, new_spacing=self.new_spacing, is_label=True)

            img_np = sitk.GetArrayFromImage(img)
            img_np = np.asarray(img_np, np.float32)
            img_np = np.transpose(img_np, (2, 1, 0))

            label_np = sitk.GetArrayFromImage(label)
            label_np = np.asarray(label_np, np.int32)
            label_np = np.transpose(label_np, (2, 1, 0))

            ############################################################################################################

            center_patch_img, center_patch_label, maskBounds = crop_center_from_label_CT(img_np, label_np, patch_size=self.patch_size, cdf=self.cdf, bins=self.bins)

            print(img_base_name, "[", maskBounds[1] - maskBounds[0], maskBounds[3] - maskBounds[2], maskBounds[5] - maskBounds[4], ")")

            img_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_img, (2, 1, 0)).astype(np.float32))
            label_patch_out = sitk.GetImageFromArray(np.transpose(center_patch_label, (2, 1, 0)).astype(np.float32))

            sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_center.nii.gz")  # img_origin
            sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_center_mask.nii.gz")

            for indx in range(num_aug):

                s_x = np.random.uniform(0.5, 1.7)
                s_y = np.random.uniform(0.5, 1.7)
                s_z = np.random.uniform(0.5, 1.7)
                r_d = np.random.uniform(30, 70)

                img_aug, label_aug = augmentCTData(img_patch_out, label_patch_out, scale=(s_x, s_y, s_z), rotate=r_d)

                if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 10:
                    sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_c_" + str(indx) + ".nii.gz")
                    sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_c_" + str(indx) + "_mask.nii.gz")


            ################## crop patches from [X_start, Y_start, Z_start] ###########################################

            num_patches = 7
            patch_img_arr, patch_label_arr = generate_random_patches(img_np, label_np, patch_size=self.patch_size, num_patches=num_patches, bins=self.bins, cdf=self.cdf)

            for indx in range(patch_img_arr.shape[0]):

                img_patch = patch_img_arr[indx, :, :, :]
                label_patch = patch_label_arr[indx, :, :, :]

                img_patch_out = sitk.GetImageFromArray(np.transpose(img_patch, (2, 1, 0)).astype(np.float32))
                label_patch_out = sitk.GetImageFromArray(np.transpose(label_patch, (2, 1, 0)).astype(np.float32))

                sitk.WriteImage(img_patch_out, self.train_img_aug_dir + img_base_name + "_" + str(indx) + ".nii.gz")
                sitk.WriteImage(label_patch_out, self.train_label_aug_dir + img_base_name + "_" + str(indx) + "_mask.nii.gz")

                if np.count_nonzero(label_patch) > 10:

                    for indx in range(num_aug):

                        s_x = np.random.uniform(0.5, 1.7)
                        s_y = np.random.uniform(0.5, 1.7)
                        s_z = np.random.uniform(0.5, 1.7)
                        r_d = np.random.uniform(30, 70)

                        img_aug, label_aug = augmentCTData(img_patch_out, label_patch_out, scale=(s_x, s_y, s_z), rotate=r_d)

                        if np.count_nonzero(sitk.GetArrayFromImage(label_aug)) > 10:
                            sitk.WriteImage(img_aug, self.train_img_aug_dir + img_base_name + "_aug_" + str(
                                num_patches) + "_" + str(indx) + ".nii.gz")
                            sitk.WriteImage(label_aug, self.train_label_aug_dir + img_base_name + "_aug_" + str(
                                num_patches) + "_" + str(indx) + "_mask.nii.gz")

        print(self.name, " : augmentation is done .. ")


    def generate_offline_test_batch(self, test_img_dir, test_name, step_size=2, write_out=False):

        test_img_base_name = test_name.split('.')[0]
        test_img = readItkData(test_img_dir + test_name)
        test_img = resampleCTData(test_img, new_spacing=self.new_spacing, is_label=False)

        test_origin_img = readItkData(self.test_img_origin_dir + test_name)
        test_origin_img = resampleCTData(test_origin_img, new_spacing=self.new_spacing, is_label=False)

        test_img_np = np.transpose(sitk.GetArrayFromImage(test_img), (2, 1, 0)).astype(np.float32)

        test_origin_img_np = np.transpose(sitk.GetArrayFromImage(test_origin_img), (2, 1, 0)).astype(np.float32)
        test_origin_img_np = histogram_matching(test_origin_img_np, bins=self.bins, cdf=self.cdf)

        patch_array, coords_array = generate_offline_test_batch_CT(origin_img_np=test_origin_img_np, img_np=test_img_np,
                                                                   patch_size=self.patch_size, step_size=step_size,
                                                                      cdf=self.cdf, bins=self.bins, test_name=test_img_base_name)  ## half of the patch size


        ###################################################################################################################

        if write_out:

            save_dir = self.path + '/imagesTs_patches/'
            print(" start writing out the test patches to ", save_dir, test_name)

            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for indx in range(patch_array.shape[0]):

                patch = patch_array[indx, :, :, :]
                coords = coords_array[indx]

                sitk.WriteImage(sitk.GetImageFromArray(np.transpose(patch, (2, 1, 0)).astype(np.float32)),
                                save_dir + test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ").nii.gz")

        return patch_array, coords_array

