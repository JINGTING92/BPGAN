import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim as slim
from sklearn.metrics import jaccard_similarity_score, f1_score

import os, sys, math
import SimpleITK as sitk
from seg_utils import *


project_path = '/home/jtma/PycharmProjects/'


############################ CT Utilites ###########################################


def augmentCTData(image, label, scale=None, rotate=None):

    affine = sitk.AffineTransform(3)

    # translation
    # t_x = 10 #np.random.randint(-20, 20)
    # t_y = 0 # np.random.randint(-20, 20)
    # t_z = 0 # np.random.randint(-20, 20)
    # affine.SetTranslation((t_x, t_y, t_z))

    affine.Scale((scale[0], scale[1], scale[2]))

    if rotate:
        radias = np.pi * rotate / 180
        affine.Rotate(axis1=0, axis2=1, angle=np.float(radias))     #affineR.Rotate(axis1=0, axis2=1, angle=radias)

    ##############################################################

    reference_image = image
    reference_label = label
    interpolator = sitk.sitkNearestNeighbor

    aug_img = sitk.Resample(image, reference_image, affine, interpolator, -1024)
    aug_label = sitk.Resample(label, reference_label, affine, interpolator, 0)


    return aug_img, aug_label


def compute_histogram_of_lung(train_img_dir, train_label_dir):

    print(" compute histogram of the lung ... ")

    total_img_list = []

    train_files = os.listdir(train_img_dir)

    for file_index in range(len(train_files)):

        img_base_name = train_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.nii.gz'

        img = readItkData(train_img_dir + train_files[file_index])
        label = readItkLabelData(train_label_dir + label_base_name + ext)

        img_np = sitk.GetArrayFromImage(img)
        img_np = np.transpose(img_np, (2, 1, 0))


        ##########################

        # dims = img_np.shape
        # arr = np.reshape(img_np, (dims[0] * dims[1] * dims[2], -1))
        #
        # total_img_list.extend(arr.tolist())
        #
        # break

        label_np = sitk.GetArrayFromImage(label)
        label_np = np.transpose(label_np, (2, 1, 0))

        ### extend from the boundingBox

        # boundingAxes = getBoundingAxes(label_np)
        #
        # print(img_base_name, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
        #
        # img_patch, _, _ = crop_center_from_label_CT(img_np, label_np, patch_size=(140, 140, 140))
        #
        # #img_patch = img_np[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3], boundingAxes[4]:boundingAxes[5]]
        #
        # dims = img_patch.shape
        # arr = np.reshape(img_patch, (dims[0] * dims[1] * dims[2], -1))
        #
        # total_img_list.extend(arr.tolist())  # .tolist()

        mask_arr = np.multiply(img_np, label_np)
        mask_arr = mask_arr.flatten()
        print(img_base_name, np.min(mask_arr), np.max(mask_arr), np.mean(mask_arr), np.std(mask_arr))

        total_img_list.extend(mask_arr)

    ########### histogram computation ########################

    print(" start computing the histogram ... ")

    ### histogram matching ###
    t_values, imhist = np.unique(total_img_list, return_counts=True)
    cdf = np.cumsum(imhist).astype(np.float64)
    cdf /= cdf[-1]

    # imhist, bins = np.histogram(total_img_list, bins=num_bins, density=True)
    # cdf = imhist.cumsum()  # cumulative distribution function
    # cdf = imhist.max() * cdf / cdf[-1]  # normalize, cdf[-1] = cdf.max()    # without * 255 !!!!

    print(" histogram computation is done ! ")

    return imhist, t_values, cdf  # imhist, t_values, cdf   # imhist, bins, cdf


def compute_histogram_of_pancreas(train_img_dir, train_label_dir, reference_img_dir):

    print(" compute histogram of the pancreas ... ")

    total_img_list = []

    train_files = os.listdir(train_img_dir)
    reference_img = readItkData(reference_img_dir)  # it is a dir
    reference_img_np = sitk.GetArrayFromImage(reference_img)
    reference_img_np = np.transpose(reference_img_np, (2, 1, 0))

    dims = reference_img_np.shape
    arr = np.reshape(reference_img_np, (dims[0] * dims[1] * dims[2], -1))
    # total_img_list.extend(arr.tolist())  # .tolist()

    t_values_refer, imhist_refer = np.unique(arr, return_counts=True)
    cdf_refer = np.cumsum(imhist_refer).astype(np.float64)
    cdf_refer /= cdf_refer[-1]

    # for file_index in range(len(train_files)):
    #
    #     img_base_name = train_files[file_index].split('.')[0]
    #     label_base_name = img_base_name + "_mask"
    #     ext = '.nii.gz'
    #
    #     img = readItkData(train_img_dir + train_files[file_index])
    #     img_np = sitk.GetArrayFromImage(img)
    #     img_np = np.transpose(img_np, (2, 1, 0))
    #
    #     label = readItkLabelData(train_label_dir + label_base_name + ext)
    #     label_np = sitk.GetArrayFromImage(label)
    #     label_np = np.transpose(label_np, (2, 1, 0))
    #
    #     mask_before = np.multiply(img_np, label_np)
    #
    #     print(' => ', img_base_name, np.min(mask_before.flatten()), np.max(mask_before.flatten()), np.mean(mask_before.flatten()), np.std(mask_before.flatten()))
    #
    #     img_np = histogram_matching(img_np, cdf=cdf_refer, bins=t_values_refer)
    #
    #     img_histo = sitk.GetImageFromArray(np.transpose(img_np, (2, 1, 0)))
    #     sitk.WriteImage(img_histo, '/home/jtma/PycharmProjects/dataset/test/' + img_base_name + "_his.nii.gz")
    #
    #     mask_arr = np.multiply(img_np, label_np)
    #     mask_arr = mask_arr.flatten()
    #     print(' => ', img_base_name, np.min(mask_arr), np.max(mask_arr), np.mean(mask_arr), np.std(mask_arr))


        ##########################

        # dims = img_np.shape
        # arr = np.reshape(img_np, (dims[0] * dims[1] * dims[2], -1))
        #
        # total_img_list.extend(arr.tolist())
        #
        # break

        ### extend from the boundingBox

        # boundingAxes = getBoundingAxes(label_np)
        #
        # print(img_base_name, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
        #
        # img_patch, _, _ = crop_center_from_label_CT(img_np, label_np, patch_size=(140, 140, 140))
        #
        # #img_patch = img_np[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3], boundingAxes[4]:boundingAxes[5]]
        #
        # dims = img_patch.shape
        # arr = np.reshape(img_patch, (dims[0] * dims[1] * dims[2], -1))
        #
        # total_img_list.extend(arr.tolist())  # .tolist()


        #
        # total_img_list.extend(mask_arr)

    ########### histogram computation ########################

    print(" start computing the histogram ... ")

    ### histogram matching ###
    # t_values, imhist = np.unique(total_img_list, return_counts=True)
    # cdf = np.cumsum(imhist).astype(np.float64)
    # cdf /= cdf[-1]

    # imhist, bins = np.histogram(total_img_list, bins=num_bins, density=True)
    # cdf = imhist.cumsum()  # cumulative distribution function
    # cdf = imhist.max() * cdf / cdf[-1]  # normalize, cdf[-1] = cdf.max()    # without * 255 !!!!

    print(" histogram computation is done ! ")

    return imhist_refer, t_values_refer, cdf_refer  # imhist, t_values, cdf   # imhist, bins, cdf


def compute_histogram_of_hepaticVessel(train_img_dir, train_label_dir, reference_img_dir):

    print(" compute histogram of the pancreas ... ")

    total_img_list = []

    train_files = os.listdir(train_img_dir)
    reference_img = readItkData(reference_img_dir)  # it is a dir
    reference_img_np = sitk.GetArrayFromImage(reference_img)
    reference_img_np = np.transpose(reference_img_np, (2, 1, 0))

    dims = reference_img_np.shape
    arr = np.reshape(reference_img_np, (dims[0] * dims[1] * dims[2], -1))
    # total_img_list.extend(arr.tolist())  # .tolist()

    print(" reference img = ", reference_img_dir, np.min(arr), np.max(arr))

    t_values_refer, imhist_refer = np.unique(arr, return_counts=True)
    cdf_refer = np.cumsum(imhist_refer).astype(np.float64)
    cdf_refer /= cdf_refer[-1]

    for file_index in range(len(train_files)):

        img_base_name = train_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.nii.gz'

        img = readItkData(train_img_dir + train_files[file_index])
        img_np = sitk.GetArrayFromImage(img)
        img_np = np.transpose(img_np, (2, 1, 0))

        label = readItkLabelData(train_label_dir + label_base_name + ext)
        label_np = sitk.GetArrayFromImage(label)
        label_np = np.transpose(label_np, (2, 1, 0))

        mask_before = np.multiply(img_np, label_np)

        print(' => ', img_base_name, np.min(mask_before.flatten()), np.max(mask_before.flatten()))

        img_np = histogram_matching(img_np, cdf=cdf_refer, bins=t_values_refer)

        img_histo = sitk.GetImageFromArray(np.transpose(img_np, (2, 1, 0)))
        sitk.WriteImage(img_histo, '/home/jtma/PycharmProjects/dataset/test/' + img_base_name + "_his.nii.gz")

        mask_arr = np.multiply(img_np, label_np)
        mask_arr = mask_arr.flatten()
        print(' => ', img_base_name, np.min(mask_arr), np.max(mask_arr))


        ##########################

        # dims = img_np.shape
        # arr = np.reshape(img_np, (dims[0] * dims[1] * dims[2], -1))
        #
        # total_img_list.extend(arr.tolist())
        #
        # break

        ### extend from the boundingBox

        # boundingAxes = getBoundingAxes(label_np)
        #
        # print(img_base_name, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
        #
        # img_patch, _, _ = crop_center_from_label_CT(img_np, label_np, patch_size=(140, 140, 140))
        #
        # #img_patch = img_np[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3], boundingAxes[4]:boundingAxes[5]]
        #
        # dims = img_patch.shape
        # arr = np.reshape(img_patch, (dims[0] * dims[1] * dims[2], -1))
        #
        # total_img_list.extend(arr.tolist())  # .tolist()


        #
        # total_img_list.extend(mask_arr)

    ########### histogram computation ########################

    print(" start computing the histogram ... ")

    ### histogram matching ###
    # t_values, imhist = np.unique(total_img_list, return_counts=True)
    # cdf = np.cumsum(imhist).astype(np.float64)
    # cdf /= cdf[-1]

    # imhist, bins = np.histogram(total_img_list, bins=num_bins, density=True)
    # cdf = imhist.cumsum()  # cumulative distribution function
    # cdf = imhist.max() * cdf / cdf[-1]  # normalize, cdf[-1] = cdf.max()    # without * 255 !!!!

    print(" histogram computation is done ! ")

    return imhist_refer, t_values_refer, cdf_refer  # imhist, t_values, cdf   # imhist, bins, cdf


def compute_histogram_of_reference(train_img_dir, train_label_dir=None, reference_img_dir=None):

    print(" compute histogram of the reference ... ")

    train_files = os.listdir(train_img_dir)
    reference_img = readItkData(reference_img_dir)  # it is a dir
    reference_img_np = sitk.GetArrayFromImage(reference_img)
    reference_img_np = np.transpose(reference_img_np, (2, 1, 0))

    dims = reference_img_np.shape
    arr = np.reshape(reference_img_np, (dims[0] * dims[1] * dims[2], -1))
    # total_img_list.extend(arr.tolist())  # .tolist()

    t_values_refer, imhist_refer = np.unique(arr, return_counts=True)
    cdf_refer = np.cumsum(imhist_refer).astype(np.float64)
    cdf_refer /= cdf_refer[-1]


    # total_img_list = []
    #
    # train_files = os.listdir(train_img_dir)
    #
    # for file_index in range(len(train_files)):
    #
    #     # img_base_name = train_files[file_index].split('.')[0]
    #
    #     img = readItkData(train_img_dir + train_files[file_index])
    #
    #     img_np = sitk.GetArrayFromImage(img)
    #     img_np = np.transpose(img_np, (2, 1, 0))
    #
    #     dims = img_np.shape
    #     arr = np.reshape(img_np, (dims[0] * dims[1] * dims[2], -1))
    #
    #     total_img_list.extend(arr.tolist())
    #
    #     break
    #
    # ########### histogram computation ########################
    #
    # print(" start computing the histogram ... ")
    #
    # ### histogram matching ###
    # t_values, imhist = np.unique(total_img_list, return_counts=True)
    # cdf = np.cumsum(imhist).astype(np.float64)
    # cdf /= cdf[-1]

    print(" histogram computation is done ! ")

    return imhist_refer, t_values_refer, cdf_refer  # imhist, t_values, cdf   # imhist, bins, cdf


def resampleCTData(img, new_spacing=(2.0, 2.0, 2.0), is_label=False):

    original_size = np.array(img.GetSize(), dtype=np.int)
    original_spacing = img.GetSpacing()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]


    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(out_size)
    resampler.SetDefaultPixelValue(-1025)   # ct = -1024, mri = 0

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)  # sitk.sitkLinear

    img_out = resampler.Execute(img)

    return img_out  # resampler.Execute(img)


def getPaddingBox_CT(array, out_size=(128, 64, 64)):

    """
    pad the array with zeros if smaller than the out size
    :param array: input array - 3d
    :param out_size: the target size
    :return: the padded array with the output size
    """

    dims = array.shape

    # out_arr = np.array(out_size, dtype=np.float32)
    # out_arr.fill(-1024)

    min_dim = [min(dims[0], out_size[0]), min(dims[1], out_size[1]), min(dims[2], out_size[2])]

    out_arr = np.full(out_size, -1024).astype(np.float64)
    out_arr[:min_dim[0], :min_dim[1], :min_dim[2]] = array[:min_dim[0], :min_dim[1], :min_dim[2]]

    return out_arr


def findMaskBoundingBox_CT(img_arr):

    mask = np.where(img_arr > -1000)

    boundingAxes = [np.min(mask[0]), np.max(mask[0]),
                    np.min(mask[1]), np.max(mask[1]),
                    np.min(mask[2]), np.max(mask[2])]

    x_dim = boundingAxes[1] - boundingAxes[0] + 1
    y_dim = boundingAxes[3] - boundingAxes[2] + 1
    z_dim = boundingAxes[5] - boundingAxes[4] + 1

    mask = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    mask[:x_dim - 1, :y_dim - 1, :z_dim - 1] = img_arr[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3],
                                               boundingAxes[4]:boundingAxes[5]]

    # xmin = img_dim[0] - 1 - np.max(mask[0])
    # xmax = img_dim[0] - 1 - np.min(mask[0])
    # ymin = img_dim[1] - 1 - np.max(mask[1])
    # ymax = img_dim[1] - 1 - np.min(mask[1])

    return boundingAxes, mask


def cropCenter_CT(array, out_size=(64, 64, 64,)):

    """
    crop the array from center
    :param array: input array which are larger than the out_size
    :param out_size: aim size
    :return: cropped area
    """

    center = [array.shape[0] // 2, array.shape[1] // 2, array.shape[2] // 2]

    X_start = max(0, center[0] - out_size[0] // 2)
    Y_start = max(0, center[1] - out_size[1] // 2)
    Z_start = max(0, center[2] - out_size[2] // 2)

    X_end = min(array.shape[0] - 1, X_start + out_size[0])
    Y_end = min(array.shape[1] - 1, Y_start + out_size[1])
    Z_end = min(array.shape[2] - 1, Z_start + out_size[2])

    cropped_arr = array[X_start:X_end, Y_start:Y_end, Z_start:Z_end]

    if cropped_arr.shape[0] < out_size[0] or cropped_arr.shape[1] < out_size[1] or cropped_arr.shape[2] < out_size[2]:
        cropped_arr = getPaddingBox_CT(cropped_arr, out_size=out_size)

    return cropped_arr


def generate_offline_test_batch_CT(origin_img_np, img_np, patch_size, step_size=2, cdf=None, bins=None, test_name=""):

    test_patch_list = []
    test_patch_coords = []

    img_dim = img_np.shape

    boundingAxes, mask_np = findMaskBoundingBox_CT(img_np)
    sliding_size = (patch_size[0] // step_size, patch_size[1] // step_size, patch_size[2] // step_size)

    # print("boundingAxes = ", boundingAxes, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
    # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0))), "/home/jtma/PycharmProjects/dataset/test/" + test_name + "_cropped.nii.gz")

    X_start = boundingAxes[0]
    while X_start < img_dim[0]:

        X_end = min(X_start + patch_size[0], img_dim[0] - 1)
        Y_start = boundingAxes[2]
        while Y_start < img_dim[1]:

            Y_end = min(Y_start + patch_size[1], img_dim[1] - 1)
            Z_start = boundingAxes[4]
            while Z_start < img_dim[2]:

                Z_end = min(Z_start + patch_size[2], img_dim[2] - 1)
                patch_img = origin_img_np[X_start:X_end, Y_start:Y_end, Z_start:Z_end]  # or img_cp_np

                if patch_img.shape[0] < patch_size[0] or patch_img.shape[1] < patch_size[1] or patch_img.shape[2] < patch_size[2]:
                    patch_img = getPaddingBox_CT(patch_img, out_size=patch_size)

                ###########################################################

                test_patch_list.append(patch_img)
                test_patch_coords.append([X_start, Y_start, Z_start])

                if Z_end < boundingAxes[5]:
                    Z_start += sliding_size[2]
                else:
                    break

            if Y_end < boundingAxes[3]:
                Y_start += sliding_size[1]
            else:
                break

        if X_end < boundingAxes[1]:
            X_start += sliding_size[0]
        else:
            break

    ############################## reshape the list ##########################3######

    test_patch_list = np.asarray(test_patch_list)  # shape = (num_patches, patch_size)
    test_patch_coords = np.asarray(test_patch_coords)

    return test_patch_list, test_patch_coords


def generate_offline_test_batch_spleen(origin_img_np, img_np, patch_size, step_size=2, test_name=""):

    test_patch_list = []
    test_patch_coords = []

    img_dim = img_np.shape

    boundingAxes, mask_np = findMaskBoundingBox_CT(img_np)
    sliding_size = (patch_size[0] // step_size, patch_size[1] // step_size, patch_size[2] // step_size)

#    print("boundingAxes = ", boundingAxes, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
#    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0))), "/home/jtma/PycharmProjects/dataset/test/" + test_name + "_cropped.nii.gz")

    X_start = boundingAxes[0]
    while X_start < img_dim[0]:

        X_end = min(X_start + patch_size[0], img_dim[0] - 1)
        Y_start = boundingAxes[2]
        while Y_start < img_dim[1]:

            Y_end = min(Y_start + patch_size[1], img_dim[1] - 1)
            Z_start = boundingAxes[4]
            while Z_start < img_dim[2]:

                Z_end = min(Z_start + patch_size[2], img_dim[2] - 1)
                patch_img = origin_img_np[X_start:X_end, Y_start:Y_end, Z_start:Z_end]  # or img_cp_np

                if patch_img.shape[0] < patch_size[0] or patch_img.shape[1] < patch_size[1] or patch_img.shape[2] < patch_size[2]:
                    patch_img = getPaddingBox_CT(patch_img, out_size=patch_size)

                ###########################################################

                test_patch_list.append(patch_img)
                test_patch_coords.append([X_start, Y_start, Z_start])

                if Z_end < boundingAxes[5]:
                    Z_start += sliding_size[2]
                else:
                    break

            if Y_end < boundingAxes[3]:
                Y_start += sliding_size[1]
            else:
                break

        if X_end < boundingAxes[1]:
            X_start += sliding_size[0]
        else:
            break

    ############################## reshape the list ##########################3######

    test_patch_list = np.asarray(test_patch_list)  # shape = (num_patches, patch_size)
    test_patch_coords = np.asarray(test_patch_coords)

    return test_patch_list, test_patch_coords


def crop_center_from_label_CT(img, label, patch_size=(64, 64, 32), cdf=None, bins=None):

    boundingAxes = getBoundingAxes(label)

    X_center = int(boundingAxes[0] + boundingAxes[1]) // 2
    Y_center = int(boundingAxes[2] + boundingAxes[3]) // 2
    Z_center = int(boundingAxes[4] + boundingAxes[5]) // 2

    X_start = max(0, X_center - patch_size[0] // 2)
    Y_start = max(0, Y_center - patch_size[1] // 2)
    Z_start = max(0, Z_center - patch_size[2] // 2)

    X_end = X_start + patch_size[0] if X_start + patch_size[0] < img.shape[0] else img.shape[0] - 1
    Y_end = Y_start + patch_size[1] if Y_start + patch_size[1] < img.shape[1] else img.shape[1] - 1
    Z_end = Z_start + patch_size[2] if Z_start + patch_size[2] < img.shape[2] else img.shape[2] - 1

    cropped_img = img[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end)]
    cropped_label = label[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end)]

    # cropped_img = histogram_matching(cropped_img, cdf=cdf, bins=bins)

    if cropped_img.shape[0] < patch_size[0] or cropped_img.shape[1] < patch_size[1] or cropped_img.shape[2] < patch_size[2]:

        cropped_img = getPaddingBox_CT(cropped_img, out_size=patch_size)
        cropped_label = getPaddingBox_MRI(cropped_label, out_size=patch_size)   # background set to 0

    return cropped_img, cropped_label, boundingAxes


def generate_random_patches(img_np, label_np, patch_size=(64, 32, 32), num_patches=5, cdf=None, bins=None):

    patch_img_arr, patch_label_arr = [], []

    boundingAxes, _, = findMaskBoundingBox_CT(img_np)

    for indx in range(num_patches):

        x_start = int(np.random.uniform(boundingAxes[0], boundingAxes[1] - patch_size[0]))
        y_start = int(np.random.uniform(boundingAxes[2], boundingAxes[3] - patch_size[1]))
        z_start = int(np.random.uniform(boundingAxes[4], boundingAxes[5] - patch_size[2]))

        x_end = min(x_start + patch_size[0], img_np.shape[0])
        y_end = min(y_start + patch_size[1], img_np.shape[1])
        z_end = min(z_start + patch_size[2], img_np.shape[2])

        img_patch = img_np[x_start:x_end, y_start:y_end, z_start:z_end]
        label_patch = label_np[x_start:x_end, y_start:y_end, z_start:z_end]

        # img_patch = histogram_matching(img_patch, bins=bins, cdf=cdf)

        if np.max(label_patch) > 0:

            if img_patch.shape[0] < patch_size[0] or img_patch.shape[1] < patch_size[1] or img_patch.shape[2] < patch_size[2]:

                img_patch = getPaddingBox_CT(img_patch, out_size=patch_size)
                label_patch = getPaddingBox_MRI(label_patch, out_size=patch_size)

        patch_img_arr.append(img_patch)
        patch_label_arr.append(label_patch)

    patch_img_arr = np.asarray(patch_img_arr)
    patch_label_arr = np.asarray(patch_label_arr)

    return patch_img_arr, patch_label_arr


def generate_patches_around_label(img_arr, label_arr, boundingAxes, patch_size=(32, 32, 32), patch_num=2):

    patch_img_list, patch_label_list = [], []

    start_x = max(0, min(boundingAxes[1] - patch_size[0], boundingAxes[0]))
    start_y = max(0, min(boundingAxes[3] - patch_size[1], boundingAxes[2]))
    start_z = max(0, min(boundingAxes[5] - patch_size[2], boundingAxes[4]))

    end_x = min(img_arr.shape[0], boundingAxes[1]) - 1
    end_y = min(img_arr.shape[1], boundingAxes[3]) - 1
    end_z = min(img_arr.shape[2], boundingAxes[5]) - 1


    for indx in range(patch_num):

        x = int(np.random.uniform(start_x, end_x - patch_size[0] // 2))
        y = int(np.random.uniform(start_y, end_y - patch_size[1] // 2))
        z = int(np.random.uniform(start_z, end_z - patch_size[2] // 2))

        patch_img = img_arr[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]
        patch_label = label_arr[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]

        if patch_img.shape[0] < patch_size[0] or patch_img.shape[1] < patch_size[1] or patch_img.shape[2] < patch_size[2]:

            patch_img = getPaddingBox_CT(patch_img, out_size=patch_size)
            patch_label = getPaddingBox_MRI(patch_label, out_size=patch_size)

        if np.max(patch_label) > 0 and np.count_nonzero(patch_label) >= 50 and np.max(patch_img) > 10:

            patch_img_list.append(patch_img)
            patch_label_list.append(patch_label)

    return patch_img_list, patch_label_list


def write_to_write_out_ct(img_np):

    total_pixel = img_np.flatten().shape[0]

    roi_pixel = np.count_nonzero(img_np.flatten() > -500)

    if roi_pixel / total_pixel < 0.67:
        return False
    else:
        return True


def merge_subfolder_to_segment(resampled_img_np, sub_folder_dir):

    """
    merge the segmented patches to the resampled image map
    :param resampled_img_np: resampled image array
    :param sub_folder_dir: folder contains the segmented patches
    :return: merged segmented patches
    """

    mask_np = np.zeros(resampled_img_np.shape, np.float32)

    for patch in os.listdir(sub_folder_dir):

        patch_name = patch.split('.')[0]  # remove the extension - .nii.gz

        patch_base_name, coords = parse_seg_name(patch_name)

        img_patch = readItkData(sub_folder_dir + patch)
        img_patch_np = np.transpose(np.asarray(sitk.GetArrayFromImage(img_patch), np.float32), (2, 1, 0))

        x_end = min(mask_np.shape[0] - 1, coords[0] + img_patch_np.shape[0])
        y_end = min(mask_np.shape[1] - 1, coords[1] + img_patch_np.shape[1])
        z_end = min(mask_np.shape[2] - 1, coords[2] + img_patch_np.shape[2])

        mask_np[coords[0]:x_end, coords[1]:y_end, coords[2]:z_end] = np.add(mask_np[coords[0]:x_end, coords[1]:y_end, coords[2]:z_end],
                                                                            img_patch_np[:x_end - coords[0], :y_end - coords[1], :z_end - coords[2]])

    return mask_np


def parse_seg_name(seg_name):

    """
    parse the segmented patch and merge into the resampled img_np
    :param img_np: resampled original image np
    :param seg_np: seg patch np
    :param seg_name: test_img_base_name + ( + x - y - z + ) + '_seg / _fuse' (no extension)
    :return: the resampled array
    """

    # print("parse_seg_name = ", seg_name)

    seg_base_name = seg_name.split("(")[0]
    coords_substring = seg_name[len(seg_base_name) + 1:-1]  # '(    # strip(seg_base_name + '(')   # remains 'x-y-z)_seg' or 'x-y-z)_fuse'

    coord_x = coords_substring.split('-')[0]
    str_y_z = coords_substring[len(coord_x) + 1:-1]    # .strip(tmp_coord_x)    # remains 'y-z)_seg' or 'y-z)_fuse'

    # print("coord_x = ", coord_x, str_y_z)

    coord_y = str_y_z.split('-')[0]
    str_z = str_y_z[len(coord_y) + 1:-1]   #   .replace(coord_y + '-', '')  # .strip(tmp_coord_y)    # remains 'z)_seg' or 'z)_fuse'
    coord_z = str_z.split(')')[0]

    # print("coord_y = ", coord_y, str_z, coord_z)

    coords = [int(coord_x), int(coord_y), int(coord_z)]

    return seg_base_name, coords


############################ MRI Utilities ##########################################


def write_multi_channel_image(img_arr, origin_spacing=None, spacing=None):

    t_slices = []
    for t in range(img_arr.shape[3]):

        img_t = img_arr[:, :, :, t]
        img_t = np.transpose(img_t, (2, 1, 0))
        img_t = sitk.GetImageFromArray(img_t)

        if spacing != None:

            img_t.SetSpacing(origin_spacing)  # this step is very important
            img_t = resampleMRIData(img_t, new_spacing=spacing, is_label=False)

        t_slices.append(img_t)

    img_out = sitk.JoinSeries(t_slices)

    return img_out


def augmentMRIData(image, label, scale, rotate):

    affine = sitk.AffineTransform(3)

    # translation
    # t_x = 10 #np.random.randint(-20, 20)
    # t_y = 0 # np.random.randint(-20, 20)
    # t_z = 0 # np.random.randint(-20, 20)
    # affine.SetTranslation((t_x, t_y, t_z))

    affine.Scale((scale[0], scale[1], scale[2]))

    radias = rotate / 180   # np.pi *
    affine.Rotate(axis1=0, axis2=1, angle=radias)     #affineR.Rotate(axis1=0, axis2=1, angle=radias)

    image = resample_aug_MRI(image, affine)
    label = resample_aug_MRI(label, affine)

    return image, label


def resample_aug_MRI(image, transform):

    reference_image = image
    interpolator = sitk.sitkNearestNeighbor
    default_value = 0.0

    img_dim = image.GetSize()

    if len(img_dim) == 4:

        img_np = sitk.GetArrayFromImage(image)
        img_np = np.asarray(img_np, np.float32)
        img_np = np.transpose(img_np, (3, 2, 1, 0))

        img_refer_np = sitk.GetArrayFromImage(reference_image)
        img_refer_np = np.asarray(img_refer_np, np.float32)
        img_refer_np = np.transpose(img_refer_np, (3, 2, 1, 0))

        t_slices = []
        for t in range(img_np.shape[3]):
            img_t = img_np[:, :, :, t]
            img_t = np.transpose(img_t, (2, 1, 0))
            img_t = sitk.GetImageFromArray(img_t)

            img_refer_t = img_refer_np[:, :, :, t]
            img_refer_t = np.transpose(img_refer_t, (2, 1, 0))
            img_refer_t = sitk.GetImageFromArray(img_refer_t)

            img_t = sitk.Resample(img_t, img_refer_t, transform, interpolator, default_value)

            ##################################################

            t_slices.append(img_t)

        img_out = sitk.JoinSeries(t_slices)
        img_out.SetOrigin(image.GetOrigin())
        img_out.SetDirection(image.GetDirection())

        return img_out

    else:

        return sitk.Resample(image, reference_image, transform, interpolator, default_value)


def generate_offline_test_batch_multi(img_cp_np, patch_size, step_size=2, cdfs=None, bins=None, test_name=""):

    test_patch_list = []
    test_patch_coords = []

    img_dim = img_cp_np.shape

    boundingAxes, mask_np = findMaskBoundingBox_multi(img_cp_np, thresIntens=10)
    sliding_size = (patch_size[0] // step_size, patch_size[1] // step_size, patch_size[2] // step_size)

    for c in range(img_dim[3]):
        img_cp_np[:, :, :, c] = histogram_matching(img_cp_np[:, :, :, c], cdf=cdfs[c], bins=bins[c])

    # mask_np = write_multi_channel_image(mask_np)
    # print("boundingAxes = ", boundingAxes, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
    # sitk.WriteImage(mask_np, "/home/jtma/PycharmProjects/dataset/test/" + test_name + "_cropped.nii.gz")

    X_start = boundingAxes[0]
    while X_start < img_dim[0]:

        X_end = min(X_start + patch_size[0], img_dim[0] - 1)
        Y_start = boundingAxes[2]
        while Y_start < img_dim[1]:

            Y_end = min(Y_start + patch_size[1], img_dim[1] - 1)
            Z_start = boundingAxes[4]
            while Z_start < img_dim[2]:

                Z_end = min(Z_start + patch_size[2], img_dim[2] - 1)
                patch_img = img_cp_np[X_start:X_end, Y_start:Y_end, Z_start:Z_end, :]  # or img_cp_np

                # for c in range(img_dim[3]):
                #     patch_img[:, :, :, c] = histogram_matching(patch_img[:, :, :, c], cdf=cdfs[c], bins=bins[c])

                if patch_img.shape[0] < patch_size[0] or patch_img.shape[1] < patch_size[1] or patch_img.shape[2] < patch_size[2]:
                    patch_img = getPaddingBox_multi(patch_img, out_size=patch_size, channel=img_dim[3])

                ###########################################################

                test_patch_list.append(patch_img)
                test_patch_coords.append([X_start, Y_start, Z_start])

                if Z_end < boundingAxes[5]:
                    Z_start += sliding_size[2]
                else:
                    break

            if Y_end < boundingAxes[3]:
                Y_start += sliding_size[1]
            else:
                break

        if X_end < boundingAxes[1]:
            X_start += sliding_size[0]
        else:
            break

    ############################## reshape the list ##########################3######

    test_patch_arr = np.asarray(test_patch_list)  # shape = (num_patches, patch_size)
    test_patch_coords_arr = np.asarray(test_patch_coords)

    return test_patch_arr, test_patch_coords_arr


def compute_histogram_of_train_images_multi(train_img_dir, train_label_dir, channel, num_bins=256):

    total_img_list = []
    train_files = os.listdir(train_img_dir)

    for file_index in range(len(train_files)):

        img_base_name = train_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.nii.gz'

        img = readItkData_multiclass(train_img_dir + train_files[file_index])
        label = readItkLabelData(train_label_dir + label_base_name + ext)

        img_np = sitk.GetArrayFromImage(img)
        img_np = np.transpose(img_np, (3, 2, 1, 0))
        img_np_t = img_np[:, :, :, channel]

        label_np = sitk.GetArrayFromImage(label)
        label_np = np.transpose(label_np, (2, 1, 0))

        boundingAxes = getBoundingAxes(label_np)
        img_patch = img_np_t[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3], boundingAxes[4]:boundingAxes[5]]

        dims = img_patch.shape
        arr = np.reshape(img_patch, (dims[0] * dims[1] * dims[2], -1))
        total_img_list.extend(arr.tolist())

    ########### histogram computation ########################

    print( " start computing the histogram of channel ", channel)

    ### histogram matching ###
    t_values, imhist = np.unique(total_img_list, return_counts=True)
    cdf = np.cumsum(imhist).astype(np.float64)
    cdf /= cdf[-1]

    print(" histogram computation is done ! ")

    return imhist, t_values, cdf   # imhist, bins, cdf


def compute_histogram_of_hippocampus(train_img_dir, train_label_dir, num_bins=256, hardIntens=2000):

    total_img_list = []

    train_files = os.listdir(train_img_dir)

    for file_index in range(len(train_files)):
        img_base_name = train_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.nii.gz'

        img = readItkData(train_img_dir + train_files[file_index])
        label = readItkLabelData(train_label_dir + label_base_name + ext)

        img_np = sitk.GetArrayFromImage(img)
        img_np = np.transpose(img_np, (2, 1, 0))

        label_np = sitk.GetArrayFromImage(label)
        label_np = np.transpose(label_np, (2, 1, 0))

        boundingAxes = getBoundingAxes(label_np)
        img_patch = img_np[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3],
                    boundingAxes[4]:boundingAxes[5]]

        ## linearly transform the intensity to [0, 2000]

        dims = img_patch.shape
        arr = np.reshape(img_patch, (dims[0] * dims[1] * dims[2], -1))
        arr_min, arr_max = np.min(arr), np.max(arr)

        arr_thres = [arr[i] * hardIntens / (arr_max - arr_min) for i in range(arr.shape[0])]

        total_img_list.extend(arr_thres)  # .tolist()

    ########### histogram computation ########################

    print(" start computing the histogram ... ")

    ### histogram matching ###
    t_values, imhist = np.unique(total_img_list, return_counts=True)
    cdf = np.cumsum(imhist).astype(np.float64)
    cdf /= cdf[-1]

    print(" histogram computation is done ! ")

    return imhist, t_values, cdf  # imhist, t_values, cdf   # imhist, bins, cdf


def resampleMRIData(img, new_spacing=(2.0, 2.0, 2.0), is_label=False):

    original_size = np.array(img.GetSize(), dtype=np.int)
    original_spacing = img.GetSpacing()

    out_size = [int(np.round(original_size[0] * (original_spacing[0] / new_spacing[0]))),
                int(np.round(original_size[1] * (original_spacing[1] / new_spacing[1]))),
                int(np.round(original_size[2] * (original_spacing[2] / new_spacing[2])))]


    resampler = sitk.ResampleImageFilter()
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(out_size)
    resampler.SetDefaultPixelValue(0)   # ct = -1024, mri = 0

    if is_label:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkBSpline)  # sitk.sitkLinear

    img_out = resampler.Execute(img)

    return img_out  # resampler.Execute(img)


def resampleItkDataMulti(img, new_spacing=(1.5, 1.5, 1.5)):

    img_np = sitk.GetArrayFromImage(img)
    img_np = np.asarray(img_np, np.float32)
    img_np = np.transpose(img_np, (3, 2, 1, 0))

    img_out = write_multi_channel_image(img_np, origin_spacing=img.GetSpacing(), spacing=new_spacing)
    img_out.SetOrigin(img.GetOrigin())
    img_out.SetDirection(img.GetDirection())

    return img_out


def findMaskBoundingBox_MRI(img_arr):

    mask = np.where(img_arr > 3)

    boundingAxes = [np.min(mask[0]), np.max(mask[0]),
                    np.min(mask[1]), np.max(mask[1]),
                    np.min(mask[2]), np.max(mask[2])]

    x_dim = boundingAxes[1] - boundingAxes[0] + 1
    y_dim = boundingAxes[3] - boundingAxes[2] + 1
    z_dim = boundingAxes[5] - boundingAxes[4] + 1

    mask = np.zeros((x_dim, y_dim, z_dim), dtype=np.float32)
    mask[:x_dim - 1, :y_dim - 1, :z_dim - 1] = img_arr[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3],
                                               boundingAxes[4]:boundingAxes[5]]

    # xmin = img_dim[0] - 1 - np.max(mask[0])
    # xmax = img_dim[0] - 1 - np.min(mask[0])
    # ymin = img_dim[1] - 1 - np.max(mask[1])
    # ymax = img_dim[1] - 1 - np.min(mask[1])

    return boundingAxes, mask


def findMaskBoundingBox_multi(img_arr, thresIntens=0):

    """
    crop the ROI in terms of the first channel for mri
    :param img_arr:
    :param thresIntens:
    :return:
    """

    img_dim = img_arr.shape

    mask = np.where(img_arr[:, :, :, 0] > thresIntens)

    boundingAxes = [np.min(mask[0]), np.max(mask[0]),
                    np.min(mask[1]), np.max(mask[1]),
                    np.min(mask[2]), np.max(mask[2])]

    x_dim = boundingAxes[1] - boundingAxes[0] + 1
    y_dim = boundingAxes[3] - boundingAxes[2] + 1
    z_dim = boundingAxes[5] - boundingAxes[4] + 1

    mask = np.zeros((x_dim, y_dim, z_dim, img_dim[3]), dtype=np.float32)
    mask[:x_dim - 1, :y_dim - 1, :z_dim - 1, :] = img_arr[boundingAxes[0]:boundingAxes[1],
                                                  boundingAxes[2]:boundingAxes[3],
                                                  boundingAxes[4]:boundingAxes[5], :]


    return boundingAxes, mask


def getPaddingBox_MRI(array, out_size=(128, 64, 64)):

    """
    pad the array with zeros if smaller than the out size
    :param array: input array - 3d
    :param out_size: the target size
    :return: the padded array with the output size
    """

    dims = array.shape
    min_dims = [min(dims[0], out_size[0]), min(dims[1], out_size[1]), min(dims[2], out_size[2])]

    out_arr = np.zeros(out_size)
    out_arr[:min_dims[0], :min_dims[1], :min_dims[2]] = array[:min_dims[0], :min_dims[1], :min_dims[2]]

    return out_arr


def getPaddingBox_multi(array, out_size=(128, 64, 64), channel=2):

    dims = array.shape

    out_arr = np.zeros((out_size[0], out_size[1], out_size[2], channel))

    out_arr[:min(int(dims[0]-1), int(out_size[0]-1)), :min(int(dims[1]-1), int(out_size[1]-1)), :min(int(dims[2]-1), int(out_size[2]-1)) :] = \
        array[:min(int(dims[0]-1), int(out_size[0]-1)), :min(int(dims[1]-1), int(out_size[1]-1)), :min(int(dims[2]-1), int(out_size[2]-1)) :]


    return out_arr


def cropCenter_multi(array, out_size=(64, 64, 64,)):

    """
    crop the array from center
    :param array: input array which are larger than the out_size
    :param out_size: aim size
    :return: cropped area
    """

    center = [array.shape[0] // 2, array.shape[1] // 2, array.shape[2] // 2]

    X_start = max(0, center[0] - out_size[0] // 2)
    Y_start = max(0, center[1] - out_size[1] // 2)
    Z_start = max(0, center[2] - out_size[2] // 2)

    X_end = min(array.shape[0] - 1, X_start + out_size[0])
    Y_end = min(array.shape[1] - 1, Y_start + out_size[1])
    Z_end = min(array.shape[2] - 1, Z_start + out_size[2])

    cropped_arr = array[X_start:X_end, Y_start:Y_end, Z_start:Z_end, :]

    if cropped_arr.shape[0] < out_size[0] or cropped_arr.shape[1] < out_size[1] or cropped_arr.shape[2] < out_size[2]:
        cropped_arr = getPaddingBox_multi(cropped_arr, out_size=out_size, channel=array.shape[3])

    return cropped_arr


def crop_center_from_label_multi(img, label, patch_size=(64, 64, 32), channel=2):

    boundingAxes = getBoundingAxes(label)

    X_center = int(boundingAxes[0] + boundingAxes[1]) // 2
    Y_center = int(boundingAxes[2] + boundingAxes[3]) // 2
    Z_center = int(boundingAxes[4] + boundingAxes[5]) // 2

    X_start = max(0, X_center - patch_size[0] // 2)
    Y_start = max(0, Y_center - patch_size[1] // 2)
    Z_start = max(0, Z_center - patch_size[2] // 2)

    X_end = min(X_start + patch_size[0], img.shape[0] - 1)
    Y_end = min(Y_start + patch_size[1], img.shape[1] - 1)
    Z_end = min(Z_start + patch_size[2], img.shape[2] - 1)

    cropped_img = img[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end), :]
    cropped_label = label[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end)]

    if cropped_img.shape[0] > patch_size[0] or cropped_img.shape[1] > patch_size[1] or cropped_img.shape[2] > patch_size[2]:

        cropped_img = cropCenter_multi(cropped_img, out_size=patch_size)
        cropped_label = cropCenter_MRI(cropped_label, out_size=patch_size)

        print("cropped_imng cropcenter = ", cropped_img.shape)

    if cropped_img.shape[0] < patch_size[0] or cropped_img.shape[1] < patch_size[1] or cropped_img.shape[2] < patch_size[2]:

        cropped_img = getPaddingBox_multi(cropped_img, out_size=patch_size, channel=channel)
        cropped_label = getPaddingBox_MRI(cropped_label, out_size=patch_size)

        print("padding shape = ", cropped_img.shape)

    return cropped_img, cropped_label, boundingAxes


def cropCenter_MRI(array, out_size=(64, 64, 64,)):

    """
    crop the array from center
    :param array: input array which are larger than the out_size
    :param out_size: aim size
    :return: cropped area
    """

    center = [array.shape[0] // 2, array.shape[1] // 2, array.shape[2] // 2]

    X_start = max(0, center[0] - out_size[0] // 2)
    Y_start = max(0, center[1] - out_size[1] // 2)
    Z_start = max(0, center[2] - out_size[2] // 2)

    X_end = min(array.shape[0] - 1, X_start + out_size[0])
    Y_end = min(array.shape[1] - 1, Y_start + out_size[1])
    Z_end = min(array.shape[2] - 1, Z_start + out_size[2])

    cropped_arr = array[X_start:X_end, Y_start:Y_end, Z_start:Z_end]

    if cropped_arr.shape[0] < out_size[0] or cropped_arr.shape[1] < out_size[1] or cropped_arr.shape[2] < out_size[2]:
        cropped_arr = getPaddingBox_MRI(cropped_arr, out_size=out_size)

    return cropped_arr


def generate_offline_test_batch_MRI(img_np, patch_size, step_size=2, cdf=None, bins=None, test_name=""):

    test_patch_list = []
    test_patch_coords = []

    img_dim = img_np.shape

    boundingAxes, mask_np = findMaskBoundingBox_MRI(img_np)
    sliding_size = (patch_size[0] // step_size, patch_size[1] // step_size, patch_size[2] // step_size)

#    print("boundingAxes = ", boundingAxes, boundingAxes[1] - boundingAxes[0], boundingAxes[3] - boundingAxes[2], boundingAxes[5] - boundingAxes[4])
#    sitk.WriteImage(sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0))), "/home/jtma/PycharmProjects/dataset/test/" + test_name + "_cropped.nii.gz")

    X_start = boundingAxes[0]
    while X_start < img_dim[0]:

        X_end = min(X_start + patch_size[0], img_dim[0] - 1)
        Y_start = boundingAxes[2]
        while Y_start < img_dim[1]:

            Y_end = min(Y_start + patch_size[1], img_dim[1] - 1)
            Z_start = boundingAxes[4]
            while Z_start < img_dim[2]:

                Z_end = min(Z_start + patch_size[2], img_dim[2] - 1)
                patch_img = img_np[X_start:X_end, Y_start:Y_end, Z_start:Z_end]  # or img_cp_np

                ###########################################################

                patch_img = histogram_matching(patch_img, cdf=cdf, bins=bins)

                if patch_img.shape[0] < patch_size[0] or patch_img.shape[1] < patch_size[1] or patch_img.shape[2] < patch_size[2]:
                    patch_img = getPaddingBox_MRI(patch_img, out_size=patch_size)

                ###########################################################

                test_patch_list.append(patch_img)
                test_patch_coords.append([X_start, Y_start, Z_start])

                if Z_end < boundingAxes[5]:
                    Z_start += sliding_size[2]
                else:
                    break

            if Y_end < boundingAxes[3]:
                Y_start += sliding_size[1]
            else:
                break

        if X_end < boundingAxes[1]:
            X_start += sliding_size[0]
        else:
            break

    ############################## reshape the list ##########################3######

    test_patch_list = np.asarray(test_patch_list)  # shape = (num_patches, patch_size)
    test_patch_coords = np.asarray(test_patch_coords)

    return test_patch_list, test_patch_coords


def crop_center_multi(img, label, patch_size=(96, 96, 32), channel=2):

    X_center, Y_center, Z_center = img.shape[0] // 2, img.shape[1] // 2, img.shape[2] // 2

    X_start = max(0, X_center - patch_size[0] // 2)
    Y_start = max(0, Y_center - patch_size[1] // 2)
    Z_start = max(0, Z_center - patch_size[2] // 2)

    X_end = X_start + patch_size[0] if X_start + patch_size[0] < img.shape[0] else img.shape[0] - 1
    Y_end = Y_start + patch_size[1] if Y_start + patch_size[1] < img.shape[1] else img.shape[1] - 1
    Z_end = Z_start + patch_size[2] if Z_start + patch_size[2] < img.shape[2] else img.shape[2] - 1

    cropped_img = img[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end), :]
    cropped_label = label[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end)]

    if cropped_img.shape[0] < patch_size[0] or cropped_img.shape[1] < patch_size[1] or cropped_img.shape[2] < patch_size[2]:

        cropped_img = getPaddingBox_multi(cropped_img, out_size=patch_size, channel=channel)
        cropped_label = getPaddingBox_MRI(cropped_label, out_size=patch_size)

    return cropped_img, cropped_label


def crop_center_from_label_MRI(img, label, patch_size=(64, 64, 32)):

    boundingAxes = getBoundingAxes(label)

    X_center = int(boundingAxes[0] + boundingAxes[1]) // 2
    Y_center = int(boundingAxes[2] + boundingAxes[3]) // 2
    Z_center = int(boundingAxes[4] + boundingAxes[5]) // 2

    X_start = max(0, X_center - patch_size[0] // 2)
    Y_start = max(0, Y_center - patch_size[1] // 2)
    Z_start = max(0, Z_center - patch_size[2] // 2)

    X_end = X_start + patch_size[0] if X_start + patch_size[0] < img.shape[0] else img.shape[0] - 1
    Y_end = Y_start + patch_size[1] if Y_start + patch_size[1] < img.shape[1] else img.shape[1] - 1
    Z_end = Z_start + patch_size[2] if Z_start + patch_size[2] < img.shape[2] else img.shape[2] - 1

    cropped_img = img[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end)]
    cropped_label = label[int(X_start):int(X_end), int(Y_start):int(Y_end), int(Z_start):int(Z_end)]

    if cropped_img.shape[0] < patch_size[0] or cropped_img.shape[1] < patch_size[1] or cropped_img.shape[2] < patch_size[2]:

        cropped_img = getPaddingBox_MRI(cropped_img, out_size=patch_size)
        cropped_label = getPaddingBox_MRI(cropped_label, out_size=patch_size)   # background set to 0

    return cropped_img, cropped_label, boundingAxes



#######################################################################################################################


def merge_patches_back(resampled_np, patch_arr, coords_arr, patch_size=(128, 64, 64)):

    out_mask = np.zeros((resampled_np.shape[0], resampled_np.shape[1], resampled_np.shape[2]), dtype=np.float32)
    print("out mask shape = ", out_mask.shape)

    for indx in range(patch_arr.shape[0]):

        patch_pred = np.argmax(patch_arr[indx, :, :, :, :], axis=-1)
        patch_coords = coords_arr[indx]

        print("pred shape = ", patch_pred.shape)

        X_start, Y_start, Z_start = patch_coords[0], patch_coords[1], patch_coords[2]
        X_end = min(out_mask.shape[0] - 1, X_start + patch_size[0])
        Y_end = min(out_mask.shape[1] - 1, Y_start + patch_size[1])
        Z_end = min(out_mask.shape[2] - 1, Z_start + patch_size[2])

        print(X_start, X_end, Y_start, Y_end, Z_start, Z_end)

        # print("insideO - ", out_mask[X_start:X_end, Y_start:Y_end, Z_start:Z_end].shape)
        # print("insideP - ", patch_pred[0:(X_end - X_start), 0:(Y_end - Y_start), 0:(Z_end - Z_start)].shape)

        out_mask[X_start:X_end, Y_start:Y_end, Z_start:Z_end] = np.add(out_mask[X_start:X_end, Y_start:Y_end, Z_start:Z_end],
                                                                       patch_pred[0:(X_end - X_start), 0:(Y_end - Y_start), 0:(Z_end - Z_start)])

    return out_mask


def merge_patches_back_multi(resampled_np, patch_arr, coords_arr, patch_size=(128, 64, 64)):

    out_mask = np.zeros((resampled_np.shape[0], resampled_np.shape[1], resampled_np.shape[2]), dtype=np.float32)
    print("out mask shape = ", out_mask.shape)

    for indx in range(patch_arr.shape[0]):

        patch_pred = np.argmax(patch_arr[indx, :, :, :, :], axis=-1)   # shape = (batch, h, w, d, c)
        patch_coords = coords_arr[indx]

        print("pred shape = ", patch_pred.shape)

        X_start, Y_start, Z_start = patch_coords[0], patch_coords[1], patch_coords[2]
        X_end = min(out_mask.shape[0] - 1, X_start + patch_size[0])
        Y_end = min(out_mask.shape[1] - 1, Y_start + patch_size[1])
        Z_end = min(out_mask.shape[2] - 1, Z_start + patch_size[2])

        print(X_start, X_end, Y_start, Y_end, Z_start, Z_end)

        # print("insideO - ", out_mask[X_start:X_end, Y_start:Y_end, Z_start:Z_end].shape)
        # print("insideP - ", patch_pred[0:(X_end - X_start), 0:(Y_end - Y_start), 0:(Z_end - Z_start)].shape)

        out_mask[X_start:X_end, Y_start:Y_end, Z_start:Z_end] = np.add(out_mask[X_start:X_end, Y_start:Y_end, Z_start:Z_end],
                                                                       patch_pred[0:(X_end - X_start), 0:(Y_end - Y_start), 0:(Z_end - Z_start)])

    return out_mask


def histogram_matching(img_arr, cdf, bins):

    img_shape = img_arr.shape

    s_values, bin_indx, s_counts = np.unique(img_arr.flatten(), return_inverse=True, return_counts=True)
    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, cdf, bins)  # t_values in origin
    img_arr = interp_t_values[bin_indx]  # mask_np = np.interp(mask_np.flatten(), self.bins[:-1], self.cdf)

    img_arr = np.reshape(img_arr, img_shape)

    return img_arr


def compute_histogram_of_train_images(train_img_dir, train_label_dir, num_bins=256):

    total_img_list = []

    train_files = os.listdir(train_img_dir)

    for file_index in range(len(train_files)):

        img_base_name = train_files[file_index].split('.')[0]
        label_base_name = img_base_name + "_mask"
        ext = '.nii.gz'

        img = readItkData(train_img_dir + train_files[file_index])
        label = readItkLabelData(train_label_dir + label_base_name + ext)

        img_np = sitk.GetArrayFromImage(img)
        img_np = np.transpose(img_np, (2, 1, 0))

        label_np = sitk.GetArrayFromImage(label)
        label_np = np.transpose(label_np, (2, 1, 0))

        boundingAxes = getBoundingAxes(label_np)
        img_patch = img_np[boundingAxes[0]:boundingAxes[1], boundingAxes[2]:boundingAxes[3], boundingAxes[4]:boundingAxes[5]]

        dims = img_patch.shape
        arr = np.reshape(img_patch, (dims[0] * dims[1] * dims[2], -1))
        total_img_list.extend(arr.tolist())

        print(img_base_name, img_patch.shape, img.GetSpacing())

    ########### histogram computation ########################

    print(" start computing the histogram ... ")

    ### histogram matching ###
    t_values, imhist = np.unique(total_img_list, return_counts=True)
    cdf = np.cumsum(imhist).astype(np.float64)
    cdf /= cdf[-1]

    # imhist, bins = np.histogram(total_img_list, bins=num_bins, density=True)
    # cdf = imhist.cumsum()  # cumulative distribution function
    # cdf = imhist.max() * cdf / cdf[-1]  # normalize, cdf[-1] = cdf.max()    # without * 255 !!!!

    print(" histogram computation is done ! ")

    return imhist, t_values, cdf  # imhist, t_values, cdf   # imhist, bins, cdf


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


def get_dirichlet_variance(alphas):

    """
    return the trace_of_covariance_matrix of Dirichlet PDF
    the calculation is followed by NIPS-19 (unsured)
    :param alphas: input the alphas with shape (h, w, d, multi_channels)
    :return: variance map (trace of covariance matrix)
    """

    ## self.gen_fused has the shape (batch_size, h, w, d, channel=3)
    sum_alphas = np.sum(alphas, axis=-1)  # shape = (h, w, d)
    squared_sum_alphas = np.square(sum_alphas)

    squared_alphas = np.square(alphas[:, :, :, 0])
    for t in range(1, alphas.shape[3]):
        squared_alphas = np.add(squared_alphas, np.square(alphas[:, :, :, t]))

    trace_cov = np.divide(squared_sum_alphas - squared_alphas, np.multiply(squared_sum_alphas, (1 + sum_alphas)))

    # # sum_trace_covariance
    # sum_alphas = np.sum(alphas, axis=-1)  # shape = (h, w, d)
    # squared_sum_alphas = np.square(sum_alphas)
    #
    # denominator = np.multiply(squared_sum_alphas, (1 + sum_alphas))
    #
    # squared_alphas = np.square(alphas[:, :, :, 0])
    # for t in range(1, alphas.shape[3]):
    #     squared_alphas = np.add(squared_alphas, np.square(alphas[:, :, :, t]))

    # trace_cov = np.divide( - squared_alphas, denominator)

    return trace_cov


def load_model(sess, checkpoint_dir):

    print(" [*] Reading checkpoint...")

    print("check point dir", checkpoint_dir)

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = os.path.basename(ckpt.model_checkpoint_path)

        print('ckpt-name', ckpt_name, os.path.join(checkpoint_dir, ckpt_name))

        tf.train.Saver().restore(sess, os.path.join(checkpoint_dir, ckpt_name))

        print(" [*] Loading checkpoint done...")

        return True
    else:

        print(" [!] Loading checkpoint failed...")

        return False


def get_session():

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False, gpu_options=gpu_options)

#    session = tf.Session(config=config)

    if tf.get_default_session() is None:
        print("Creating new session")

        tf.reset_default_graph()
        _SESSION = tf.InteractiveSession(config=config)
    else:
        print("Using old session")

        _SESSION = tf.get_default_session(config=config)

    return _SESSION


##########################################################################################################################


def readItkData(filename):

    # open the shape with sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    img = reader.Execute()

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkFloat32)
    img = castImageFilter.Execute(img)

    return img


def readItkLabelData(filename):

    # open the shape with sitk
    reader = sitk.ImageFileReader()
    reader.SetFileName(filename)
    label = reader.Execute()

    castImageFilter = sitk.CastImageFilter()
    castImageFilter.SetOutputPixelType(sitk.sitkUInt8)
    label = castImageFilter.Execute(label)

    #    label = sitk.BinaryThreshold(label, lowerThreshold=1.0, upperThreshold=2.5, insideValue=1, outsideValue=0)

    return label


def readItkData_multiclass(filename):

    img = sitk.ReadImage(filename, sitk.sitkVectorFloat32)

    return img


def getBoundingAxes(label):

    """
    :param label: an array
    :return: return the bounding box axes (xmin, xmax, ymin, ymax, zmin, zmax)
    """

    r = np.any(label, axis=(1, 2))
    c = np.any(label, axis=(0, 2))
    z = np.any(label, axis=(0, 1))

    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]

    bounding_axes = [rmin, rmax, cmin, cmax, zmin, zmax]

    return bounding_axes


def getBoundingBox(label):

    """
    find the bounding box of an ITK image
    :param label: itk label
    :return:
    """

    label_shape_filter = sitk.LabelShapeStatisticsImageFilter()
    label_shape_filter.Execute(label)

    return label_shape_filter.GetBoundingBox(1)


def flip_rotate(img_np, label_np):

    f_r = [True, False]
    to_flip = np.random.choice(f_r, 1)
    to_rotate = np.random.choice(f_r, 1)

    if to_flip:
        img_np = img_np[::-1]
        label_np = label_np[::-1]

    if to_rotate:
        img_np = np.rot90(img_np, 3)
        label_np = np.rot90(label_np, 3)

    return img_np, label_np



def get_val_accuracy_bpgan_fusion_pgan(session, model, val_imgs, val_labels, model_name, batch_val_files, FLAGS):

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/'

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)


    predicts = session.run(model.predicts, feed_dict={model.imgs: val_imgs})
    predicts_arr = np.asarray(predicts)

    dice_total_mean, dice_total_fused_mean = [], []

    for g in range(predicts_arr.shape[0]):

        dice_g_mean, dice_g_fuse_mean = [], []

        for b in range(FLAGS.batch_size):

            val_img_base_name = batch_val_files[b].split(".")[0]

            pred_b_g = predicts_arr[g, b, :, :, :, :]    # shape = (d, w, h, channel)   # b * (num_crop + 1)
            label_b_g = val_labels[b, :, :, :]           # shape = (d, w, h, channel)
            img_b_g = val_imgs[b, :, :, :, :]            # shape = (d, w, h)

            pred_b_g_argmax = np.argmax(np.round(pred_b_g), axis=-1).astype(np.float32)   # shape = (d, w, h)

            print("f1 score of geng = %d " % g, " batch = %d" % b, " : ", f1_score(pred_b_g_argmax.flatten(), label_b_g.flatten(), average=None))

            dice_g_mean.append(dice_coefficient(pred_b_g_argmax, label_b_g))  #    jaccard_similarity_score(pred_b_g_argmax.flatten(), label_b_g.flatten()))
            dice_g_fuse_mean.append(dice_coefficient(pred_b_g_argmax, label_b_g))

                #######################################################################################################

            slices_img = []
            for t in range(img_b_g.shape[3]):
                slices_img.append(sitk.GetImageFromArray(np.transpose(img_b_g[:, :, :, t], (2, 1, 0))))
            img = sitk.JoinSeries(slices_img)

            base_dir = project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/'

            pred_b_g_argmax = np.transpose(pred_b_g_argmax, (2, 1, 0))
            label_b_g = np.transpose(label_b_g, (2, 1, 0))

            sitk.WriteImage(sitk.GetImageFromArray(pred_b_g_argmax), base_dir + val_img_base_name + '_seg.nii.gz')  # + str(indx)  pred_b_g_argmax
            sitk.WriteImage(sitk.GetImageFromArray(label_b_g), base_dir + val_img_base_name + '_label.nii.gz')
            sitk.WriteImage(img, base_dir + val_img_base_name + '_img.nii.gz')

            ##############################################################

        dice_total_mean.append(np.mean(dice_g_mean))
        dice_total_fused_mean.append(np.mean(dice_g_fuse_mean))


    return dice_total_mean, dice_total_fused_mean




##############################################################################################
