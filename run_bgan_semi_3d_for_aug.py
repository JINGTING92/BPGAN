#!/usr/bin/env python

import os, sys, argparse, json, time, shutil
import numpy as np
from math import ceil
from random import shuffle

import SimpleITK as sitk
import tensorflow as tf
import cv2
from tensorflow.contrib import slim
from sklearn.metrics import jaccard_similarity_score

from tensorflow.python.framework import ops


from bgan_util_3d import *
from unet import UNet
from bpgan_3d import B_PATCHGAN_3D
from bpgan_3d_reduced import B_PATCHGAN_3D_REDUCE
from bpgan_3d_extra import B_PATCHGAN_3D_EXTRA
from seg_utils import *
from utilities import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# session = tf.Session(config=config)

project_path = '/home/jtma/PycharmProjects/'

flags = tf.app.flags
flags.DEFINE_integer("gf_dim", 32, "num of gen features")  # change it to 32
flags.DEFINE_integer("df_dim", 32, "num of disc features")   # default 96
flags.DEFINE_string("data_path", project_path + "dataset", "path to where the datasets live")
flags.DEFINE_string("dataset", "Task07_Pancreas", "datasets name cifar etc.")   # cifar | histopatho | mnist | abdomenCT
flags.DEFINE_integer("batch_size", 1, "mini batch size")
flags.DEFINE_integer("num_aug", 0, 'number of augmented data for each datasets in supervised training branch')
flags.DEFINE_float("prior_std", 1.0, "NN weight prior std")
flags.DEFINE_integer("J", 1, "number of samples of z / generators")
flags.DEFINE_integer("J_d", 1, "number of discriminator weight samples")
flags.DEFINE_integer("M", 1, "number of MCMC NN weight samples per z")
flags.DEFINE_integer("train_iter", 80000, "number of training iterations")
flags.DEFINE_integer("random_seed", 2222, "random seed")
flags.DEFINE_float("lr", 0.0005, "learning rate")
flags.DEFINE_float("lr_decay", 3.0, "learning rate")
flags.DEFINE_string("optimizer", "adam", "optimizer -- adam or sgd")
flags.DEFINE_boolean("ml", False, "if specified, disable bayesian things")
flags.DEFINE_string("fn", "f1", "number of the fold of the cross validation")
flags.DEFINE_boolean("da", False, "whether to do data augmentation00")
flags.DEFINE_integer("img_size", 128, "default image width / length")
flags.DEFINE_boolean("wassertein", True, "wasserstein GAN")
flags.DEFINE_float("l1_lambda", 1, "l1_lambda of the l1 norm for segmentation")  # 10 works well
flags.DEFINE_string('checkpoint_dir', project_path + 'checkpoint', 'models are saved here')
flags.DEFINE_integer('num_classes', 1, "number of classes for segmentation")
flags.DEFINE_string('phase', 'train', "train or test")
flags.DEFINE_string('task', 'BrainTumor', 'select the task')
flags.DEFINE_boolean('restore', False, 'whether restore the model and continue training')
flags.DEFINE_boolean('extra_pooling', False, 'whether restore the model and continue training')
flags.DEFINE_boolean('reduce_pooling', False, 'whether restore the model and continue training')
flags.DEFINE_boolean('write_out', False, 'whether restore the model and continue training')
flags.DEFINE_float('val_ratio', 0.1, 'the validation ratio; if 0, full size training')
FLAGS = flags.FLAGS

########################################################################################################################


def augmentation(dataset, FLAGS):

    train_img_files = random.sample(dataset.aug_img_files, len(dataset.aug_img_files))

    num_train = len(dataset.aug_img_files)
    print("num_train = ", num_train)

    train_indices = np.random.choice(num_train, num_train, replace=False)

    batch_train_files = []
    for _, idx in enumerate(train_indices):
        batch_train_files.append(train_img_files[idx])

    dataset.augment_data(train_img_dir=dataset.train_img_dir,
                             train_label_dir=dataset.train_label_dir,
                             batch_img_files=batch_train_files,
                             num_aug=FLAGS.num_aug)


##########################################################################################################################


def test_ct(dataset, FLAGS, patch_size):


    model_dir = dataset.name + "_%s_%s_%s_%s" % ('bpgan3d', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    tf.set_random_seed(FLAGS.random_seed)

    save_seg_dir = project_path + 'segmentation_results/' + model_dir + '/'
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)

    ##########################################################################################################################

    test_img_files = random.sample(os.listdir(dataset.test_img_dir), len(os.listdir(dataset.test_img_dir)))
    num_test = len(os.listdir(dataset.test_img_dir))

    test_indices = np.random.choice(num_test, num_test, replace=False)

    batch_test_files = []
    for _, idx in enumerate(test_indices):
        batch_test_files.append(test_img_files[idx])


    for indx in range(len(batch_test_files)):

        test_img_name = batch_test_files[indx]
        img_base_name = test_img_name.split('.')[0]

        test_origin_img = readItkData(dataset.test_img_origin_dir + test_img_name)
        resampled_origin_img = resampleCTData(test_origin_img, new_spacing=dataset.new_spacing, is_label=False)
        resampled_img_np = np.transpose(sitk.GetArrayFromImage(resampled_origin_img), (2, 1, 0)).astype(np.float32)

        print("read image from ", dataset.test_img_dir + test_img_name)

        patch_array, coords_array = dataset.generate_offline_test_batch(test_img_dir=dataset.test_img_dir, test_name=test_img_name, write_out=FLAGS.write_out)

        print("get offline test batch = ", img_base_name, " get patches no. = ", patch_array.shape[0])

        batch_size = patch_array.shape[0]

        ######################## train the model #####################################

        # ops.reset_default_graph()
        # session = get_session()
        #
        # if FLAGS.extra_pooling:
        #
        #     model = B_PATCHGAN_3D_EXTRA(x_dim=patch_size, batch_size=batch_size, J=FLAGS.J,
        #                           lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
        #                           num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
        #                           num_classes=FLAGS.num_classes, channel=dataset.channel)
        #
        #
        # elif FLAGS.reduce_pooling:
        #
        #     model = B_PATCHGAN_3D_REDUCE(x_dim=patch_size, batch_size=batch_size, J=FLAGS.J,
        #                                  lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
        #                                  num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
        #                                  num_classes=FLAGS.num_classes, channel=dataset.channel)
        #
        # else:
        #
        #     model = B_PATCHGAN_3D(x_dim=patch_size, batch_size=batch_size, J=FLAGS.J,
        #                           lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
        #                           num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
        #                           num_classes=FLAGS.num_classes, channel=dataset.channel)
        #
        #
        #
        # load_model(session, checkpoint_dir)
        #
        # patch_array = patch_array[:, :, :, :, np.newaxis]
        # predicts_all, fusion = session.run([model.predicts, model.gen_fused], feed_dict={model.imgs: patch_array})
        # predicts_all = np.asarray(predicts_all)
        #
        # print("predicts_all = ", predicts_all.shape)   # shape = (J, batch_size, h, w, d, num_classes)

        #####################################################################################################################

        # for j in range(predicts_all.shape[0]):
        #
        #     mask_np = merge_patches_back(resampled_np=resampled_img_np, patch_arr=predicts_all[j, :, :, :, :, :], coords_arr=coords_array, patch_size=dataset.patch_size)
        #
        #     mask = sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0)))
        #     mask.SetSpacing(dataset.new_spacing)
        #
        #     mask = resampleCTData(mask, new_spacing=test_origin_img.GetSpacing(), is_label=False)
        #     mask.SetDirection(test_origin_img.GetDirection())
        #     mask.SetOrigin(test_origin_img.GetOrigin())
        #
        #     sitk.WriteImage(mask, save_seg_dir + img_base_name + "_" + str(j) + "_seg.nii.gz")
        #
        # #######################################################################################################################
        #
        # mask_fused = merge_patches_back(resampled_np=resampled_img_np, patch_arr=fusion, coords_arr=coords_array, patch_size=dataset.patch_size)
        #
        # mask_fused_img = sitk.GetImageFromArray(np.transpose(mask_fused, (2, 1, 0)))
        # mask_fused_img.SetSpacing(dataset.new_spacing)
        # mask_fused_img = resampleMRIData(mask_fused_img, new_spacing=test_origin_img.GetSpacing(), is_label=False)
        # mask_fused_img.SetDirection(test_origin_img.GetDirection())
        # mask_fused_img.SetOrigin(test_origin_img.GetOrigin())
        #
        # sitk.WriteImage(mask_fused_img, save_seg_dir + img_base_name + '_fuse.nii.gz')

        # session.close()


##########################################################################################################################


def test_mri(dataset, FLAGS, patch_size):

    model_dir = dataset.name + "_%s_%s_%s_%s" % ('bpgan3d', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    tf.set_random_seed(FLAGS.random_seed)

    save_seg_dir = project_path + 'segmentation_results/' + model_dir + '/'
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)

    ##############################################################################################

    test_img_files = random.sample(os.listdir(dataset.test_img_dir), len(os.listdir(dataset.test_img_dir)))
    num_test = len(os.listdir(dataset.test_img_dir))

    test_indices = np.random.choice(num_test, num_test, replace=False)

    batch_test_files = []
    for _, idx in enumerate(test_indices):
        batch_test_files.append(test_img_files[idx])


    for indx in range(len(batch_test_files)):

        test_img_name = batch_test_files[indx]
        img_base_name = test_img_name.split('.')[0]

        test_origin_img = readItkData(dataset.test_img_origin_dir + test_img_name)
        resampled_origin_img = resampleMRIData(test_origin_img, new_spacing=dataset.new_spacing, is_label=False)
        resampled_img_np = np.transpose(sitk.GetArrayFromImage(resampled_origin_img), (2, 1, 0)).astype(np.float32)

        print("read image from ", dataset.test_img_dir + test_img_name)

        patch_array, coords_array = dataset.generate_offline_test_batch(test_img_dir=dataset.test_img_dir, test_name=test_img_name, write_out=FLAGS.write_out)

        print("get offline test batch = ", img_base_name, " get patches no. = ", patch_array.shape[0])

        batch_size = patch_array.shape[0]
        print("batch_size = ", batch_size)

        ######################## train the model #####################################

        ops.reset_default_graph()
        session = get_session()

        model = B_PATCHGAN_3D(x_dim=patch_size, batch_size=batch_size, J=FLAGS.J,
                              lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                              num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
                              num_classes=FLAGS.num_classes, channel=dataset.channel)


        load_model(session, checkpoint_dir)   # previous model_dir

        patch_array = patch_array[:, :, :, :, np.newaxis]
        predicts_all, fusion = session.run([model.predicts, model.gen_fused], feed_dict={model.imgs: patch_array})
        predicts_all = np.asarray(predicts_all)

        print("predicts_all = ", predicts_all.shape, fusion.shape)   # shape = (J, batch_size, h, w, d, num_classes)

        #####################################################################################################################

        for j in range(predicts_all.shape[0]):

            mask_np = merge_patches_back(resampled_np=resampled_img_np, patch_arr=predicts_all[j, :, :, :, :, :], coords_arr=coords_array, patch_size=dataset.patch_size)

            mask = sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0)).astype(np.float32))
            mask.SetSpacing(dataset.new_spacing)

            mask = resampleMRIData(mask, new_spacing=test_origin_img.GetSpacing(), is_label=False)
            mask.SetDirection(test_origin_img.GetDirection())
            mask.SetOrigin(test_origin_img.GetOrigin())

            sitk.WriteImage(mask, save_seg_dir + img_base_name + "_" + str(j) + "_seg.nii.gz")

        #######################################################################################################################

        mask_fused = merge_patches_back(resampled_np=resampled_img_np, patch_arr=fusion, coords_arr=coords_array, patch_size=dataset.patch_size)

        mask_fused_img = sitk.GetImageFromArray(np.transpose(mask_fused, (2, 1, 0)).astype(np.float32))
        mask_fused_img.SetSpacing(dataset.new_spacing)

        mask_fused_img = resampleMRIData(mask_fused_img, new_spacing=test_origin_img.GetSpacing(), is_label=False)
        mask_fused_img.SetDirection(test_origin_img.GetDirection())
        mask_fused_img.SetOrigin(test_origin_img.GetOrigin())

        sitk.WriteImage(mask_fused_img, save_seg_dir + img_base_name + '_fuse.nii.gz')

        session.close()


##########################################################################################################################


def test_multi_mri(dataset, FLAGS, patch_size):

    model_dir = dataset.name + "_%s_%s_%s_%s" % ('bpgan3d', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    tf.set_random_seed(FLAGS.random_seed)

    save_seg_dir = project_path + 'segmentation_results/' + model_dir + '/'
    if not os.path.exists(save_seg_dir):
        os.makedirs(save_seg_dir)

    ##############################################################################################

    test_img_files = random.sample(os.listdir(dataset.test_img_dir), len(os.listdir(dataset.test_img_dir)))
    num_test = len(os.listdir(dataset.test_img_dir))

    test_indices = np.random.choice(num_test, num_test, replace=False)

    batch_test_files = []
    for _, idx in enumerate(test_indices):
        batch_test_files.append(test_img_files[idx])


    for indx in range(len(batch_test_files)):

        test_img_name = batch_test_files[indx]
        img_base_name = test_img_name.split('.')[0]

        test_origin_img = readItkData_multiclass(dataset.test_img_origin_dir + test_img_name)
        resampled_origin_img = resampleItkDataMulti(test_origin_img, new_spacing=dataset.new_spacing)
        resampled_img_np = np.transpose(sitk.GetArrayFromImage(resampled_origin_img), (3, 2, 1, 0)).astype(np.float32)

        direction = np.asarray(test_origin_img.GetDirection())
        direction = np.reshape(direction, (4, 4))[:3, :3]
        new_direction = tuple(direction.reshape(1, -1)[0])

        print("read image from ", dataset.test_img_dir + test_img_name)

        patch_array, coords_array = dataset.generate_offline_test_batch(test_img_dir=dataset.test_img_dir, test_name=test_img_name, write_out=FLAGS.write_out)

        print("get offline test batch = ", img_base_name, " get patches no. = ", patch_array.shape)

        batch_size = patch_array.shape[0]

        ######################## train the model #####################################

        ops.reset_default_graph()
        session = get_session()
        #
        model = B_PATCHGAN_3D(x_dim=patch_size, batch_size=batch_size, J=FLAGS.J,
                              lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                              num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
                              num_classes=FLAGS.num_classes, channel=dataset.channel)

        load_model(session, checkpoint_dir)

        predicts_all, fusion = session.run([model.predicts, model.gen_fused], feed_dict={model.imgs: patch_array})
        predicts_all = np.asarray(predicts_all)

        print("predicts_all = ", predicts_all.shape)   # shape = (J, batch_size, h, w, d, num_classes)

        #####################################################################################################################

        for j in range(predicts_all.shape[0]):

            mask_np = merge_patches_back_multi(resampled_np=resampled_img_np, patch_arr=predicts_all[j, :, :, :, :, :], coords_arr=coords_array, patch_size=dataset.patch_size)

            mask = sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0)).astype(np.float32))
            mask.SetSpacing(dataset.new_spacing)
            mask = resampleMRIData(mask, new_spacing=test_origin_img.GetSpacing())
            mask.SetDirection(new_direction)
            mask.SetOrigin(test_origin_img.GetOrigin())

            sitk.WriteImage(mask, save_seg_dir + img_base_name + "_" + str(j) + "_seg.nii.gz")

        #######################################################################################################################

        mask_fused = merge_patches_back_multi(resampled_np=resampled_img_np, patch_arr=fusion, coords_arr=coords_array, patch_size=dataset.patch_size)

        mask_fused_img = sitk.GetImageFromArray(np.transpose(mask_fused, (2, 1, 0)).astype(np.float32))
        mask_fused_img.SetSpacing(dataset.new_spacing)
        mask_fused_img = resampleMRIData(mask_fused_img, new_spacing=test_origin_img.GetSpacing())
        mask_fused_img.SetDirection(new_direction)
        mask_fused_img.SetOrigin(test_origin_img.GetOrigin())

        sitk.WriteImage(mask_fused_img, save_seg_dir + img_base_name + '_fuse.nii.gz')

        session.close()


##########################################################################################################################


def test_segmentation_3d(dataset):

    session = get_session()
    tf.set_random_seed(FLAGS.random_seed)

    if FLAGS.extra_pooling:

        model = B_PATCHGAN_3D_EXTRA(x_dim=dataset.patch_size, batch_size=FLAGS.batch_size, J=FLAGS.J,
                                    lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                                    num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
                                    num_classes=FLAGS.num_classes, channel=dataset.channel)


    elif FLAGS.reduce_pooling:

        model = B_PATCHGAN_3D_REDUCE(x_dim=dataset.patch_size, batch_size=FLAGS.batch_size, J=FLAGS.J,
                                     lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                                     num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
                                     num_classes=FLAGS.num_classes, channel=dataset.channel)

    else:

        model = B_PATCHGAN_3D(x_dim=dataset.patch_size, batch_size=FLAGS.batch_size, J=FLAGS.J,
                              lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                              num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda,
                              num_classes=FLAGS.num_classes, channel=dataset.channel)

    model_dir = dataset.name + "_%s_%s_%s_%s" % ('bpgan3d', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    load_model(session, checkpoint_dir)

    ########################################################################################################################

    save_seg_dir = project_path + 'segmentation_results/' + model_dir + '/'

    for g in range(FLAGS.J):

        save_g_dir = save_seg_dir + '/generator_%d' % g + '/'
        print(save_g_dir)

        if not os.path.exists(save_g_dir):
            os.makedirs(save_g_dir)

    # select files from the list
    test_imgs_files = os.listdir(dataset.test_patch_dir)

    print("start testing .... ")

    # indx = 0
    # while indx <= len(test_imgs_files):
    #
    #     if indx <= len(test_imgs_files) - FLAGS.batch_size:
    #
    #         test_name_list = test_imgs_files[indx: indx + FLAGS.batch_size]
    #
    #     else:
    #
    #         # test_img_base_name + "(" + str(coords[0]) + '-' + str(coords[1]) + '-' + str(coords[2]) + ").nii.gz"
    #         test_name_list = test_imgs_files[len(test_imgs_files) - FLAGS.batch_size : len(test_imgs_files)]
    #
    #     # read image array
    #     test_imgs = dataset.get_test_batch(batch_img_files=test_name_list)
    #
    #     # pass the test_imgs to the model
    #     predicts_all, fusion = session.run([model.predicts, model.gen_fused], feed_dict={model.imgs: test_imgs})
    #     predicts_all = np.asarray(predicts_all)
    #
    #     #######################################################################################################################
    #
    #
    #     for b_indx in range(FLAGS.batch_size):
    #
    #         test_name = test_name_list[b_indx].split('.')[0]
    #         test_base_name = test_name.split('(')[0]
    #
    #         ## create the file folder path ##
    #
    #         for g in range(predicts_all.shape[0]):
    #
    #             save_file_dir = save_seg_dir + '/generator_%d' % g + '/' + test_base_name + '/'
    #             if not os.path.exists(save_file_dir):
    #                 os.makedirs(save_file_dir)
    #
    #             pred_g = np.argmax(predicts_all[g, b_indx, :, :, :, :], axis=-1).astype(np.uint8)
    #
    #             mask = sitk.GetImageFromArray(np.transpose(pred_g, (2, 1, 0)).astype(np.float32))
    #             sitk.WriteImage(mask, save_file_dir + test_name + "_seg.nii.gz")
    #
    #         save_fuse_dir = save_seg_dir + '/fused/' + test_base_name + '/'
    #         if not os.path.exists(save_fuse_dir):
    #             os.makedirs(save_fuse_dir)
    #
    #         fused = np.argmax(fusion[b_indx, :, :, :, :], axis=-1).astype(np.float64)
    #
    #         fused = sitk.GetImageFromArray(np.transpose(fused, (2, 1, 0)).astype(np.float32))
    #         sitk.WriteImage(fused, save_fuse_dir + test_name + "_fuse.nii.gz")
    #
    #     indx = indx + FLAGS.batch_size


    print("done ... complete writing all the segmented patches ... start merging ")

    for g in range(FLAGS.J):

        generator_dir = save_seg_dir + '/generator_%d' % g + '/'
        merge_segmented_patches(dataset, result_path=generator_dir)

    fused_dir = save_seg_dir + '/fused/'
    merge_segmented_patches(dataset, result_path=fused_dir)

    print(" done ")



def merge_segmented_patches(dataset, result_path):

    """
    merge the segmented patches of the fused folder, each generator folder
    :param dataset: this task
    :param model_dir: model name
    :param result_path: project_path + '/segmentation_results/' + model_dir + '/fused/' or project_path + '/segmentation_results/' + model_dir + 'g/'
    :return: the final segmentation map
    """

    result_dir = result_path + '/result/'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for file_name in os.listdir(result_path):

        if file_name.find("result") != -1:
            print("skip this folder")
            continue

        origin_img = readItkData(dataset.test_img_origin_dir + file_name + '.nii.gz')
        resampled_origin_img = resampleCTData(origin_img, new_spacing=dataset.new_spacing, is_label=False)

        resampled_np = np.transpose(sitk.GetArrayFromImage(resampled_origin_img), (2, 1, 0)).astype(np.float32)

        sub_folder_dir = result_path + file_name + '/'
        mask_np = merge_subfolder_to_segment(resampled_img_np=resampled_np, sub_folder_dir=sub_folder_dir)

        mask = sitk.GetImageFromArray(np.transpose(mask_np, (2, 1, 0)).astype(np.uint8))
        mask.SetSpacing(dataset.new_spacing)
        mask = resampleCTData(mask, new_spacing=origin_img.GetSpacing(), is_label=False)

        mask.SetDirection(origin_img.GetDirection())
        mask.SetOrigin(origin_img.GetOrigin())

        sitk.WriteImage(mask, result_dir + file_name + ".nii.gz")


#######################################################################################################


if __name__ == "__main__":


    if FLAGS.task == 'Pancreas':

        patch_size = [96, 80, 64]  # [128, 64, 64]  # otherwise [64, 64, 64]
        resampled_spacing = (1.5, 1.5, 2.5)   # average spacing
        FLAGS.num_classes = 3
        channel = 1
        pan_path = os.path.join(FLAGS.data_path, "Task07_Pancreas")
        dataset = Pancreas_3D(pan_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, num_bins=256)

    elif FLAGS.task == 'Spleen':

        patch_size = [96, 96, 64]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 2
        FLAGS.batch_size = 2
        FLAGS.J = 3
        FLAGS.extra_pooling = True
        spleen_path = os.path.join(FLAGS.data_path, "Task09_Spleen")
        dataset = Spleen_3D(spleen_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    elif FLAGS.task == 'Prostate':

        patch_size = [96, 96, 32]
        resampled_spacing = (1.5, 1.5, 1.0)
        FLAGS.num_classes = 3
        FLAGS.batch_size = 3
        FLAGS.J = 5
        channel = 2
        prostate_path = os.path.join(FLAGS.data_path, "Task05_Prostate")
        dataset = Prostate_3D(prostate_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, channel=channel)


    elif FLAGS.task == 'BrainTumor':

        patch_size = [64, 96, 64]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 4
        FLAGS.batch_size = 2
        FLAGS.J = 5
        FLAGS.num_classes = 4
        brain_path = os.path.join(FLAGS.data_path, "Task01_BrainTumour")
        dataset = BrainTumor_3D(brain_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, channel=channel)


    elif FLAGS.task == 'Heart':

        patch_size = [64, 64, 96]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 2
        FLAGS.batch_size = 2
        FLAGS.J = 5
        heart_path = os.path.join(FLAGS.data_path, "Task02_Heart")
        dataset = Heart_3D(heart_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, num_bins=256)


    elif FLAGS.task == 'Liver':   # this must be train in patch !

        patch_size = [64, 64, 64]
        resampled_spacing = (2.0, 2.0, 2.0)
        channel = 1
        FLAGS.num_classes = 3
        FLAGS.batch_size = 4
        FLAGS.J = 4
        FLAGS.reduce_pooling = True
        liver_path = os.path.join(FLAGS.data_path, "Task03_Liver")
        dataset = Liver_3D(liver_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    elif FLAGS.task == 'Hippocampus':

        patch_size = [32, 48, 32]
        resampled_spacing = (1.0, 1.0, 1.0)
        FLAGS.num_classes = 3
        channel = 1
        FLAGS.batch_size = 8
        FLAGS.J = 7
        hippo_path = os.path.join(FLAGS.data_path, "Task04_Hippocampus")
        dataset = Hippocampus_3D(hippo_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, num_bins=256)


    elif FLAGS.task == 'Lung':

        patch_size = [64, 64, 64]
        resampled_spacing = (1.5, 1.5, 1.5)   # should testing
        FLAGS.num_classes = 2
        channel = 1
        FLAGS.batch_size = 7
        FLAGS.J = 2
        lung_path = os.path.join(FLAGS.data_path, "Task06_Lung")
        dataset = Lung_3D(lung_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    elif FLAGS.task == 'HepaticVessel':   # this must be train in patch !

        patch_size = [128, 96, 80]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 3
        vessel_path = os.path.join(FLAGS.data_path, "Task08_HepaticVessel")
        dataset = HepaticVessel_3D(vessel_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    elif FLAGS.task == 'Colon':   # this must be train in patch !

        patch_size = [64, 64, 64]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 3
        colon_path = os.path.join(FLAGS.data_path, "Task10_Colon")
        dataset = Colon_3D(colon_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    else:

        print("select a task!")


##########################################################################################################################################

    if FLAGS.phase == 'augmentation':
        augmentation(dataset, FLAGS)

    elif FLAGS.phase == 'test' and (FLAGS.task == 'BrainTumor' or FLAGS.task == 'Prostate'):
        test_multi_mri(dataset, FLAGS, patch_size=patch_size)

    elif FLAGS.phase == 'test' and (FLAGS.task == 'Heart' or FLAGS.task == 'Hippocampus'):
        test_mri(dataset, FLAGS, patch_size=patch_size)

    elif FLAGS.phase == 'test' and (FLAGS.task == 'Liver' or FLAGS.task == 'Spleen' or FLAGS.task == 'Pancreas'):
        test_segmentation_3d(dataset)

    else:
        test_ct(dataset, FLAGS, patch_size=patch_size)



