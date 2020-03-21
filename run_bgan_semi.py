#!/usr/bin/env python

import os, sys, argparse, json, time, shutil
import numpy as np
from math import ceil

import SimpleITK as sitk
import tensorflow as tf
import cv2
from tensorflow.contrib import slim
from sklearn.metrics import jaccard_similarity_score


from bgan_util import *
from b_patchGan import B_PATCHGAN
from seg_utils import *
from utilities import *


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0, 1, 2, 3"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

project_path = '/home/jtma/PycharmProjects/'

flags = tf.app.flags
flags.DEFINE_integer("gf_dim", 32, "num of gen features")  # change it to 32
flags.DEFINE_integer("df_dim", 64, "num of disc features")   # default 96
flags.DEFINE_string("data_path", project_path + "dataset", "path to where the datasets live")
flags.DEFINE_string("dataset", "Task07_Pancreas", "datasets name cifar etc.")   # cifar | histopatho | mnist | abdomenCT
flags.DEFINE_integer("batch_size", 6, "mini batch size")
flags.DEFINE_integer("num_aug", 0, 'number of augmented data for each datasets in supervised training branch')
flags.DEFINE_float("prior_std", 1.0, "NN weight prior std")
flags.DEFINE_integer("J", 4, "number of samples of z / generators")
flags.DEFINE_integer("J_d", 1, "number of discriminator weight samples")
flags.DEFINE_integer("M", 1, "number of MCMC NN weight samples per z")
flags.DEFINE_integer("train_iter", 80000, "number of training iterations")
flags.DEFINE_integer("random_seed", 2222, "random seed")
flags.DEFINE_float("lr", 0.0005, "learning rate")
flags.DEFINE_float("lr_decay", 3.0, "learning rate")
flags.DEFINE_string("optimizer", "adam", "optimizer -- adam or sgd")
flags.DEFINE_boolean("ml", False, "if specified, disable bayesian things")
flags.DEFINE_boolean("restore", False, "if specified, disable bayesian things")
flags.DEFINE_string("fn", "f1", "number of the fold of the cross validation")
flags.DEFINE_boolean("da", False, "whether to do data augmentation00")
flags.DEFINE_integer("img_size", 128, "default image width / length")
flags.DEFINE_boolean("wassertein", True, "wasserstein GAN")
flags.DEFINE_float("l1_lambda", 1, "l1_lambda of the l1 norm for segmentation")  # 10 works well
flags.DEFINE_string('checkpoint_dir', project_path + 'checkpoint', 'models are saved here')
flags.DEFINE_integer('num_classes', 1, "number of classes for segmentation")
flags.DEFINE_integer('num_crop', 4, "number of cropped patches for training, negative samples")
flags.DEFINE_string('phase', 'train', "train or test")
FLAGS = flags.FLAGS

########################################################################################################################


def generate_2d_slices():

    # def GenerateSlices(img_dir, label_dir, save_img_path, save_label_path, num_pixels, thres=True):

    dataset_path = '/home/jtma/PycharmProjects/dataset/Task07_Pancreas/'
    # img_dir = dataset_path + 'imagesTs_cropped/'
    # label_dir = dataset_path + 'labelsTr/'
    #
    # save_img_path = dataset_path + 'imagesTs_cropped_2d/'
    # save_label_path = dataset_path + 'labelsTr_2d/'
    #
    # num_pixels = 0
    # GenerateSlices(img_dir=img_dir, label_dir=label_dir,
    #                save_img_path=save_img_path, save_label_path=save_label_path,
    #                num_pixels=num_pixels, thres=[50, 500])

    img_dir = dataset_path + 'imagesTs/'
    save_img_path = dataset_path + 'imagesTs_2d/'

    Generate_negative_slices(img_dir=img_dir, save_img_path=save_img_path, thres=[50, 500])

    print("2d-slices-generating-done")


def submit_spleen():

    x_dim = dataset.x_dim

    #test_data_dir = FLAGS.data_path + '/Task09_Spleen/'
    test_data_dir = FLAGS.data_path + '/Task07_Pancreas/imagesTs_2d/'

    session = get_session()
    tf.set_random_seed(FLAGS.random_seed)

    model = B_PATCHGAN(x_dim, batch_size=FLAGS.batch_size, J=FLAGS.J,  # FLAGS.batch_size
                       lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                       num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda, num_classes=FLAGS.num_classes)

    model_dir = dataset.name + "%s_%s_%s_%s" % ('bpgan', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn  # FLAGS.batch_size
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    show_all_variables()

    load_model(session, model_dir, checkpoint_dir)

    ########################################################################################################################

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/'
        print(save_results_dir)

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    # select files from the list
    test_imgs_files = os.listdir(test_data_dir)

    indx = 0
    while indx <= len(test_imgs_files):

        if indx <= len(test_imgs_files) - FLAGS.batch_size:

            test_name_list = test_imgs_files[indx: indx + FLAGS.batch_size]

        else:

            test_name_list = test_imgs_files[len(test_imgs_files) - FLAGS.batch_size: len(test_imgs_files)]

        test_imgs, test_crop_store = get_test_img_unlabel(test_img_dir=test_data_dir, batch_img_files=test_name_list, output_size=FLAGS.img_size)

        predicts_all, fusion = session.run([model.predicts, model.gen_fused], feed_dict={model.imgs: test_imgs})
        predicts_all = np.asarray(predicts_all)

        for b_indx in range(FLAGS.batch_size):

            test_name = test_name_list[b_indx].split('.')[0]
            sub_folder = test_name[0: len(test_name) - 4]

            fuse_b = fusion[b_indx, :, :, :]
            fuse_b = cv2.resize(fuse_b, (fuse_b.shape[0] * 4, fuse_b.shape[1] * 4), interpolation=cv2.INTER_CUBIC)
            print(fuse_b.shape)

            crop_coord_b = test_crop_store[b_indx]
            print("crop+coord_b = ", crop_coord_b[1] - crop_coord_b[0], crop_coord_b[3] - crop_coord_b[2])
            empty_slice_fuse = np.zeros((512, 512), np.float32)
            empty_slice_fuse[crop_coord_b[0]:crop_coord_b[1], crop_coord_b[2]:crop_coord_b[3]] = \
                fuse_b[:crop_coord_b[1] - crop_coord_b[0], :crop_coord_b[3] - crop_coord_b[2]]

            fuse_save_path = project_path + 'segmentation_results/' + model_dir + '/fused/' + sub_folder + '/'
            if not os.path.exists(fuse_save_path):
                os.makedirs(fuse_save_path)

            cv2.imwrite(fuse_save_path + test_name + '_fuse.png', empty_slice_fuse)

            for g in range(predicts_all.shape[0]):

                move_dir = project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + sub_folder + '/'
                if not os.path.exists(move_dir):
                    os.makedirs(move_dir)

                pred_b_g = np.round(predicts_all[g, b_indx, :, :, :])
                pred_b_g = cv2.resize(pred_b_g, (pred_b_g.shape[0] * 4, pred_b_g.shape[1] * 4), interpolation=cv2.INTER_CUBIC)

                empty_slice_pred = np.zeros((512, 512), np.float32)
                empty_slice_pred[crop_coord_b[0]:crop_coord_b[1], crop_coord_b[2]:crop_coord_b[3]] = \
                    pred_b_g[:crop_coord_b[1] - crop_coord_b[0], :crop_coord_b[3] - crop_coord_b[2]]

                cv2.imwrite(move_dir + test_name + '_seg.png', empty_slice_pred)

            #############################################################################################################################################################################################################################

        indx = indx + FLAGS.batch_size


    ############# merge all the segmented slices #####################

    fused_folder = '/home/jtma/PycharmProjects/segmentation_results/' + model_dir + '/fused/'
    submit_dir = '/home/jtma/PycharmProjects/segmentation_results/' + model_dir + '/submit/'

    for sub_folder in os.listdir(fused_folder):

        merge_to_3d_volume(slice_dir=fused_folder + sub_folder, origin_img_dir=test_data_dir,
                           save_dir=submit_dir, img_base_name=sub_folder, binary=True, strip='_fuse')



    for g in range(FLAGS.J):

        generator_dir = project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/'
        saved_g_dir = project_path + 'segmentation_results/' + model_dir + '/submit_%d' %g + '/'

        for sub_folder in os.listdir(generator_dir):

            merge_to_3d_volume(slice_dir=generator_dir + sub_folder, origin_img_dir=test_data_dir,
                               save_dir=saved_g_dir, img_base_name=sub_folder, binary=True, strip='_seg')



    print("submit_spleen_done")


def prepare_for_submit():

    results_dir = '/home/jtma/PycharmProjects/segmentation_results/'
    model_name = "%s_%s_%s_%s" % ('bpgan', 16, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    generator = 'generator_1'

    origin_img_dir = '/home/jtma/PycharmProjects/dataset/Task07_Pancreas/all_3ds/ALL_IMG_PAN/'
    save_dir = '/home/jtma/PycharmProjects/segmentation_results/' + model_name + '/segments/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for img_folder in os.listdir(results_dir + model_name + '/' + generator + '/'):

        slice_dir = results_dir + model_name + '/' + generator + '/' + img_folder + '/merge/'

        merge_to_3d_volume(slice_dir=slice_dir, origin_img_dir=origin_img_dir, save_dir=save_dir, img_base_name=img_folder)

    print("merging done ... ")


def test_segmentation(dataset):

    x_dim = dataset.x_dim
    img_dim = (512, 512, 1)

    test_data_dir = FLAGS.data_path + '/Task07_Pancreas/imagesTs_2d/'

    session = get_session()
    tf.set_random_seed(FLAGS.random_seed)

    model = B_PATCHGAN(x_dim, batch_size=FLAGS.batch_size * (FLAGS.num_crop + 1), J=FLAGS.J,    # FLAGS.batch_size
                       lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                       num_train=dataset.num_train, l1lambda=FLAGS.l1_lambda, num_classes=FLAGS.num_classes)

    model_dir = "%s_%s_%s_%s" % ('bpgan', 16, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn   # FLAGS.batch_size
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    show_all_variables()

    load_model(session, model_dir, checkpoint_dir)

########################################################################################################################

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/'
        print(save_results_dir)

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    # select files from the list
    test_imgs_files = os.listdir(test_data_dir)

    total_mean = []
    total_mean_best = []
    total_mean_best_g = []
    total_fused = []

    indx = 0
    while indx <= len(test_imgs_files):

        if indx <= len(test_imgs_files) - FLAGS.batch_size:

            test_name_list = test_imgs_files[indx: indx + FLAGS.batch_size]

        else:

            test_name_list = test_imgs_files[len(test_imgs_files) - FLAGS.batch_size: len(test_imgs_files)]

        # read image array
        test_imgs, test_labels, test_crop_store = get_test_batch_offline(test_data_dir + 'img/', test_data_dir + 'label/', test_name_list, output_size=FLAGS.img_size, crop_num=FLAGS.num_crop)
        test_imgs_origin, test_labels_origin = get_test_img_label_offline(test_data_dir + 'img/', test_data_dir + 'label/', test_name_list, img_size=img_dim)

 ############################################################################################################################

    #     mc_dropout_iters = 50
    #
    #     best_mean_mc = []
    #     for i in range(mc_dropout_iters):
    #
    #         predicts_all = session.run(model.predicts, feed_dict={model.imgs: test_imgs})
    #
    #         # save the predicts with respect to generator
    #         predicts_all = np.asarray(predicts_all)
    #
    #         dice_batch = []
    #         best_dice = []
    #         for b_indx in range(FLAGS.batch_size):
    #
    #             best_dice_i = []
    #
    #             test_name = test_name_list[b_indx].split('.')[0]
    #             #if test_name.find("pancreas_") < 0:   # only calculate the NIH datasets
    #
    #             for g in range(predicts_all.shape[0]):
    #
    #                 pred_g = predicts_all[g, b_indx, :, :, :]
    #
    #                 dice_batch.append(dice_coefficient(test_labels[b_indx], pred_g))
    #                 best_dice_i.append(dice_coefficient(test_labels[b_indx], pred_g))
    #
    #             best_dice.append(np.max(best_dice_i))
    #
    #         dice_batch_mean = np.mean(dice_batch)  # np.sum(dice_batch) / (FLAGS.batch_size * predicts_all.shape[0])
    #         best_mean_mc.append(best_dice)
    #         print('mc - ', i, ' patch = ', int(indx / FLAGS.batch_size), ' mean dice = ', dice_batch_mean, ', best dice = ', np.mean(best_dice))
    #
    #     indx = indx + FLAGS.batch_size
    #
    #     total_mean.append(dice_batch_mean)
    #     total_mean_best.append(np.mean(best_mean_mc))  # previously - best_dice
    #
    # print("total_mean = ", np.mean(total_mean), np.mean(total_mean_best))


########################################################################################################################

        # pass the test_imgs to the model
        predicts_all, maps_all = session.run([model.predicts, model.gen_fused], feed_dict={model.imgs: test_imgs})

        # save the predicts with respect to generator
        predicts_all = np.asarray(predicts_all)

        for b_indx in range(FLAGS.batch_size):

            test_name = test_name_list[b_indx].split('.')[0]
            sub_folder = test_name[0: len(test_name) - 4]

            batch_dice = []
            batch_tumor_dice = []
            batch_fused_dice = []

            best_batch_dice = 0
            best_batch_dice_g = 0

            b_test_img_origin = test_imgs_origin[b_indx]
            b_test_label_origin = test_labels_origin[b_indx]

            for g in range(predicts_all.shape[0]):

                move_dir = project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + sub_folder + '/'
                if not os.path.exists(move_dir):
                    os.makedirs(move_dir)

                merge_dir = project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + sub_folder + '/merge/'
                if not os.path.exists(merge_dir):
                    os.makedirs(merge_dir)

                b_g_seg = np.zeros((img_dim[0], img_dim[1]), dtype=np.float32)
                b_g_label = np.zeros((img_dim[0], img_dim[1]), dtype=np.float32)
                b_g_fused = np.zeros((img_dim[0], img_dim[1]), dtype=np.float32)

                for crop_i in range(FLAGS.num_crop + 1):

                    test_name_write = test_name + "_" + str(crop_i) + "_" + str(test_crop_store[b_indx * (FLAGS.num_crop + 1) + crop_i][0]) + "-" + str(test_crop_store[b_indx * (FLAGS.num_crop + 1) + crop_i][1])
                    start_x_y = test_crop_store[b_indx * (FLAGS.num_crop + 1) + crop_i]

                    path_tumor = ""
                    if np.max(test_labels[b_indx * (FLAGS.num_crop + 1)]) == 2:
                        path_tumor += '_tumor'

                    pred_g = np.round(predicts_all[g, b_indx * (FLAGS.num_crop + 1) + crop_i, :, :, :])

                    ##################################### merge back ###################################################

                    empty_slice_pred = np.zeros((img_dim[0] // 2, img_dim[1] // 2, img_dim[2]))
                    empty_slice_pred[int(start_x_y[1]): int(start_x_y[1] + pred_g.shape[1]), int(start_x_y[0]): int(start_x_y[0] + pred_g.shape[1]), :] = pred_g
                    empty_slice_pred = cv2.resize(empty_slice_pred, (img_dim[0], img_dim[1]), interpolation=cv2.INTER_CUBIC)

                    empty_slice_gt = np.zeros((img_dim[0] // 2, img_dim[1] // 2, img_dim[2]))
                    empty_slice_gt[int(start_x_y[1]): int(start_x_y[1] + pred_g.shape[1]), int(start_x_y[0]): int(start_x_y[0] + pred_g.shape[1]), :] = test_labels[b_indx * (FLAGS.num_crop + 1) + crop_i]
                    empty_slice_gt = cv2.resize(empty_slice_gt, (img_dim[0], img_dim[1]), interpolation=cv2.INTER_CUBIC)

                    empty_slice_fuse = np.zeros((img_dim[0] // 2, img_dim[1] // 2, img_dim[2]))
                    empty_slice_fuse[int(start_x_y[1]): int(start_x_y[1] + pred_g.shape[1]), int(start_x_y[0]): int(start_x_y[0] + pred_g.shape[0]), :] = maps_all[b_indx * (FLAGS.num_crop + 1) + crop_i, :, :, :]
                    empty_slice_fuse = cv2.resize(empty_slice_fuse, (img_dim[0], img_dim[1]), interpolation=cv2.INTER_CUBIC)

                    b_g_seg = np.add(b_g_seg, empty_slice_pred)
                    b_g_label = np.add(b_g_label, empty_slice_gt)
                    b_g_fused = np.add(b_g_fused, empty_slice_fuse)

                    ####################################################################################################

                    cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name_write + path_tumor + '_seg.png', empty_slice_pred)  # pred_g
                    cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name_write + path_tumor + '_label.png', empty_slice_gt)  # test_labels[b_indx + crop_i]
                    cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name_write + path_tumor + '_fuse.png', empty_slice_fuse)  # test_labels[b_indx + crop_i]

                    shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name_write + path_tumor + '_seg.png', move_dir + test_name_write + path_tumor + '_seg.png')
                    shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name_write + path_tumor + '_label.png', move_dir + test_name_write + path_tumor + '_label.png')
                    shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name_write + path_tumor + '_fuse.png', move_dir + test_name_write + path_tumor + '_fuse.png')


                ########## measures for this generator for this batch ########################

                b_g_seg[b_g_seg > 1] = 1
                b_g_fused[b_g_fused > 1] = 1

                dice_b_g = dice_coefficient(b_test_label_origin, b_g_seg)   # dice_coefficient
                dice_b_g_tumor = dice_coef_tumor(b_test_label_origin, b_g_seg)

                dice_fused_b = dice_coefficient(b_test_label_origin, b_g_fused)
                batch_fused_dice.append(dice_fused_b)

                if dice_b_g > 0: batch_dice.append(dice_b_g)
                batch_tumor_dice.append(dice_b_g_tumor)

                if dice_b_g > best_batch_dice:
                    best_batch_dice = dice_b_g
                    best_batch_dice_g = g

            ############################################################################################################################################################################################################################

                cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_img.png', (b_test_img_origin + 1.0) * 127.5)
                cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_mask.png', b_test_label_origin)  # variance_g
                cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_seg.png', b_g_seg)
                cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_label.png', b_g_label)  # variance_g
                cv2.imwrite(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + '_fuse.png', b_g_fused)

                shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_img.png', move_dir + test_name + path_tumor + '_img.png')
                shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_mask.png', move_dir + test_name + path_tumor + '_mask.png')
                shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_seg.png', move_dir + test_name + path_tumor + '_seg.png')
                shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + path_tumor + '_label.png', move_dir + test_name + path_tumor + '_label.png')
                shutil.move(project_path + 'segmentation_results/' + model_dir + '/generator_%d' % g + '/' + test_name + '_fuse.png', merge_dir + test_name + '_fuse.png')

            #############################################################################################################################################################################################################################

        indx = indx + FLAGS.batch_size

        print('batch ', indx, int(indx / FLAGS.batch_size), ' mean dice = ', np.mean(batch_dice), ', best dice = ', best_batch_dice, ' best generator = ', best_batch_dice_g, " mean fused = ", np.mean(batch_fused_dice))

        if len(batch_dice) > 0: total_mean.append(np.mean(batch_dice))
        if len(batch_fused_dice) > 0: total_fused.append(np.mean(batch_fused_dice))
        total_mean_best.append(best_batch_dice)
        total_mean_best_g.append(best_batch_dice_g)

    print('total dice mean = ', np.mean(total_mean), ', total mean best = ', np.mean(total_mean_best), ', total best generator = ', np.mean(total_mean_best_g), ", total fused dice = ", np.mean(total_fused))

    ######################################################################################################################################################################


########################################################################################################################


def get_val_accuracy_bpgan_fused(session, dcgan, val_imgs, val_labels, model_name, batch_test_files, num_crop):

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/'

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    predict, fused, weighted_maps = session.run([dcgan.predicts, dcgan.gen_fused, dcgan.weighted_maps], feed_dict={dcgan.imgs: val_imgs})

    predict_arr = np.asarray(predict)
    map_arr = np.asarray(weighted_maps)

    dice_gen_mean = []
    dice_tumor_mean = []
    dice_map_mean = []

    for g in range(predict_arr.shape[0]):
        dice_g, dice_g_tumor, dice_fused = [], [], []

        for b_indx in range(FLAGS.batch_size):

            file_name_base = batch_test_files[b_indx].split('.')[0]

            for crop_i in range(num_crop + 1):

                seg = predict_arr[g, b_indx * (num_crop + 1) + crop_i, :, :, :]
                seg = np.round(seg)
                weighted_map_g = map_arr[g, b_indx * (num_crop + 1) + crop_i, :, :, :] * 100
                fused_ = fused[b_indx * (num_crop + 1) + crop_i, :, :, :]

                dice_g.append(dice_coefficient(seg, val_labels[b_indx * (num_crop + 1) + crop_i]))
                dice_fused.append(dice_coefficient(fused_, val_labels[b_indx * (num_crop + 1) + crop_i]))

                if dice_coef_tumor(seg, val_labels[b_indx * (num_crop + 1) + crop_i]) >= 0:
                    dice_g_tumor.append(dice_coef_tumor(seg, val_labels[b_indx * (num_crop + 1) + crop_i]))

                path_tumor = ""
                if np.max(val_labels[b_indx * (num_crop + 1) + crop_i]) == 2:
                    path_tumor += '_tumor'

                ########################################################################################################

                cv2.imwrite(project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + path_tumor + '_seg.png', seg)
                cv2.imwrite(project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + path_tumor + '_map.png', weighted_map_g)
                cv2.imwrite(project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + path_tumor + '_fuse.png', fused_)
                cv2.imwrite(project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + path_tumor + '_label.png', val_labels[b_indx * (num_crop + 1) + crop_i])  # cv2.cvtColor(val_labels[b],cv2.COLOR_BGR2GRAY)

                #########################################################################################################

        dice_gen_mean.append(np.mean(dice_g))
        dice_tumor_mean.append(np.mean(dice_g_tumor))
        dice_map_mean.append(np.mean(dice_fused))

    return dice_gen_mean, dice_tumor_mean, dice_map_mean


def get_val_accuracy_bpgan(session, dcgan, val_imgs, val_labels, model_name, batch_test_files, num_crop=FLAGS.num_crop):

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'segmentation_results/' + model_name + '/generator_%d' % g + '/'

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    predict, map = session.run([dcgan.predicts, dcgan.gen_fused], feed_dict={dcgan.imgs: val_imgs})  # [, dcgan.variances ]
    predict_arr = np.asarray(predict)
#    variance_arr = np.asarray(variances)

    dice_gen_mean = []
    dice_tumor_mean = []
    dice_map_mean = []

    for g in range(predict_arr.shape[0]):

        dice_g, dice_g_tumor = [], []

        for b_indx in range(FLAGS.batch_size):

            file_name_base = batch_test_files[b_indx].split('.')[0]

            for crop_i in range(num_crop + 1):

                seg = predict_arr[g, b_indx * (num_crop + 1) + crop_i, :, :, :]
                seg = np.round(seg)
                map_ = map[b_indx * (num_crop + 1) + crop_i, :, :, :]
#                var = np.exp(variance_arr[g, b_indx * (num_crop + 1) + crop_i, :, :, :])

                dice_g.append(dice_coefficient(seg, val_labels[b_indx * (num_crop + 1) + crop_i]))
                dice_map_mean.append(dice_coefficient(map_, val_labels[b_indx * (num_crop + 1) + crop_i]))

                if dice_coef_tumor(seg, val_labels[b_indx * (num_crop + 1) + crop_i]) >= 0:
                    dice_g_tumor.append(dice_coef_tumor(seg, val_labels[b_indx * (num_crop + 1) + crop_i]))

                path_tumor = ""
                if np.max(val_labels[b_indx * (num_crop + 1) + crop_i]) == 2:
                    path_tumor += '_tumor'

                ##############################################################################

                cv2.imwrite(project_path + 'segmentation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + '_seg.png', seg)
                cv2.imwrite(project_path + 'segmentation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + '_fused.png', map_)
                cv2.imwrite(project_path + 'segmentation_results/' + model_name + '/generator_%d' % g + '/' + file_name_base + path_tumor + '_label.png', val_labels[b_indx * (num_crop + 1) + crop_i])  # cv2.cvtColor(val_labels[b],cv2.COLOR_BGR2GRAY)

        dice_gen_mean.append(np.mean(dice_g))
        dice_tumor_mean.append(np.mean(dice_g_tumor))

    print("dice_map_mean = ", np.mean(dice_map_mean))

    return dice_gen_mean, dice_tumor_mean


def b_patchGan(dataset, FLAGS):

    model_base_name = dataset.name + 'bpgan' + '_%d' % FLAGS.gf_dim + '_%d' % FLAGS.df_dim + '_%d' % FLAGS.J + '_' + FLAGS.fn

    # save model weights
    model_dir = dataset.name + "%s_%s_%s_%s" % ('bpgan', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    x_dim = dataset.x_dim
    n_classes = FLAGS.num_classes

    train_img_files = dataset.train_img_files
    test_img_files = dataset.test_img_files

    num_train = dataset.num_train
    num_test = dataset.num_test

    session = get_session()
    tf.set_random_seed(FLAGS.random_seed)


    with tf.device("/gpu:3"):

        bpgan = B_PATCHGAN(x_dim, batch_size=FLAGS.batch_size * (1 + FLAGS.num_crop), J=FLAGS.J,    # FLAGS.num_crop
                                    lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J==1), prior_std=10,
                                    num_train=num_train, l1lambda=FLAGS.l1_lambda, num_classes=n_classes)

        show_all_variables()


    ################################################################################################

    print("Starting session")

    session.run(tf.global_variables_initializer())

#    train_writer = tf.summary.FileWriter("log_" + model_base_name)
#    train_writer.add_graph(session.graph)

    if FLAGS.restore:

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint(checkpoint_dir))  # search for checkpoint file

    # tf.get_default_graph()

    print("Starting training loop")
    optimizer_dict = {"disc": bpgan.d_optims_adam, "gen": bpgan.g_optims_adam, "distrib": bpgan.dist_optims_adam}

    base_learning_rate = FLAGS.lr     # for now we use same learning rate for Ds and Gs
    lr_decay_rate = FLAGS.lr_decay

    for train_iter in range(FLAGS.train_iter):

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate * min(1.0, (train_iter * FLAGS.batch_size)/float(num_train)))

    ### get supervised_train_batch ###

        train_indices = np.random.choice(num_train, FLAGS.batch_size, replace=True)

        batch_train_files = []
        for _, idx in enumerate(train_indices):
            batch_train_files.append(train_img_files[idx])

        labeled_image_batch, labels = dataset.get_train_batch(train_img_dir=dataset.train_img_dir, train_label_dir=dataset.train_label_dir,
                                                              batch_img_files=batch_train_files, output_size=FLAGS.img_size,
                                                              num_aug=FLAGS.num_aug, crop_num=FLAGS.num_crop)


    ### compute disc losses for with supervised patches

#        merge = tf.summary.merge_all()

        # summary, \   [merge, optimization_dict["disc"] + bpgan.d_losses]
        disc_info = session.run(optimizer_dict["disc"] + bpgan.d_losses,  # + bpgan.d_real_loss_pos + bpgan.d_fake_loss_pos, # + bpgan.d_real_loss_neg + bpgan.d_fake_loss_neg,
                                feed_dict={bpgan.imgs: labeled_image_batch,
                                           bpgan.labels: labels,
                                           bpgan.d_semi_learning_rate: learning_rate})

 #       train_writer.add_summary(summary, train_iter)


        d_losses = [d_ for d_ in disc_info if d_ is not None]   # len(disc_info) = 4

    ## compute generative losses

        gen_info = session.run(optimizer_dict["gen"] + bpgan.g_losses + bpgan.g_seg_losses, # + bpgan.g_aleatoric_losses,
                               feed_dict={bpgan.imgs: labeled_image_batch,
                                          bpgan.labels: labels,
                                          bpgan.g_learning_rate: learning_rate})


        g_losses = [g_ for g_ in gen_info if g_ is not None]   # len(gen_info) = 40 # , g_fake_losses, g_seg_losses


        distrib_info = session.run(optimizer_dict["distrib"] + bpgan.distrib_loss,
                                   feed_dict={bpgan.imgs: labeled_image_batch,
                                              bpgan.labels: labels,
                                              bpgan.g_learning_rate: learning_rate})

        distrib_losses = [dist_ for dist_ in distrib_info if dist_ is not None]


        if train_iter % 20 == 0:

            print(" Iter %i" % train_iter,
                  " : Disc losses = %s" % (", ".join(["%.4f" % dl for dl in d_losses])),
                  " , Gen losses = %s" % (", ".join(["%.4f" % gl for gl in g_losses])),
                  " : Distrib losses = %s" % (", ".join(["%.4f" % dl for dl in distrib_losses])),
                  " : disc_loss (mean) = %.4f " % np.mean(d_losses),
                  " : gen_loss (mean) = %.4f" % np.mean(g_losses))


        with tf.device("/cpu:1"):

            if train_iter % 100 == 0 and train_iter > 1:

                test_indices = np.random.choice(num_test, FLAGS.batch_size, replace=True)

                batch_test_files = []
                for _, idx in enumerate(test_indices):
                    batch_test_files.append(test_img_files[idx])

                test_imgs, test_labels = dataset.get_test_batch(test_img_dir=dataset.test_img_dir,
                                                                test_label_dir=dataset.test_label_dir,
                                                                batch_img_files=batch_test_files, output_size=FLAGS.img_size, crop_num=FLAGS.num_crop)

                dice_mean_list, dice_tumor_mean_list, dice_fused_mean_list = get_val_accuracy_bpgan_fused(session, bpgan, test_imgs, test_labels, model_dir, batch_test_files=batch_test_files, num_crop=FLAGS.num_crop) #test_imgs, test_labels, batch_test_files

                print(" Iter %i" % train_iter,
                      " : dice means = %s" % (", ".join(["%.4f" % dm for dm in dice_mean_list])),
                      " : dice tumor mean = %s" % (", ".join(["%.4f" % dm for dm in dice_tumor_mean_list])),
                      " : dice fused = %s" % (", ".join(["%.4f" % dm for dm in dice_fused_mean_list])))

            if train_iter % 1000 == 0 and train_iter > 1:

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                saver = tf.train.Saver(max_to_keep=2, keep_checkpoint_every_n_hours=2)
                saver.save(session, os.path.join(checkpoint_dir, model_base_name), global_step=train_iter, write_meta_graph=False)


    print("bpgan training done")


if __name__ == "__main__":

#    generate_2d_slices()

    # set seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    pan_path = os.path.join(FLAGS.data_path, "Task07_Pancreas")
    dataset = PanCT(pan_path, fn=FLAGS.fn, img_size=FLAGS.img_size)

    # spleen_path = os.path.join(FLAGS.data_path, "Task09_Spleen")
    # dataset = SpleenCT(spleen_path, img_size=FLAGS.img_size, val_ratio=0)

    # prostate_path = os.path.join(FLAGS.data_path, "Task05_Prostate")
    # dataset = ProstCT(prostate_path, img_size=FLAGS.img_size, val_ratio=0)

    if FLAGS.phase == 'train':
        b_patchGan(dataset, FLAGS)
    else:
        submit_spleen()   # test_segmentation(dataset)
