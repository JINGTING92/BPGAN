#!/usr/bin/env python

import os, sys, argparse, json, time, shutil
import numpy as np
from math import ceil
from random import shuffle

import SimpleITK as sitk
import tensorflow as tf
import cv2
from tensorflow.contrib import slim
from sklearn.metrics import jaccard_similarity_score, f1_score

from bgan_util_3d import *
from unet import UNet
from bpgan_3d import B_PATCHGAN_3D
from bpgan_3d_reduced import B_PATCHGAN_3D_REDUCE
from bpgan_3d_extra import B_PATCHGAN_3D_EXTRA
from pgan_3d import PATCHGAN_3D
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
flags.DEFINE_float("lr", 1e-4, "learning rate")
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
flags.DEFINE_boolean('extra_pooling', False, 'whether to active the fifth pooling layer')
flags.DEFINE_boolean('reduce_pooling', False, "3 pooling")
flags.DEFINE_boolean('pgan', False, 'use pgan without fused layer')
flags.DEFINE_float('val_ratio', 0.1, 'the validation ratio; if 0, full size training')
FLAGS = flags.FLAGS


def get_val_accuracy_bpgan_fusion(session, model, val_imgs, val_labels, model_name, batch_val_files):

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/'

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    predicts, fusion, epistemic_var = session.run([model.predicts, model.gen_fused, model.epistemic_var], feed_dict={model.imgs: val_imgs})
    predicts_arr = np.asarray(predicts)

    dice_total_mean, dice_total_fused_mean = [], []

    for g in range(predicts_arr.shape[0]):

        dice_g_mean, dice_g_fuse_mean = [], []

        for b in range(FLAGS.batch_size):

            val_img_base_name = batch_val_files[b].split(".")[0]

            pred_b_g = predicts_arr[g, b, :, :, :, :]    # shape = (d, w, h, channel)   # b * (num_crop + 1)
            label_b_g = val_labels[b, :, :, :]           # shape = (d, w, h, channel)
            img_b_g = val_imgs[b, :, :, :, :]            # shape = (d, w, h)
            fuse_b = fusion[b, :, :, :, :]
            epistemic_b = epistemic_var[b, :, :, :, :]   # shape = (d, w, h, channel)

            ### normalize to [0, 1]
            # epistemic_b = 1.0 - (epistemic_b - np.min(epistemic_b)) / (np.max(epistemic_b) - np.min(epistemic_b))

            pred_b_g_argmax = np.argmax(np.round(pred_b_g), axis=-1)      #.astype(np.uint8)   # shape = (d, w, h)

            fused_seg = np.argmax(fuse_b, axis=-1)                                      # lower accuracy ? # shape = (d, w, h)

            print("f1 score of geng = %d " % g, " batch = %d" % b, " : ", f1_score(pred_b_g_argmax.flatten(), label_b_g.flatten(), average=None))
            print("f1 score of genf = %d " % g, " batch = %d" % b, " : ", f1_score(fused_seg.flatten(), label_b_g.flatten(), average=None))

            dice_g_mean.append(dice_coefficient(pred_b_g_argmax, label_b_g))  #    jaccard_similarity_score(pred_b_g_argmax.flatten(), label_b_g.flatten()))
            dice_g_fuse_mean.append(dice_coefficient(fused_seg, label_b_g))

            #######################################################################################################

            slices_img = []
            for t in range(img_b_g.shape[3]):
                slices_img.append(sitk.GetImageFromArray(np.transpose(img_b_g[:, :, :, t], (2, 1, 0))))
            img = sitk.JoinSeries(slices_img)

            slice_fuse = []
            for t in range(fuse_b.shape[3]):
                slice_fuse.append(sitk.GetImageFromArray(np.transpose(fuse_b[:, :, :, t], (2, 1, 0))))
            fuse = sitk.JoinSeries(slice_fuse)

            slice_epis = []
            for t in range(epistemic_b.shape[3]):
                slice_epis.append(sitk.GetImageFromArray(np.transpose(epistemic_b[:, :, :, t], (2, 1, 0))))
            epis = sitk.JoinSeries(slice_epis)


            base_dir = project_path + 'validation_results/' + model_name + '/generator_%d' % g + '/'

            pred_b_g_argmax = np.transpose(pred_b_g_argmax, (2, 1, 0))
            label_b_g = np.transpose(label_b_g, (2, 1, 0))

            if FLAGS.pgan == False:
                fused_seg = np.transpose(fused_seg, (2, 1, 0))

            sitk.WriteImage(sitk.GetImageFromArray(pred_b_g_argmax.astype(np.uint8)), base_dir + val_img_base_name + '_seg.nii.gz')  # + str(indx)  pred_b_g_argmax
            sitk.WriteImage(sitk.GetImageFromArray(label_b_g), base_dir + val_img_base_name + '_label.nii.gz')
            sitk.WriteImage(img, base_dir + val_img_base_name + '_img.nii.gz')
            # sitk.WriteImage(fuse, base_dir + val_img_base_name + '_fuse.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(fused_seg.astype(np.uint8)), base_dir + val_img_base_name + '_fuse_seg.nii.gz')
            # sitk.WriteImage(epis, base_dir + val_img_base_name + '_epis.nii.gz')

            #################################### Uncertainty ###################################################

            # trace_cov = get_dirichlet_variance(fuse_b)
            # trace_cov = (trace_cov - np.min(trace_cov)) / (np.max(trace_cov) - np.min(trace_cov))
            #  sitk.WriteImage(sitk.GetImageFromArray(np.transpose(trace_cov, (2, 1, 0))), base_dir + val_img_base_name + '_var.nii.gz')

            ##############################################################

        dice_total_mean.append(np.mean(dice_g_mean))
        dice_total_fused_mean.append(np.mean(dice_g_fuse_mean))


    return dice_total_mean, dice_total_fused_mean


def get_on_the_fly_test_data(session, model, test_imgs, model_name, batch_test_files):

    for g in range(FLAGS.J):

        save_results_dir = project_path + 'segmentation_results/' + model_name + '/generator_%d' % g + '/'

        if not os.path.exists(save_results_dir):
            os.makedirs(save_results_dir)

    predicts, fusion, epistemic_var = session.run([model.predicts, model.gen_fused, model.epistemic_var], feed_dict={model.imgs: test_imgs})
    predicts_arr = np.asarray(predicts)

    for g in range(predicts_arr.shape[0]):

        for b in range(FLAGS.batch_size):

            test_img_base_name = batch_test_files[b].split(".")[0]

            pred_b_g = predicts_arr[g, b, :, :, :, :]    # shape = (d, w, h, num_classes)
            img_b_g = test_imgs[b, :, :, :, :]          # shape = (d, w, h, channel=1)
            fuse_b = fusion[b, :, :, :, :]               # shape = (d, w, h, num_classes)
            epistemic_b = epistemic_var[b, :, :, :, :]   # shape = (d, w, h, channel)

            # pred_b_g = np.transpose(pred_b_g, (3, 2, 1, 0))
            # img_b_g = np.transpose(img_b_g, (2, 1, 0))
            # fuse_b = np.transpose(fuse_b, (3, 2, 1, 0))
            # epistemic_b = np.transpose(epistemic_b, (3, 2, 1, 0))

            ### normalize to [0, 1]
            epistemic_b = 1.0 - (epistemic_b - np.min(epistemic_b)) / (np.max(epistemic_b) - np.min(epistemic_b))

            pred_b_g_argmax = np.argmax(np.round(pred_b_g), axis=-1).astype(np.uint8)   # shape = (d, w, h)
            fused_seg = np.argmax(fuse_b, axis=-1).astype(np.uint8)  # lower accuracy ？                 # shape = (d, w, h)

            #######################################################################################################

            #img = sitk.GetImageFromArray(img_b_g)

            slice_img = []
            for t in range(img_b_g.shape[3]):
                slice_img.append(sitk.GetImageFromArray(np.transpose(img_b_g[:, :, :, t], (2, 1, 0))))
            img = sitk.JoinSeries(slice_img)

            slice_fuse = []
            for t in range(fuse_b.shape[3]):
                slice_fuse.append(sitk.GetImageFromArray(np.transpose(fuse_b[:, :, :, t], (2, 1, 0))))
            fuse = sitk.JoinSeries(slice_fuse)

            slice_epis = []
            for t in range(epistemic_b.shape[3]):
                slice_epis.append(sitk.GetImageFromArray(np.transpose(epistemic_b[:, :, :, t], (2, 1, 0))))
            epis = sitk.JoinSeries(slice_epis)

            base_dir = project_path + 'segmentation_results/' + model_name + '/generator_%d' % g + '/'

            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(pred_b_g_argmax, (2, 1, 0))), base_dir + test_img_base_name + '_seg.nii.gz')  # + str(indx)  pred_b_g_argmax
            sitk.WriteImage(img, base_dir + test_img_base_name + '_img.nii.gz')
            # sitk.WriteImage(fuse, base_dir + test_img_base_name + '_fuse.nii.gz')
            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(fused_seg, (2, 1, 0))), base_dir + test_img_base_name + '_fuse_seg.nii.gz')
            # sitk.WriteImage(epis, base_dir + test_img_base_name + '_epis.nii.gz')

            #################################### Uncertainty ###################################################

            trace_cov = get_dirichlet_variance(fuse_b)
            trace_cov = (trace_cov - np.min(trace_cov)) / (np.max(trace_cov) - np.min(trace_cov))
            # sitk.WriteImage(sitk.GetImageFromArray(np.transpose(trace_cov, (2, 1, 0))), base_dir + test_img_base_name + '_var.nii.gz')

            #####################################################################################################


def b_patchGan_3d(dataset, FLAGS, patch_size, channel=1):

    model_base_name = dataset.name + '_bpgan3d' + '_%d' % FLAGS.gf_dim + '_%d' % FLAGS.df_dim + '_%d' % FLAGS.J + '_' + FLAGS.fn

    # save model weights
    model_dir = dataset.name + "_%s_%s_%s_%s" % ('bpgan3d', FLAGS.batch_size, FLAGS.gf_dim, FLAGS.J) + '_' + FLAGS.fn
    checkpoint_dir = os.path.join(FLAGS.checkpoint_dir, model_dir)

    x_dim = patch_size
    n_classes = FLAGS.num_classes

    train_img_files = random.sample(dataset.train_img_files, len(dataset.train_img_files))
    val_img_files = random.sample(dataset.val_img_files, len(dataset.val_img_files))

    num_train = dataset.num_train
    num_val = dataset.num_val

    print("num_train = ", num_train, num_val)

    session = get_session()
    tf.set_random_seed(FLAGS.random_seed)

    ######################################################################################################################

    with tf.device("/gpu:3"):

        if FLAGS.reduce_pooling:

            bpgan = B_PATCHGAN_3D_REDUCE(x_dim, batch_size=FLAGS.batch_size, J=FLAGS.J,
                                    lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J==1),
                                    num_train=num_train, l1lambda=FLAGS.l1_lambda, num_classes=n_classes, channel=channel)

            # show_all_variables()

        elif FLAGS.extra_pooling:

            bpgan = B_PATCHGAN_3D_EXTRA(x_dim, batch_size=FLAGS.batch_size, J=FLAGS.J,
                                         lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                                         num_train=num_train, l1lambda=FLAGS.l1_lambda, num_classes=n_classes, channel=channel)


        elif FLAGS.pgan:

            bpgan = PATCHGAN_3D(x_dim, batch_size=FLAGS.batch_size, J=FLAGS.J,
                                        lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J == 1),
                                        num_train=num_train, l1lambda=FLAGS.l1_lambda, num_classes=n_classes, channel=channel)


        else:

            bpgan = B_PATCHGAN_3D(x_dim, batch_size=FLAGS.batch_size, J=FLAGS.J,
                                    lr=FLAGS.lr, gf_dim=FLAGS.gf_dim, df_dim=FLAGS.df_dim, ml=(FLAGS.J==1),
                                    num_train=num_train, l1lambda=FLAGS.l1_lambda, num_classes=n_classes, channel=channel)


    #######################################################################################################################

    print("Starting session")

    session.run(tf.global_variables_initializer())

    if FLAGS.restore:

        saver = tf.train.Saver()
        saver.restore(session, tf.train.latest_checkpoint('/home/jtma/PycharmProjects/checkpoint/' + model_dir))

    print("Starting training loop")
    if FLAGS.pgan:
        optimizer_dict = {"disc": bpgan.d_optims_adam, "gen": bpgan.g_optims_adam}
    else:
        optimizer_dict = {"disc": bpgan.d_optims_adam, "gen": bpgan.g_optims_adam, "distrib": bpgan.distrib_optims_adam}

    base_learning_rate = FLAGS.lr     # for now we use same learning rate for Ds and Gs
    lr_decay_rate = FLAGS.lr_decay

    for train_iter in range(FLAGS.train_iter):

        learning_rate = base_learning_rate * np.exp(-lr_decay_rate * min(1.0, (train_iter * FLAGS.batch_size)/float(num_train)))

    ### get supervised_train_batch ###

        train_indices = np.random.choice(num_train, FLAGS.batch_size, replace=True)

        batch_train_files = []
        for _, idx in enumerate(train_indices):
            batch_train_files.append(train_img_files[idx])

        labeled_image_batch, labels = dataset.get_train_batch(train_img_dir=dataset.train_img_aug_dir,      # dataset.train_img_dir
                                                              train_label_dir=dataset.train_label_aug_dir,  # dataset.train_label_dir
                                                              batch_img_files=batch_train_files)

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

        gen_info = session.run(optimizer_dict["gen"] + bpgan.g_losses + bpgan.g_seg_losses,
                               feed_dict={bpgan.imgs: labeled_image_batch,
                                          bpgan.labels: labels,
                                          bpgan.g_learning_rate: learning_rate})


        g_losses = [g_ for g_ in gen_info if g_ is not None]   # len(gen_info) = 40 # , g_fake_losses, g_seg_losses


        if FLAGS.pgan == False:

            distrib_info = session.run(optimizer_dict["distrib"] + bpgan.distrib_losses,
                                       feed_dict={bpgan.imgs: labeled_image_batch,
                                                  bpgan.labels: labels,
                                                  bpgan.g_learning_rate: learning_rate})

            distrib_losses = [dist_ for dist_ in distrib_info if dist_ is not None]


            if train_iter % 20 == 0:

                print(" Iter %i" % train_iter,
                      " : Disc losses = %s" % (", ".join(["%.4f" % dl for dl in d_losses])),
                      " , Gen losses = %s" % (", ".join(["%.4f" % gl for gl in g_losses])),
                      " , Distrib losses = %s" % (", ".join(["%.4f" % gl for gl in distrib_losses])),
                      " : disc_loss (mean) = %.4f " % np.mean(d_losses),
                      " : gen_loss (mean) = %.4f" % np.mean(g_losses))

            else:

                if train_iter % 20 == 0:
                    print(" Iter %i" % train_iter,
                          " : Disc losses = %s" % (", ".join(["%.4f" % dl for dl in d_losses])),
                          " , Gen losses = %s" % (", ".join(["%.4f" % gl for gl in g_losses])),
                          " : disc_loss (mean) = %.4f " % np.mean(d_losses),
                          " : gen_loss (mean) = %.4f" % np.mean(g_losses))


        with tf.device("/cpu:1"):

            if train_iter % 100 == 0:

                val_indices = np.random.choice(num_val, FLAGS.batch_size, replace=True)

                batch_val_files = []
                for _, idx in enumerate(val_indices):
                    batch_val_files.append(val_img_files[idx])

                val_imgs, val_labels = dataset.get_val_batch(val_img_dir=dataset.train_img_aug_dir,    # dataset.train_img_dir
                                                             val_label_dir=dataset.train_label_aug_dir,    # dataset.train_label_dir
                                                             batch_img_files=batch_val_files)

                # batch_val_files = batch_train_files
                # val_imgs, val_labels = labeled_image_batch, labels

                if FLAGS.pgan:
                    dice_g, dice_f = get_val_accuracy_bpgan_fusion_pgan(session, bpgan, val_imgs, val_labels, model_dir, batch_val_files=batch_val_files, FLAGS=FLAGS)
                else:
                    dice_g, dice_f = get_val_accuracy_bpgan_fusion(session, bpgan, val_imgs, val_labels, model_dir, batch_val_files=batch_val_files)

                print(" Iter %i" % train_iter,
                      " : dice 1 = %s" % (", ".join(["%.4f" % dm for dm in dice_g])),
                      " : dice 2 = %s" % (", ".join(["%.4f" % dm for dm in dice_f])))


                #############################################################################################################

                if os.path.exists(dataset.test_patch_dir):

                    patch_files = os.listdir(dataset.test_patch_dir)

                    if len(patch_files) > 0:

                        test_indices = np.random.choice(int(len(patch_files)), FLAGS.batch_size, replace=True)

                        batch_test_files = []
                        for _, idx in enumerate(test_indices):
                            batch_test_files.append(patch_files[idx])

                        test_imgs = dataset.get_test_batch(batch_img_files=batch_test_files)

                        get_on_the_fly_test_data(session, bpgan, test_imgs, model_dir, batch_test_files=batch_test_files)


            if train_iter % 500 == 0:

                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                saver = tf.train.Saver()
                saver.save(session, os.path.join(checkpoint_dir, model_base_name), global_step=train_iter, write_meta_graph=False)


    print("bpgan training done")


if __name__ == "__main__":

    # set seeds
    np.random.seed(FLAGS.random_seed)
    tf.set_random_seed(FLAGS.random_seed)

    if FLAGS.task == 'Pancreas':

        patch_size = [96, 80, 64] # [128, 64, 64] # [96, 64, 64] # [32, 32, 32] # [64, 64, 64]
        resampled_spacing = (1.5, 1.5, 2.5)   # (1.5 ,1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 3

        pan_path = os.path.join(FLAGS.data_path, "Task07_Pancreas")
        dataset = Pancreas_3D(pan_path, patch_size=patch_size, resampled_spacing=resampled_spacing)

    elif FLAGS.task == 'Spleen':    # plus 1 pooling

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
        patch_num = 0  # this should be augmentation number
        resampled_spacing = (1.5, 1.5, 1.0)
        channel = 2
        FLAGS.num_classes = 3
        prostate_path = os.path.join(FLAGS.data_path, "Task05_Prostate")
        dataset = Prostate_3D(prostate_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, channel=channel)


    elif FLAGS.task == 'BrainTumor':  # plus 1 pooling

        patch_size = [64, 96, 64]
        patch_num = 0   # or augmentation, the brain does not need patch-cropping
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 4
        FLAGS.num_classes = 4
        brain_path = os.path.join(FLAGS.data_path, "Task01_BrainTumour")
        dataset = BrainTumor_3D(brain_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, channel=channel)


    elif FLAGS.task == 'Heart':

        patch_size = [64, 64, 96]
        patch_num = 0
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 2
        heart_path = os.path.join(FLAGS.data_path, "Task02_Heart")
        dataset = Heart_3D(heart_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio, num_bins=256)


    elif FLAGS.task == 'Hippocampus':

        patch_size = [32, 48, 32]
        resampled_spacing = (1.0, 1.0, 1.0)
        FLAGS.num_classes = 3
        channel = 1
        pan_path = os.path.join(FLAGS.data_path, "Task04_Hippocampus")
        dataset = Hippocampus_3D(pan_path, patch_size=patch_size, resampled_spacing=resampled_spacing, num_bins=256)


    elif FLAGS.task == 'Lung':

        patch_size = [64, 64, 64]
        resampled_spacing = (1.5, 1.5, 1.5)   # should testing
        FLAGS.num_classes = 2
        channel = 1
        lung_path = os.path.join(FLAGS.data_path, "Task06_Lung")
        dataset = Lung_3D(lung_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    elif FLAGS.task == 'Liver':   # this must be train in patch !

        patch_size = [64, 64, 64]
        resampled_spacing = (2.0, 2.0, 2.0)
        channel = 1
        FLAGS.num_classes = 3
        FLAGS.J = 3
        FLAGS.batch_size = 3
        FLAGS.reduce_pooling = True
        liver_path = os.path.join(FLAGS.data_path, "Task03_Liver")
        dataset = Liver_3D(liver_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)



    elif FLAGS.task == 'HepaticVessel':  # this must be train in patch !

        patch_size = [128, 96, 80]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 3
        FLAGS.reduce_pooling = True
        vessel_path = os.path.join(FLAGS.data_path, "Task08_HepaticVessel")
        dataset = HepaticVessel_3D(vessel_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    elif FLAGS.task == 'Colon':   # this must be train in patch !

        patch_size = [64, 64, 64]
        resampled_spacing = (1.5, 1.5, 1.5)
        channel = 1
        FLAGS.num_classes = 2
        colon_path = os.path.join(FLAGS.data_path, "Task10_Colon")
        dataset = Colon_3D(colon_path, patch_size=patch_size, resampled_spacing=resampled_spacing, val_ratio=FLAGS.val_ratio)


    else:

        print("select a task!")

##########################################################################################################################################


    if FLAGS.phase == "train":
        b_patchGan_3d(dataset, FLAGS, patch_size=patch_size, channel=channel)     #    unet(dataset, FLAGS)
    else:
        print("please select the phase !")
