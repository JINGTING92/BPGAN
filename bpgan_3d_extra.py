import numpy as np
import tensorflow as tf
import pandas as pd

import sklearn.ensemble.voting_classifier

from collections import OrderedDict, defaultdict

from bgan_util import AttributeDict, dice_coef, stable_nll_loss

from dcgan_ops import *

from scipy.special import gamma, gammaln


def conv_out_size(size, stride):
    return int(math.ceil(size / float(stride)))


def kernel_sizer(size, stride):
    ko = int(math.ceil(size / float(stride)))
    if ko % 2 == 0:
        ko += 1
    return ko


class B_PATCHGAN_3D_EXTRA(object):

    def __init__(self, x_dim, batch_size=64, gf_dim=64, df_dim=64,
                 prior_std=1.0, J=1, alpha=0.01, lr=0.005, optimizer='adam', wasserstein=False, num_classes=1,
                 ml=False, num_train=64, l1lambda=0.1, channel=1):  # eta=2e-4,

        self.optimizer = optimizer.lower()
        self.batch_size = batch_size
        self.num_train = num_train
        self.channel = channel
        self.num_classes = num_classes

        self.x_dim = x_dim
        self.gf_dim = gf_dim
        self.df_dim = df_dim
        self.lr = lr
        self.l1_lambda = l1lambda

        # Bayes
        self.prior_std = prior_std   # sigma
        self.num_gen = J  # what is num_gen ??
        self.num_disc = 1
        self.num_mcmc = 1  # M
        self.alpha = alpha
        self.noise_std = np.sqrt(2 * self.alpha * self.lr)  # self.eta = self.lr as default | 10

        # ML
        self.ml = ml
        if self.ml:
            assert self.num_gen == 1 and self.num_disc == 1 and self.num_mcmc == 1, "invalid settings for ML training"

        def get_strides(num_layers, num_pool):
            interval = int(math.floor(num_layers / float(num_pool)))
            strides = np.array([1] * num_layers)
            strides[0:interval * num_pool:interval] = 2
            return strides

        # generator : number of pooling 4 pool + 4 unpool
        self.num_pool = 4
        num_layers = 4
        self.gen_strides = get_strides(num_layers, self.num_pool)
        self.disc_strides = self.gen_strides
        num_dfs = np.cumprod(np.array([self.df_dim] + list(self.disc_strides)))[:-1]
        self.num_dfs = list(num_dfs)
        self.num_gfs = self.num_dfs  # [::-1] - not reverse

        self.construct_from_hypers()

        self.build_bpgan_graph()


    def construct_from_hypers(self, gen_kernel_size=3, num_dfs=None, num_gfs=None):

        self.num_disc_layers = 5
        self.num_gen_layers = 19 + 4


        self.d_batch_norm = AttributeDict([("d_bn%i" % dbn_i, batch_norm(name='d_bn%i' % dbn_i)) for dbn_i in range(self.num_disc_layers)])
        self.g_batch_norm = AttributeDict([("g_bn%i" % gbn_i, batch_norm(name='g_bn%i' % gbn_i)) for gbn_i in range(self.num_gen_layers)])


        if num_dfs is None:
            num_dfs = [self.df_dim, self.df_dim * 2, self.df_dim * 4, self.df_dim * 8, self.df_dim]

        if num_gfs is None:

            num_gfs = [self.gf_dim, self.gf_dim, self.gf_dim * 2, self.gf_dim * 2,
                       self.gf_dim * 4, self.gf_dim * 4, self.gf_dim * 8, self.gf_dim * 8,
                       self.gf_dim * 16, self.gf_dim * 16,
                       self.gf_dim * 8, self.gf_dim * 8,
                       self.gf_dim * 4, self.gf_dim * 4,
                       self.gf_dim * 2, self.gf_dim * 2,
                       self.gf_dim, self.gf_dim,
                       self.num_classes, self.num_classes,   ## output logits
                       self.gf_dim * 16, self.gf_dim * 16,   ## extra pooling layer
                       self.gf_dim * 16, self.gf_dim * 16]   ## extra pooling layer

        s_h, s_w, s_d = self.x_dim[0], self.x_dim[1], self.x_dim[2]
        s_h2, s_w2, s_d2 = conv_out_size(s_h, 2), conv_out_size(s_w, 2), conv_out_size(s_d, 2)
        s_h4, s_w4, s_d4 = conv_out_size(s_h2, 2), conv_out_size(s_w2, 2), conv_out_size(s_d2, 2)
        s_h8, s_w8, s_d8 = conv_out_size(s_h4, 2), conv_out_size(s_w4, 2), conv_out_size(s_d4, 2)
        s_h16, s_w16, s_d16 = conv_out_size(s_h8, 2), conv_out_size(s_w8, 2), conv_out_size(s_d8, 2)

        ks = gen_kernel_size
#        self.gen_output_dims = OrderedDict()
        self.gen_weight_dims = OrderedDict()

        num_gfs = num_gfs + [self.channel]
        self.gen_kernel_sizes = [ks]

        #### build unet_generator from the one-by-one

        self.gen_weight_dims["g_h%i_W" % 0] = (3, 3, 3, self.channel, num_gfs[0])  # from the image
        self.gen_weight_dims["g_h%i_b" % 0] = (num_gfs[0],)
        self.gen_weight_dims["g_h%i_W" % 1] = (3, 3, 3, num_gfs[1], num_gfs[1])  # conv1
        self.gen_weight_dims["g_h%i_b" % 1] = (num_gfs[1],)


        self.gen_weight_dims["g_h%i_W" % 2] = (3, 3, 3, num_gfs[1], num_gfs[2])
        self.gen_weight_dims["g_h%i_b" % 2] = (num_gfs[2],)
        self.gen_weight_dims["g_h%i_W" % 3] = (3, 3, 3, num_gfs[3], num_gfs[3])  # conv2
        self.gen_weight_dims["g_h%i_b" % 3] = (num_gfs[3],)


        self.gen_weight_dims["g_h%i_W" % 4] = (3, 3, 3, num_gfs[3], num_gfs[4])
        self.gen_weight_dims["g_h%i_b" % 4] = (num_gfs[4],)
        self.gen_weight_dims["g_h%i_W" % 5] = (3, 3, 3, num_gfs[5], num_gfs[5])  # conv3
        self.gen_weight_dims["g_h%i_b" % 5] = (num_gfs[5],)


        self.gen_weight_dims["g_h%i_W" % 6] = (3, 3, 3, num_gfs[5], num_gfs[6])
        self.gen_weight_dims["g_h%i_b" % 6] = (num_gfs[6],)
        self.gen_weight_dims["g_h%i_W" % 7] = (3, 3, 3, num_gfs[7], num_gfs[7])  # conv4
        self.gen_weight_dims["g_h%i_b" % 7] = (num_gfs[7],)



        self.gen_weight_dims["g_h%i_W" % 10] = (3, 3, 3, num_gfs[9] + num_gfs[7], num_gfs[10])  # conv6 concat conv4
        self.gen_weight_dims["g_h%i_b" % 10] = (num_gfs[10],)
        self.gen_weight_dims["g_h%i_W" % 11] = (3, 3, 3, num_gfs[11], num_gfs[11])
        self.gen_weight_dims["g_h%i_b" % 11] = (num_gfs[11],)


        self.gen_weight_dims["g_h%i_W" % 12] = (3, 3, 3, num_gfs[11] + num_gfs[5], num_gfs[12])  # conv7 concat conv3
        self.gen_weight_dims["g_h%i_b" % 12] = (num_gfs[12],)
        self.gen_weight_dims["g_h%i_W" % 13] = (3, 3, 3, num_gfs[13], num_gfs[13])
        self.gen_weight_dims["g_h%i_b" % 13] = (num_gfs[13],)


        self.gen_weight_dims["g_h%i_W" % 14] = (3, 3, 3, num_gfs[13] + num_gfs[3], num_gfs[14])  # conv8 concat conv2
        self.gen_weight_dims["g_h%i_b" % 14] = (num_gfs[14],)
        self.gen_weight_dims["g_h%i_W" % 15] = (3, 3, 3, num_gfs[15], num_gfs[15])
        self.gen_weight_dims["g_h%i_b" % 15] = (num_gfs[15],)


        self.gen_weight_dims["g_h%i_W" % 16] = (3, 3, 3, num_gfs[15] + num_gfs[1], num_gfs[16])  # conv9 concat conv1
        self.gen_weight_dims["g_h%i_b" % 16] = (num_gfs[16],)
        self.gen_weight_dims["g_h%i_W" % 17] = (3, 3, 3, num_gfs[17], num_gfs[17])
        self.gen_weight_dims["g_h%i_b" % 17] = (num_gfs[17],)


        ################### output layer #########################

        self.gen_weight_dims["g_h%i_W" % 18] = (1, 1, 1, num_gfs[17], num_gfs[18])
        self.gen_weight_dims["g_h%i_b" % 18] = (num_gfs[18],)


        ##################  Extra Pooling Layer ###################

        # if self.more_pooling == True:

        self.gen_weight_dims["g_h%i_W" % 19] = (3, 3, 3, 8 * self.gf_dim, 16 * self.gf_dim)
        self.gen_weight_dims["g_h%i_b" % 19] = (16 * self.gf_dim,)
        self.gen_weight_dims["g_h%i_W" % 20] = (3, 3, 3, 16 * self.gf_dim, 16 * self.gf_dim)  # conv4
        self.gen_weight_dims["g_h%i_b" % 20] = (16 * self.gf_dim,)

        self.gen_weight_dims["g_h%i_W" % 8] = (3, 3, 3, 16 * self.gf_dim, 16 * self.gf_dim)
        self.gen_weight_dims["g_h%i_b" % 8] = (16 * self.gf_dim,)
        self.gen_weight_dims["g_h%i_W" % 9] = (3, 3, 3, 16 * self.gf_dim, 16 * self.gf_dim)  # conv5
        self.gen_weight_dims["g_h%i_b" % 9] = (16 * self.gf_dim,)

        self.gen_weight_dims["g_h%i_W" % 21] = (3, 3, 3, 16 * self.gf_dim + 16 * self.gf_dim, 16 * self.gf_dim)
        self.gen_weight_dims["g_h%i_b" % 21] = (16 * self.gf_dim,)
        self.gen_weight_dims["g_h%i_W" % 22] = (3, 3, 3, 16 * self.gf_dim, 16 * self.gf_dim)  # conv4
        self.gen_weight_dims["g_h%i_b" % 22] = (16 * self.gf_dim,)


#########################################################################################################


        self.disc_weight_dims = OrderedDict()
        self.disc_weight_dims["d_h%i_W" % 0] = (5, 5, 5, self.num_classes + self.channel, num_dfs[0])  # output = ( s_h / 2, s_w / 2 )
        self.disc_weight_dims["d_h%i_b" % 0] = (num_dfs[0],)
        self.disc_weight_dims["d_h%i_W" % 1] = (5, 5, 5, num_dfs[0], num_dfs[1])  # output = ( s_h / 4, s_w / 4 )
        self.disc_weight_dims["d_h%i_b" % 1] = (num_dfs[1],)
        self.disc_weight_dims["d_h%i_W" % 2] = (5, 5, 5, num_dfs[1], num_dfs[2])  # output = ( s_h / 8, s_w / 8 )
        self.disc_weight_dims["d_h%i_b" % 2] = (num_dfs[2],)
        self.disc_weight_dims["d_h%i_W" % 3] = (5, 5, 5, num_dfs[2], num_dfs[3])  # output = ( s_h / 16, s_w / 16 )   # pre: 1, 1,
        self.disc_weight_dims["d_h%i_b" % 3] = (num_dfs[3],)
        # self.disc_weight_dims["d_h%i_W" % 3] = (1, 1, 1, num_dfs[2], 1)  # output = ( s_h / 16, s_w / 16 )   # pre: 1, 1,
        # self.disc_weight_dims["d_h%i_b" % 3] = (1,)
        self.disc_weight_dims["d_h%i_W" % 4] = (1, 1, 1, num_dfs[3], 1)
        self.disc_weight_dims["d_h%i_b" % 4] = (1,)

        # self.disc_weight_dims.update(OrderedDict([("d_h_out_lin_W", (num_dfs[3] * s_h8 * s_w8, 1)),
        #                                           ("d_h_out_lin_b", (1,))]))


#####################################################################################################################

        self.distrib_weight_dims = OrderedDict()
        for zi in range(self.num_gen):
            self.distrib_weight_dims["dist_%i_mu" % zi] = (1, self.x_dim[0], self.x_dim[1], self.x_dim[2], 1)   # self.num_classes
            self.distrib_weight_dims["dist_%i_var" % zi] = (1, self.x_dim[0], self.x_dim[1], self.x_dim[2], 1)


#####################################################################################################################


        # for k, v in self.gen_weight_dims.items():   # k is the name, v is the dim
        #     print("gen_weight_dims - %s: %s" % (k, v))
        # print('****')
        # for k, v in self.disc_weight_dims.items():
        #     print("dics_weight_dims - %s: %s" % (k, v))
        # print('****')
        # for k, v in self.distrib_weight_dims.items():
        #     print("dics_weight_dims - %s: %s" % (k, v))


    def initialize_dist_wgts(self, scope_str):

        if scope_str == "distrib_classifier":
            weight_dims = self.distrib_weight_dims
        else:
            raise RuntimeError("invalid scope!")

        param_list = []
        with tf.variable_scope(scope_str) as scope:
            wgts_ = AttributeDict()

            for zi in range(self.num_gen):
                mu_name = "dist_%i_mu" % (zi)
                mu_shape = weight_dims[mu_name]

                var_name = "dist_%i_var" % (zi)
                var_shape = weight_dims[var_name]

                wgts_[mu_name] = tf.get_variable("%s" % mu_name, mu_shape, initializer=tf.random_normal_initializer(mean=1/self.num_gen, stddev=0.02))
                wgts_[var_name] = tf.get_variable("%s" % var_name, var_shape, initializer=tf.random_normal_initializer(stddev=0.02))

            # for name, shape in weight_dims.items():
            #    wgts_[name] = tf.get_variable("%s" % name, shape, initializer=tf.random_normal_initializer(mean=1/self.num_gen, stddev=0.02))

            param_list.append(wgts_)

            return param_list


    def initialize_wgts(self, scope_str):

        if scope_str == "generator":
            weight_dims = self.gen_weight_dims  # len(weight_dims) = 10 / 2
            numz = self.num_gen
        elif scope_str == "patch_discriminator":
            weight_dims = self.disc_weight_dims  # len(weight_dims) = 10 / 2
            numz = self.num_disc
        else:
            raise RuntimeError("invalid scope!")

        param_list = []
        with tf.variable_scope(scope_str) as scope:  # iterated J (numz / num_gen) x num_mcmc = 20
            for zi in range(numz):  # numz: num_gen / num_disc
                for m in range(self.num_mcmc):
                    wgts_ = AttributeDict()
                    for name, shape in weight_dims.items():
                        wgts_[name] = tf.get_variable("%s_%04d_%04d" % (name, zi, m), shape, initializer=tf.random_normal_initializer(stddev=0.02))
                    param_list.append(wgts_)

            return param_list


    ########################################################################################################################


    def build_bpgan_graph(self):

        data_pair = tf.placeholder(tf.float32, [self.batch_size, self.x_dim[0], self.x_dim[1], self.x_dim[2], self.num_classes + self.channel], name='stacked_data_pair')

        self.imgs = data_pair[:, :, :, :, self.num_classes: self.num_classes + self.channel]  #  self.imgs = data_pair[:, :, :, :, self.channel: self.channel + self.channel]
        #self.labels = data_pair[:, :, :, :, :1]  #  self.labels = data_pair[:, :, :, :, : self.channel]

        self.labels = tf.placeholder(tf.float32, [self.batch_size, self.x_dim[0], self.x_dim[1], self.x_dim[2]], name='input_labels')
        self.one_hot_labels = tf.one_hot(tf.to_int32(self.labels), depth=self.num_classes)  # reduce the last dim

        self.pos_pair = tf.concat([self.imgs, self.one_hot_labels], axis=4)

        self.gen_param_list = self.initialize_wgts("generator")
        self.disc_param_list = self.initialize_wgts("patch_discriminator")
        self.distrib_param_list = self.initialize_dist_wgts("distrib_classifier")

        self.d_semi_learning_rate = tf.placeholder(tf.float32, shape=[])
        self.g_learning_rate = tf.placeholder(tf.float32, shape=[])

        t_vars = tf.trainable_variables()  # compile all disciminative weights  # returns a list of trainable variables

        self.d_vars = []
        self.d_vars.append([var for var in t_vars if 'd_' in var.name and "_%04d_%04d" % (0, 0) in var.name])  # var for var in t_vars if 'd_' in var.name

        self.g_vars = []
        for gi in range(self.num_gen):
            for m in range(self.num_mcmc):
                self.g_vars.append([var for var in t_vars if 'g_' in var.name and "_%04d_%04d" % (gi, m) in var.name])

        self.dist_vars = []
        self.dist_vars.append([var for var in t_vars if 'dist_' in var.name])

        self.d_losses, self.d_optims_adam = [], []
        self.g_losses, self.g_optims_adam = [], []
        self.distrib_losses, self.distrib_optims_adam = [], []

        self.g_seg_losses = []
        self.predicts = []
        self.patch_discs = []

        ############################ build discrimitive losses and optimizers ##########################################

        d_probs_pos_real, d_logits_pos_real = self.patch_discriminator(self.pos_pair, self.disc_param_list[0])  # self.pos_pair, self.disc_param_list[0]
        d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_pos_real, labels=tf.ones_like(d_probs_pos_real)))
#        tf.summary.histogram("d_real", d_loss_real)

        #######################################################

        ### Part II: fake - multi generators, multi fake losses ###
        d_loss_fakes = []
        d_losses_semi = []

        for gi, gen_params in enumerate(self.gen_param_list):

            gi_losses = []

            logit = self.generator(self.imgs, gen_params)  # sigma,

            #################################

            predict_relu = tf.nn.relu(logit)   # (batch, h, w, d, 3)

            if self.num_classes == 2:
                predict_softmax = tf.nn.sigmoid(logit)   # for a binary classification problem, the result is either 0 or 1.
            else:
                predict_softmax = tf.nn.softmax(logit)

            self.predicts.append(predict_softmax)   # (batch, h, w, d, 1)

            ##################################

            fake_AB = tf.concat([self.imgs, predict_softmax], axis=4)   # predict

            seg_loss_dice = 1. - dice_coef(predict_relu, self.one_hot_labels)   # self.labels

            if self.num_classes <= 2:
                seg_loss_cross = tf.losses.sigmoid_cross_entropy(multi_class_labels=self.one_hot_labels, logits=predict_softmax)
            else:
                seg_loss_cross = tf.losses.softmax_cross_entropy(onehot_labels=self.one_hot_labels, logits=predict_softmax)


            with tf.device("/gpu:0"):

                d_probs_fake_, d_logits_fake_ = self.patch_discriminator(fake_AB, self.disc_param_list[0])
                d_loss_fake_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake_, labels=tf.zeros_like(d_probs_fake_)))

                self.patch_discs.append(d_probs_fake_)

                d_loss_fakes.append(d_loss_fake_pos)

                g_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_logits_fake_, labels=tf.ones_like(d_probs_fake_)))        #logits=tf.log(tf.clip_by_value(d_logits_fake_,1e-10,1.0))
                g_loss_ = g_loss_fake

                if not self.ml:
                    g_loss_ += self.gen_prior(gen_params) + self.gen_noise(gen_params)  # return the prior_loss + noise_loss

                gi_losses.append(tf.reshape(g_loss_, [1]))

                g_loss = tf.reduce_sum(tf.concat(gi_losses, 0))

                ################# seg loss #########################################################################################

                g_loss += self.l1_lambda * seg_loss_dice + seg_loss_cross

                ####################################################################################################################

            with tf.device("/gpu:1"):

                self.g_losses.append(g_loss)
                self.g_seg_losses.append(seg_loss_dice)     # seg_loss_dice

#                tf.summary.histogram("g_loss_fake_%d" % gi, self.g_losses)
#                tf.summary.histogram("g_seg_%d" % gi, self.g_seg_losses)

                self.g_optimizer = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=0.5)
                self.g_optims_adam.append(self.g_optimizer.minimize(g_loss, var_list=self.g_vars[gi]))


        with tf.device("/gpu:2"):

            for d_loss_fake_ in d_loss_fakes:
                d_loss_semi_ = d_loss_real + d_loss_fake_ # * float(self.num_gen)
                d_losses_semi.append(tf.reshape(d_loss_semi_, [1]))

            d_loss_semi = tf.reduce_sum(tf.concat(d_losses_semi, 0))
            self.d_losses.append(d_loss_semi)

#            tf.summary.histogram("d_loss_fake", self.d_losses)

            # default iterations
            self.d_optimizer = tf.train.AdamOptimizer(learning_rate=self.d_semi_learning_rate, beta1=0.5)
            self.d_optims_adam.append(self.d_optimizer.minimize(d_loss_semi, var_list=self.d_vars[0]))

    ####################################################################################################################

            # is the self.gen_fused is the alpha in the dirichlet distribution
            self.gen_fused, exp_vars, log_vars = self.distrib_classifier(self.predicts, self.distrib_param_list[0])   # self.predicts

            if self.num_classes <= 2:
                fused_loss_seg = 1.0 - dice_coef(self.one_hot_labels, self.gen_fused) + \
                                 tf.losses.sigmoid_cross_entropy(multi_class_labels=self.one_hot_labels, logits=self.gen_fused)
            else:
                fused_loss_seg = 1.0 - dice_coef(self.one_hot_labels, self.gen_fused) + \
                             tf.losses.softmax_cross_entropy(onehot_labels=self.one_hot_labels, logits=self.gen_fused)   # output is relu

            fused_loss = tf.multiply(exp_vars, fused_loss_seg) + log_vars

            self.distrib_losses.append(1.0 - dice_coef(self.one_hot_labels, self.gen_fused))   # this loss is to print

            dist_optimizer = tf.train.AdamOptimizer(learning_rate=self.g_learning_rate, beta1=0.5)
            self.distrib_optims_adam.append(dist_optimizer.minimize(fused_loss, var_list=self.dist_vars[0]))


    #################################################################################################################################


    def distrib_classifier(self, image_list, dist_params):

        with tf.variable_scope("distrib_classifier", reuse=tf.AUTO_REUSE) as scope:

            # self.weighted_maps = []

            dist_0 = tf.distributions.Normal(loc=dist_params["dist_%i_mu" % 0], scale=dist_params["dist_%i_var" % 0]).sample()
            y = tf.multiply(tf.to_float(image_list[0]), dist_0)

            # self.weighted_maps.append(dist_params["dist_%i_var" % 0])

            log_vars = dist_params["dist_%i_var" % 0]
            exp_vars = tf.exp(- dist_params["dist_%i_var" % 0])

            sum_predictions = image_list[0]
            square_prediction = tf.square(image_list[0])

            for zi in range(1, self.num_gen):

                dist_zi = tf.distributions.Normal(loc=dist_params["dist_%i_mu" % zi], scale=dist_params["dist_%i_var" % zi]).sample()
                y = tf.add(y, tf.multiply(tf.to_float(image_list[zi]), dist_zi))   # dist_params["dist_%i_mu" % zi]

                # self.weighted_maps.append(dist_params["dist_%i_var" % zi])

                log_vars = tf.add(log_vars, dist_params["dist_%i_var" % zi])
                exp_vars = tf.add(exp_vars, tf.exp(- dist_params["dist_%i_var" % zi]))

                sum_predictions = tf.add(sum_predictions, image_list[zi])
                square_prediction = tf.add(square_prediction, tf.square(image_list[zi]))


            self.epistemic_var = (square_prediction - sum_predictions) / (self.num_gen)

            return y, exp_vars, log_vars

        # with tf.variable_scope("distrib_classifier", reuse=tf.AUTO_REUSE) as scope:
        #
        #     y = tf.multiply(image_list[0], dist_params["dist_%i_mu" % 0])
        #     for zi in range(1, self.num_gen):
        #
        #         y = tf.add(y, tf.multiply(image_list[zi], dist_params["dist_%i_mu" % zi]))   # dist_params["dist_%i_mu" % zi]
        #
        #     return y


    def patch_discriminator(self, image, disc_params):  # , disc_params

        with tf.variable_scope("patch_discriminator", reuse=tf.AUTO_REUSE) as scope:

            h0 = lrelu(conv3d(image, self.df_dim, name='d_h0_conv', w=disc_params["d_h%i_W" % 0], biases=disc_params["d_h%i_b" % 0]))


            h1 = lrelu(self.d_batch_norm["d_bn%i" % 1](
                conv3d(h0, self.df_dim * 2, name='d_h1_conv', w=disc_params["d_h%i_W" % 1],
                       biases=disc_params["d_h%i_b" % 1])))


            h2 = lrelu(self.d_batch_norm["d_bn%i" % 2](
                conv3d(h1, self.df_dim * 4, name='d_h2_conv', w=disc_params["d_h%i_W" % 2],
                       biases=disc_params["d_h%i_b" % 2])))


            h3 = lrelu(self.d_batch_norm["d_bn%i" % 3](
                conv3d(h2, self.df_dim * 8, name='d_h3_conv', w=disc_params["d_h%i_W" % 3],
                       biases=disc_params["d_h%i_b" % 3]))) #  k_w=1, k_h=1,

            #### brain_tumor_j=3, new h3 , output h3
            # h3 = self.d_batch_norm["d_bn%i" % 3](
            #     conv3d(h2, self.df_dim * 8, name='d_h3_conv', k_w=1, k_h=1, w=disc_params["d_h%i_W" % 3], biases=disc_params["d_h%i_b" % 3]))  #


            h4 = self.d_batch_norm["d_bn%i" % 4](
                conv3d(h3, 1, name='d_h4_conv', k_w=1, k_h=1, w=disc_params["d_h%i_W" % 4],
                       biases=disc_params["d_h%i_b" % 4]))  #

            # # h3 is (16 x 16 x self.df_dim*8)
            # h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h_out_lin_W', matrix=disc_params.d_h_out_lin_W,
            #                 bias=disc_params.d_h_out_lin_b)
            #
            # print('discrinimator h4', h4.shape)  # shape = (64, 1)


            return tf.nn.sigmoid(h3), h3  # probs for each sample (image), logits


    def generator(self, image, gen_params):

        with tf.variable_scope("generator", reuse=tf.AUTO_REUSE) as scope:

            conv1 = conv3d(image, self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv1_conv1', w=gen_params["g_h%i_W" % 0], biases=gen_params["g_h%i_b" % 0])
            conv1 = self.g_batch_norm["g_bn%i" % 0](conv1)
            conv1 = tf.nn.relu(conv1, name='conv1_relu1')
            conv1 = conv3d(conv1, self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv1_conv2', w=gen_params["g_h%i_W" % 1], biases=gen_params["g_h%i_b" % 1])
            conv1 = self.g_batch_norm["g_bn%i" % 1](conv1)
            conv1 = tf.nn.relu(conv1, name='conv1_relu2')
            pool1 = maxpooling_3d(conv1, name='maxpool1')

            conv2 = conv3d(pool1, 2 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv2_conv1', w=gen_params["g_h%i_W" % 2], biases=gen_params["g_h%i_b" % 2])
            conv2 = self.g_batch_norm["g_bn%i" % 2](conv2)
            conv2 = tf.nn.relu(conv2, name='conv2_relu1')
            conv2 = conv3d(conv2, 2 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv2_conv2', w=gen_params["g_h%i_W" % 3], biases=gen_params["g_h%i_b" % 3])
            conv2 = self.g_batch_norm["g_bn%i" % 3](conv2)
            conv2 = tf.nn.relu(conv2, name='conv2_relu2')
            pool2 = maxpooling_3d(conv2, name='maxpool2')


            conv3 = conv3d(pool2, 4 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv3_conv1', w=gen_params["g_h%i_W" % 4], biases=gen_params["g_h%i_b" % 4])
            conv3 = self.g_batch_norm["g_bn%i" % 4](conv3)
            conv3 = tf.nn.relu(conv3, name='conv3_relu1')
            conv3 = conv3d(conv3, 4 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv3_conv2', w=gen_params["g_h%i_W" % 5], biases=gen_params["g_h%i_b" % 5])
            conv3 = self.g_batch_norm["g_bn%i" % 5](conv3)
            conv3 = tf.nn.relu(conv3, name='conv3_relu2')
            pool3 = maxpooling_3d(conv3, name='maxpool3')
            pool3 = tf.nn.dropout(pool3, .5)


            conv4 = conv3d(pool3, 8 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv4_conv1', w=gen_params["g_h%i_W" % 6], biases=gen_params["g_h%i_b" % 6])
            conv4 = self.g_batch_norm["g_bn%i" % 6](conv4)
            conv4 = tf.nn.relu(conv4, name='conv4_relu1')
            conv4 = conv3d(conv4, 8 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv4_conv2', w=gen_params["g_h%i_W" % 7], biases=gen_params["g_h%i_b" % 7])
            conv4 = self.g_batch_norm["g_bn%i" % 7](conv4)
            conv4 = tf.nn.relu(conv4, name='conv4_relu2')
            pool4 = maxpooling_3d(conv4, name='maxpool4')   # maxpooling_3d
            pool4 = tf.nn.dropout(pool4, .5)

            ####################################################################################################################


            conv10 = conv3d(pool4, 16 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv10_conv1', w=gen_params["g_h%i_W" % 19], biases=gen_params["g_h%i_b" % 19])
            conv10 = self.g_batch_norm["g_bn%i" % 19](conv10)
            conv10 = tf.nn.relu(conv10, name='conv10_relu1')
            conv10 = conv3d(conv10, 16 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv10_conv2', w=gen_params["g_h%i_W" % 20], biases=gen_params["g_h%i_b" % 20])
            conv10 = self.g_batch_norm["g_bn%i" % 20](conv10)
            conv10 = tf.nn.relu(conv10, name='conv10_relu2')
            pool5 = maxpooling_3d(conv10, name='maxpool5')
            pool5 = tf.nn.dropout(pool5, .5)

            conv5 = conv3d(pool5, 16 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv5_conv1', w=gen_params["g_h%i_W" % 8], biases=gen_params["g_h%i_b" % 8])
            conv5 = self.g_batch_norm["g_bn%i" % 8](conv5)
            conv5 = tf.nn.relu(conv5, name='conv5_relu1')
            conv5 = conv3d(conv5, 16 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv5_conv2', w=gen_params["g_h%i_W" % 9], biases=gen_params["g_h%i_b" % 9])
            conv5 = self.g_batch_norm["g_bn%i" % 9](conv5)
            conv5 = tf.nn.relu(conv5, name='conv5_relu2')
            conv5 = tf.nn.dropout(conv5, .5)

            up5 = unpool3d(conv5)
            conv11 = tf.concat([up5, conv10], axis=4, name='conv11_concat')  # axis = 3 ??
            conv11 = conv3d(conv11, 16 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv11_conv1', w=gen_params["g_h%i_W" % 21], biases=gen_params["g_h%i_b" % 21])
            conv11 = self.g_batch_norm["g_bn%i" % 21](conv11)
            conv11 = tf.nn.relu(conv11, name='conv11_relu1')
            conv11 = conv3d(conv11, 16 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv11_conv2', w=gen_params["g_h%i_W" % 22], biases=gen_params["g_h%i_b" % 22])
            conv11 = self.g_batch_norm["g_bn%i" % 22](conv11)
            conv11 = tf.nn.relu(conv11, name='conv11_relu2')
            conv11 = tf.nn.dropout(conv11, .5)

            conv5 = conv11  # to be consistent with the original unpooling layers
            #
            # ########################################################################################################################
            #

            up1 = unpool3d(conv5)
            conv6 = tf.concat([up1, conv4], axis=4, name='conv6_concat')  # axis = 3 ??
            conv6 = conv3d(conv6, 8 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv6_conv1', w=gen_params["g_h%i_W" % 10], biases=gen_params["g_h%i_b" % 10])
            conv6 = self.g_batch_norm["g_bn%i" % 10](conv6)
            conv6 = tf.nn.relu(conv6, name='conv6_relu1')
            conv6 = conv3d(conv6, 8 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv6_conv2', w=gen_params["g_h%i_W" % 11], biases=gen_params["g_h%i_b" % 11])
            conv6 = self.g_batch_norm["g_bn%i" % 11](conv6)
            conv6 = tf.nn.relu(conv6, name='conv6_relu2')
            conv6 = tf.nn.dropout(conv6, .5)


            up2 = unpool3d(conv6)  # up2 = tf_utils.upsampling3d(conv6, size=(2, 2, 2), name='conv7_up')
            conv7 = tf.concat([up2, conv3], axis=4, name='conv7_concat')
            conv7 = conv3d(conv7, 4 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv7_conv1', w=gen_params["g_h%i_W" % 12], biases=gen_params["g_h%i_b" % 12])
            conv7 = self.g_batch_norm["g_bn%i" % 12](conv7)
            conv7 = tf.nn.relu(conv7, name='conv7_relu1')
            conv7 = conv3d(conv7, 4 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv7_conv2', w=gen_params["g_h%i_W" % 13], biases=gen_params["g_h%i_b" % 13])
            conv7 = self.g_batch_norm["g_bn%i" % 13](conv7)
            conv7 = tf.nn.relu(conv7, name='conv7_relu2')
            conv7 = tf.nn.dropout(conv7, .5)


            up3 = unpool3d(conv7)  # up3 = tf_utils.upsampling3d(conv7, size=(2, 2, 2), name='conv8_up')
            conv8 = tf.concat([up3, conv2], axis=4, name='conv8_concat')
            conv8 = conv3d(conv8, 2 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv8_conv1', w=gen_params["g_h%i_W" % 14], biases=gen_params["g_h%i_b" % 14])
            conv8 = self.g_batch_norm["g_bn%i" % 14](conv8)
            conv8 = tf.nn.relu(conv8, name='conv8_relu1')
            conv8 = conv3d(conv8, 2 * self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv8_conv2', w=gen_params["g_h%i_W" % 15], biases=gen_params["g_h%i_b" % 15])
            conv8 = self.g_batch_norm["g_bn%i" % 15](conv8)
            conv8 = tf.nn.relu(conv8, name='conv8_relu2')


            up4 = unpool3d(conv8)  # up4 = tf_utils.upsampling3d(conv8, size=(2, 2, 2), name='conv9_up')
            conv9 = tf.concat([up4, conv1], axis=4, name='conv9_concat')
            conv9 = conv3d(conv9, self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv9_conv1', w=gen_params["g_h%i_W" % 16], biases=gen_params["g_h%i_b" % 16])
            conv9 = self.g_batch_norm["g_bn%i" % 16](conv9)
            conv9 = tf.nn.relu(conv9, name='conv9_relu1')
            conv9 = conv3d(conv9, self.gf_dim, k_h=3, k_w=3, k_d=3, d_h=1, d_w=1, d_d=1, name='conv9_conv2', w=gen_params["g_h%i_W" % 17], biases=gen_params["g_h%i_b" % 17])
            conv9 = self.g_batch_norm["g_bn%i" % 17](conv9)
            conv9 = tf.nn.relu(conv9, name='conv9_relu2')


            # output layer: (N, 640, 640, 32) -> (N, 640, 640, 1)
            logit = conv3d(conv9, self.channel + 1, k_h=1, k_w=1, k_d=1, d_h=1, d_w=1, d_d=1, name='conv_logit', w=gen_params["g_h%i_W" % 18], biases=gen_params["g_h%i_b" % 18])
#            log_var = conv3d(conv9, 1, k_h=1, k_w=1, k_d=1, d_h=1, d_w=1, d_d=1, name='conv_logit', w=gen_params["g_h%i_W" % 19], biases=gen_params["g_h%i_b" % 19])


            return logit  #, log_var  # previous: tf.nn.sigmoid(output)


    def gen_prior(self, gen_params):

        with tf.variable_scope("generator") as scope:
            prior_loss = 0.0
            for var in gen_params.values():
                nn = tf.divide(var, self.prior_std)   # x^2 / sigma^2
                prior_loss += tf.reduce_mean(tf.multiply(nn, nn))

        prior_loss /= self.num_train

        return prior_loss


    def gen_noise(self, gen_params):

        with tf.variable_scope("generator") as scope:
            noise_loss = 0.0
            for name, var in gen_params.items():  # .iteritems():

                # Normal() returns a probability density function
                noise_ = tf.distributions.Normal(loc=0., scale=self.noise_std * tf.ones(var.get_shape()))  # tf.contrib.distributions.Normal(mu=0., sigma=self.noise_std*tf.ones(var.get_shape()))
                noise_loss += tf.reduce_sum(var * noise_.sample())   # stacked samples from the pdf

        noise_loss /= self.num_train
        return noise_loss
