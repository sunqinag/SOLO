#!/usr/bin/env python
# encoding: utf-8

'''
# @Time    : 2020/5/29 上午9:22
# @Author  : 孙强
# @Site    : 
# @File    : train.py
# @Software: PyCharm

'''

# from core.config import cfg
from branch1 import cfg
import time
from core.dataset import Dataset
import tensorflow as tf
from core.yolov3 import YOLOV3
import numpy as np
import os
import shutil
from tqdm import tqdm
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class YoloTrain:
    def __init__(self, train_image_list,
                 train_label_list,
                 val_image_list,
                 val_label_list):
        self.anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.classes = cfg.CLASSES_NAMES
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.LEARN_RATE_INIT
        self.learn_rate_end = cfg.LEARN_RATE_END
        self.first_stage_epochs = cfg.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.SECOND_STAGE_EPOCHS
        self.warmup_periods = cfg.WARMUP_EPOCHS
        self.initial_weight = None
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_log_dir = "./log"
        self.trainset = Dataset(image_list=train_image_list, label_list=train_label_list)
        self.testset = Dataset(image_list=val_image_list, label_list=val_label_list)
        self.steps_per_period = len(self.trainset)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.epoch_num = 300
        # self.train_image_list = train_image_list
        # self.train_label_list = train_label_list
        # self.val_image_list = val_image_list
        # self.val_label_list = val_label_list
        # self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        # self.classes = cfg.YOLO.CLASSES
        # self.num_classes = len(self.classes)
        # self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        # self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        # self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        # self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        # self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        # # self.initial_weight = cfg.TRAIN.INITIAL_WEIGHT
        # self.initial_weight = None
        # self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        # self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        # self.max_bbox_per_scale = 150
        # self.train_logdir = "./data/log/train"
        # self.trainset = Dataset(image_list=self.train_image_list, label_list=self.train_label_list)
        # self.testset = Dataset(image_list=self.val_image_list, label_list=self.val_label_list)
        # self.steps_per_period = len(self.trainset)
        # self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        # self.epoch_num = 30000

        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope('define_loss'):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(self.label_sbbox, self.label_mbbox,
                                                                                     self.label_lbbox,
                                                                                     self.true_sbboxes,
                                                                                     self.true_mbboxes,
                                                                                     self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_step')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step,
                                               1.0)  # 将　value 赋值给　ref，并输出 ref，即 ref = value,这是一个用来更新节点值的方法

        with tf.name_scope('define_weight_decay'):
            '''滑动平均可以看作是变量的过去一段时间取值的均值，相比对变量直接赋值而言，滑动平均得到的值在图像上更加平缓光滑，抖动性更小，不会因为某次的异常取值而使得滑动平均值波动很大
                        TensorFlow 提供了 tf.train.ExponentialMovingAverage 来实现滑动平均。在初始化 ExponentialMovingAverage 时，需要提供一个衰减率（decay），即公式(1)(2)中的 β。
                        这个衰减率将用于控制模型的更新速度。ExponentialMovingAverage 对每一个变量（variable）会维护一个影子变量（shadow_variable），这个影子变量的初始值就是相应变量的初始值，
                        而每次运行变量更新时，影子变量的值会更新为'''
            # 理解滑动平均：https://www.cnblogs.com/wuliytTaotao/p/9479958.html
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('define_first_stage_train'):
            self.first_stage_train_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_train_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                     var_list=self.first_stage_train_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5)

        with tf.name_scope('summary'):
            tf.summary.scalar('learn_rate', self.learn_rate)
            tf.summary.scalar('giou_loss', self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = './log/'
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter(logdir, graph=self.sess.graph)

    def train(self):
        test_loss = 100
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.initial_weight)
            self.loader.restore(self.sess, self.initial_weight)
        except:
            print('=> %s does not exist !!!' % self.initial_weight)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, self.epoch_num):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step], feed_dict={
                        self.input_data: train_data[0],
                        self.label_sbbox: train_data[1],
                        self.label_mbbox: train_data[2],
                        self.label_lbbox: train_data[3],
                        self.true_sbboxes: train_data[4],
                        self.true_mbboxes: train_data[5],
                        self.true_lbboxes: train_data[6],
                        self.trainable: True,
                    })

                train_epoch_loss.append(train_step_loss)
                self.summary_writer.add_summary(summary, global_step_val)
                pbar.set_description("train loss: %.2f" % train_step_loss)

            for test_data in self.testset:
                test_step_loss = self.sess.run(self.loss, feed_dict={
                    self.input_data: test_data[0],
                    self.label_sbbox: test_data[1],
                    self.label_mbbox: test_data[2],
                    self.label_lbbox: test_data[3],
                    self.true_sbboxes: test_data[4],
                    self.true_mbboxes: test_data[5],
                    self.true_lbboxes: test_data[6],
                    self.trainable: False,
                })

                test_epoch_loss.append(test_step_loss)

            if epoch % 1 == 0:
                train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
                ckpt_file = "./checkpoint/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
                log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                      % (epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
                self.saver.save(self.sess, ckpt_file, global_step=epoch)

                if np.mean(train_epoch_loss) < test_loss:
                    test_loss = np.mean(train_epoch_loss)
                    ckpt_dir = './checkpoint/best_checkpoint'
                    if not os.path.exists(ckpt_dir):
                        os.mkdir(ckpt_dir)
                    else:
                        shutil.rmtree(ckpt_dir)
                    model_name = os.path.split(ckpt_file)[-1]
                    self.saver.save(self.sess, 'checkpoint/best_checkpoint/' + model_name, global_step=epoch)


if __name__ == '__main__':
    from private_tools.file_opter import get_file_path

    train_image_dir = '../train/images/'
    train_label_dir = '../train/labels/'
    train_image_list = sorted(get_file_path(train_image_dir, ret_full_path=True))
    train_label_list = sorted(get_file_path(train_label_dir, ret_full_path=True))
    print('训练集数量为：',len(train_image_list))

    val_image_dir='../test/images/'
    val_label_dir = '../test/labels/'
    val_image_list = sorted(get_file_path(val_image_dir,ret_full_path=True))
    val_label_list = sorted(get_file_path(val_label_dir,ret_full_path=True))

    model = YoloTrain(train_image_list=train_image_list,
                      train_label_list=train_label_list,
                      val_image_list=val_image_list,
                      val_label_list=val_label_list)
    model.train()
