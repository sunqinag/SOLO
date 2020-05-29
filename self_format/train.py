#!/usr/bin/env python
# encoding: utf-8

'''
# @Time    : 2020/5/29 上午9:22
# @Author  : 孙强
# @Site    : 
# @File    : train.py
# @Software: PyCharm

'''

import self_format.cfg as cfg
import time
from self_format.dataset import Dataset
import tensorflow as tf
from self_format.YOLOV3 import YOLOV3


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
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_log_dir = './log'
        self.trainset = Dataset(image_list=train_image_list, label_list=train_label_list)
        self.valset = Dataset(image_list=val_image_list, label_list=val_label_list)
        self.steps_per_period = len(self.trainset)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.epoch_num = 300

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
            self.loss = self.giou_loss+self.conf_loss+self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0,dtype=tf.float64,trainable=False,name='global_step')
            warmup_steps=tf.constant(self.warmup_periods*self.steps_per_period,
                                     dtype=tf.float64,name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs+self.second_stage_epochs)*self.steps_per_period,
                                      dtype=tf.float64,name='train_step')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)#将　value 赋值给　ref，并输出 ref，即 ref = value,这是一个用来更新节点值的方法

        with tf.name_scope('define_weight_decay'):
            '''滑动平均可以看作是变量的过去一段时间取值的均值，相比对变量直接赋值而言，滑动平均得到的值在图像上更加平缓光滑，抖动性更小，不会因为某次的异常取值而使得滑动平均值波动很大
                        TensorFlow 提供了 tf.train.ExponentialMovingAverage 来实现滑动平均。在初始化 ExponentialMovingAverage 时，需要提供一个衰减率（decay），即公式(1)(2)中的 β。
                        这个衰减率将用于控制模型的更新速度。ExponentialMovingAverage 对每一个变量（variable）会维护一个影子变量（shadow_variable），这个影子变量的初始值就是相应变量的初始值，
                        而每次运行变量更新时，影子变量的值会更新为'''
            # 理解滑动平均：https://www.cnblogs.com/wuliytTaotao/p/9479958.html
            moving_ave=tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope('define_first_stage_train'):
            self.first_stage_train_var_list=[]
            for var in tf.trainable_variables():
                var_name=var.op.name
                var_name_mess=str(var_name).split('/')
                if var_name_mess:
                    pass

if __name__ == '__main__':
    from private_tools.file_opter import get_file_path
    image_dir = '../train/images/'
    label_dir = '../train/labels/'
    image_list = sorted(get_file_path(image_dir,ret_full_path=True))
    label_list=sorted(get_file_path(label_dir,ret_full_path=True))

    YoloTrain(train_image_list=image_list,
              train_label_list=label_list,
              val_image_list=image_list,
              val_label_list=label_list)