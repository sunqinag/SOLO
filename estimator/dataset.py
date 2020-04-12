#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
# ================================================================

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
from core.config import cfg
from tqdm import tqdm


class Dataset(object):
    """implement Dataset here"""

    def __init__(self, dataset_type):
        self.annot_path = cfg.TRAIN.ANNOT_PATH if dataset_type == 'train' else cfg.TEST.ANNOT_PATH
        self.input_sizes = cfg.TRAIN.INPUT_SIZE if dataset_type == 'train' else cfg.TEST.INPUT_SIZE
        self.batch_size = cfg.TRAIN.BATCH_SIZE if dataset_type == 'train' else cfg.TEST.BATCH_SIZE
        self.data_aug = cfg.TRAIN.DATA_AUG if dataset_type == 'train' else cfg.TEST.DATA_AUG

        self.train_input_sizes = cfg.TRAIN.INPUT_SIZE
        self.strides = np.array(cfg.YOLO.STRIDES)
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(dataset_type)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

        self.tfrecord_name = '../data/' + str(dataset_type) + '.tfrecords'

    # 获取图像信息列表
    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)  # 随机打乱
        return annotations

    def __iter__(self):
        return self

    def create_tfrecord(self):
        self.train_input_size = random.choice(self.train_input_sizes)  # 随机选择一个训练图像的尺寸
        self.train_output_sizes = self.train_input_size // self.strides  # 图像尺寸除以缩放倍率，得到输出图像尺寸  【52,26,13】3个尺寸
        write = tf.python_io.TFRecordWriter(self.tfrecord_name)
        for annotation in tqdm(self.annotations):
            image, bboxes = self.parse_annotation(annotation)  # 解析annotation，获取图像数据和标记框（做一些预处理）
            # 对标记框做一些预处理
            label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(
                bboxes)
            print('image shape:', image.shape)
            print('label sbbox shape:', label_sbbox.shape)
            print('label mbbox shape:', label_mbbox.shape)
            print('label lbbox shape:', label_lbbox.shape)
            print('sbboxes shape:', sbboxes.shape)
            print('mbboxes shape:', mbboxes.shape)
            print('lbboxes shape:', lbboxes.shape)

            image = image.tostring()
            label_sbbox = label_sbbox.tostring()
            label_mbbox = label_mbbox.tostring()
            label_lbbox = label_lbbox.tostring()
            sbboxes = sbboxes.tostring()
            mbboxes = mbboxes.tostring()
            lbboxes = lbboxes.tostring()

            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image])),
                        'label_sbbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_sbbox])),
                        'label_mbbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_mbbox])),
                        'label_lbbox': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_lbbox])),
                        'sbboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[sbboxes])),
                        'mbboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[mbboxes])),
                        'lbboxes': tf.train.Feature(bytes_list=tf.train.BytesList(value=[lbboxes])),
                    }
                )
            )
            write.write(example.SerializeToString())
        write.close()
        print('tfrecord制作完毕！')

    def pareser(self, serialized):
        features = tf.parse_single_example(serialized,
                                    features={
                                        'image': tf.FixedLenFeature([], tf.string),
                                        'label_sbbox': tf.FixedLenFeature([], tf.string),
                                        'label_mbbox': tf.FixedLenFeature([], tf.string),
                                        'label_lbbox': tf.FixedLenFeature([], tf.string),
                                        'sbboxes': tf.FixedLenFeature([], tf.string),
                                        'mbboxes': tf.FixedLenFeature([], tf.string),
                                        'lbboxes': tf.FixedLenFeature([], tf.string),
                                    })

        # 将字符串解析成对应的像素数组
        image = tf.decode_raw(features['image'], tf.float32)
        image = tf.reshape(image, [320, 320, 3])
        label_sbbox = tf.decode_raw(features['label_sbbox'], tf.float32)
        label_mbbox = tf.decode_raw(features['label_mbbox'], tf.float32)
        label_lbbox = tf.decode_raw(features['label_lbbox'], tf.float32)

        sbboxes = tf.decode_raw(features['sbboxes'], tf.float32)
        mbboxes = tf.decode_raw(features['mbboxes'], tf.float32)
        lbboxes = tf.decode_raw(features['lbboxes'], tf.float32)

        label_sbbox = tf.reshape(label_sbbox, [40, 40, 3, 25])
        label_mbbox = tf.reshape(label_mbbox, [20, 20, 3, 25])
        label_lbbox = tf.reshape(label_lbbox, [10, 10, 3, 25])

        sbboxes = tf.reshape(sbboxes, [150, 4])
        mbboxes = tf.reshape(mbboxes, [150, 4])
        lbboxes = tf.reshape(lbboxes, [150, 4])

        return image,label_sbbox,label_mbbox,label_lbbox,sbboxes,mbboxes,lbboxes

    def reload_tfrecord(self):
        dataset = tf.data.TFRecordDataset(r'D:\Pycharm_Project\SOLO\data\train.tfrecords')
        dataset = dataset.map(self.pareser)
        dataset = dataset.shuffle(buffer_size=10000)
        dataset = dataset.batch(batch_size=self.batch_size)
        dataset = dataset.repeat(cfg.TRAIN.EPOCH_NUM)

        iteror = dataset.make_one_shot_iterator()
        image, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = iteror.get_next()
        return image,label_sbbox,label_mbbox,label_lbbox,sbboxes,mbboxes,lbboxes

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]
            bboxes[:, [0, 2]] = w - bboxes[:, [2, 0]]

        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin: crop_ymax, crop_xmin: crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

        return image, bboxes

    def random_translate(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

        return image, bboxes

    '''
    根据列表信息读取图像以及获取该图像标记框信息
    annotation="./VOC2007/JPEGImages/000002.jpg 139,200,207,301,18"
    '''

    def parse_annotation(self, annotation):

        line = annotation.split()  # 分解当前图像路径
        image_path = line[0]  # 图像路径
        if not os.path.exists(image_path):  # 判断路径是否存在
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))  # 读取图像
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]])  # 分解出标记框的信息

        if self.data_aug:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))  # 随机水平翻转
            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))  # 随机裁剪图像
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))  # 随机移动图像

        # 将输入图像按训练选取的图像尺寸进行缩放，空白部分用灰度值128填充
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size],
                                               np.copy(bboxes))
        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                 boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                 boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    '''
    标记框预处理
    '''

    def preprocess_true_boxes(self, bboxes):
        #  train_output_sizes[i] =》 52 ，26，13三个值
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]  # 3,150,4
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            bbox_coor = bbox[:4]  # 取x1,y1,x2,y2
            bbox_class_ind = bbox[4]  # 取类别序号

            onehot = np.zeros(self.num_classes, dtype=np.float)  # onehot  [0,0,1,0,0...]
            onehot[bbox_class_ind] = 1.0  # 将类别索引出置为1
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)  # num_classes维，每维以类别总数的倒数填充
            deta = 0.01
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution

            # 根据x1,y1,x2,y2计算得出x,y,w,h，（x,y）为矩形框中心点坐标，相对于下采样前图的坐标
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # 除以下采样率，对应到特征图上的坐标，包含小中大三个尺寸信息
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]

            iou = []
            exist_positive = False  # 存在标记框
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))  # anchor_per_scale每个框产生几个anchor
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # np.floor()向下取整
                anchors_xywh[:, 2:4] = self.anchors[i]  # 获取基准anchor的宽高

                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # 计算缩放后的GT与anchor框的IOU
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3  # 大于0.3为1，否则为0

                if np.any(iou_mask):  # 有1个非0
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)  # 标记框中心坐标

                    # # 减少数据敏感性， 9月19号添加，可以打开，不影响结果。
                    xind, yind = abs(xind), abs(yind)
                    if yind >= label[i].shape[1]:  # shape[1] 13，26,52
                        yind = label[i].shape[1] - 1
                    if xind >= label[i].shape[0]:  # shape[0] 13，26,52
                        xind = label[i].shape[0] - 1

                    label[i][yind, xind, iou_mask, :] = 0  # 先初始化为0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh  # 标记框的坐标信息
                    label[i][yind, xind, iou_mask, 4:5] = 1.0  # 置信度
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot  # 分类概率

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh  # 第i个box的标记框
                    bbox_count[i] += 1

                    exist_positive = True

            if not exist_positive:  # 没有标记框，找iou最大值
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)  # 找iou最大值
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                # 减少数据敏感性 9月19号添加
                xind, yind = abs(xind), abs(yind)
                if yind >= label[best_detect].shape[1]:
                    yind = label[best_detect].shape[1] - 1
                if xind >= label[best_detect].shape[0]:
                    xind = label[best_detect].shape[0] - 1

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh  # 标记框坐标
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0  # 置信度
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot  # 分类概率

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1
        label_sbbox, label_mbbox, label_lbbox = label  # 获取小中大标记框的标签
        sbboxes, mbboxes, lbboxes = bboxes_xywh  # 获取小中大标记框的坐标值
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __len__(self):
        return self.num_batchs


if __name__ == '__main__':
    print("当前路径 -> %s" % os.getcwd())

    current_path = os.path.dirname(__file__)
    print('正确路径：', current_path)

    # Dataset('train').create_tfrecord()
    image,label_sbbox,label_mbbox,label_lbbox,sbboxes,mbboxes,lbboxes=Dataset('train').reload_tfrecord()
