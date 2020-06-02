#!/usr/bin/env python
# encoding: utf-8

'''
# @Time    : 2020/5/27 上午10:54
# @Author  : 孙强
# @Site    : 
# @File    : dataset.py
# @Software: PyCharm

'''
import cv2
import os
import numpy as np
from private_tools.diabetic_package.file_operator.bz_path import get_file_path
from private_tools.imgFileOpterator import Img_processing as process
from private_tools.diabetic_package.image_processing_operator.python_data_augmentation.python_base_data_augmentation import resize_image_and_label
import branch1.cfg as cfg
import tensorflow as tf

class Dataset:
    def __init__(self, image_list, label_list):
        self.image_list=image_list
        self.label_list=label_list
        self.max_bbox_per_scale = 150
        self.num_sample = len(self.image_list)
        self.batch_size = cfg.BATCH_SIZE
        self.anchor_per_scale = cfg.ANCHOR_PER_SCALE
        self.classes_names = cfg.CLASSES_NAMES
        self.num_classes = len(self.classes_names)
        self.stride = np.array(cfg.STRIDES)
        self.anchors = np.array(cfg.ANCHORS).reshape(3, 3, 2)
        self.num_batchs = int(np.ceil(self.num_sample / self.batch_size))
        self.batch_count = 0

    def __iter__(self):
        return self

    def __next__(self):
        '''该函数最终返回的是：按batch_size大小返回图像、大中小标记框的标签及坐标信息。'''
        with tf.device('/cpu:0'):
            self.train_input_size = cfg.INPUT_SIZE
            self.output_size = np.array(self.train_input_size) // np.array(cfg.STRIDES)

            # batch_image [batch_size,input_size[0],input_size[1],3]
            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))

            # batch_size*52*52*ANCHOR_PER_SCALE*class_num+1
            batch_label_sbbox = np.zeros((self.batch_size, self.output_size[0], self.output_size[0], self.anchor_per_scale, 5 + self.num_classes))
            # batch_size*26*26*ANCHOR_PER_SCALE*class_num+1
            batch_label_mbbox = np.zeros((self.batch_size, self.output_size[1], self.output_size[1], self.anchor_per_scale, 5 + self.num_classes))
            # batch_size*13*13*ANCHOR_PER_SCALE*class_num+1
            batch_label_lbbox = np.zeros((self.batch_size, self.output_size[2], self.output_size[2], self.anchor_per_scale, 5 + self.num_classes))

            ## batch_size*150*4
            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num  # 计算获取图像的索引
                    if index >= self.num_sample: index -= self.num_sample  # 如果索引超出了图像数量，则减去图像数量，从头开始
                    #这里用的还是原图尺寸
                    image = cv2.imread(self.image_list[index])
                    # print('图像：',os.path.split(self.image_list[index])[-1])
                    bboxes = process().parseBoxAndLabel(self.label_list[index])  # 获得原始的图像和标记框
                    #要将原图尺寸变为416*416尺寸
                    resized_image,reized_bboxes = resize_image_and_label(image,bboxes,(cfg.INPUT_SIZE,cfg.INPUT_SIZE))
                    # bboxes = process().parseBoxAndLabel('../train/labels/20200513150210348-E960G7.txt')
                    # self._view_image_and_box(resized_image,reized_bboxes)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self._processing(bboxes)

                    batch_image[num, :, :, :] = resized_image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox

                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, \
                       batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                raise StopIteration #结束

    def _view_image_and_box(self,origimg,box):

        for i in range(len(box)):
            p1 = (box[i][0], box[i][1])
            p2 = (box[i][2], box[i][3])
            cv2.rectangle(origimg, p1, p2, (0, 255, 0))
            p3 = (max(p1[0], 15), max(p1[1], 15))
            title = "%s:%d" % ('label::', box[i][-1])
            cv2.putText(origimg, title, p3, cv2.FONT_ITALIC, 0.6, (0, 255, 0), 1)
        cv2.imshow("view", origimg)
        cv2.waitKey(0)


    def _processing(self, bboxes):
        '''将原始尺寸的图像和box放缩到三种尺度上并返回'''
        label = [np.zeros((self.output_size[i], self.output_size[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)]  # 3,150,4
        bbox_count = np.zeros((3,))

        for bbox in bboxes:
            box_coor = bbox[:4]  # 取x1,y1,x2,y2
            bbox_class_id = bbox[4]  # 取类别序号

            # 将类别做onehot编码
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[bbox_class_id] = 1.0  # 将类别索引出置为1
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)  # num_classes维，每维以类别总数的倒数填充
            # 这是嫌弃onehot不够平滑吗？？？？
            deta = 0.01
            smooth_onehot = one_hot * (1 - deta) + deta + uniform_distribution

            # 根据x1,y1,x2,y2计算得出x,y,w,h，（x,y）为矩形框中心点坐标，相对于下采样前图的坐标
            bbox_xywh = np.concatenate(((box_coor[:2] + box_coor[2:]) * 0.5, box_coor[2:] - box_coor[:2]), axis=-1)
            # 除以下采样率，对应到特征图上的坐标，包含小中大三个尺寸信息
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.stride[:, np.newaxis]

            iou = []
            exist_positive = False  # 存在标记框
            for i in range(3):
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))  # anchor_per_scale每个框产生几个anchor
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5  # np.floor()向下取整
                # 这里三个尺度的xy都是原图坐标而不是下采样下来的坐标 70
                anchors_xywh[:, 2:4] = self.anchors[i]  # 获取基准anchor的宽高.这是一个混合体，由中心点坐标和anchor尺寸拼接而成
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # 计算缩放后的GT与anchor框的IOU
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3
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

    def __len__(self):
        return self.num_batchs

if __name__ == '__main__':
    from tqdm import tqdm
    image_dir = '../train/images'
    label_dir = '../train/labels'
    image_list = sorted(get_file_path(image_dir, ret_full_path=True))
    label_list = sorted(get_file_path(label_dir, ret_full_path=True))
    train = Dataset(image_list=image_list, label_list=label_list)
    pbar = tqdm(train)
    for i in range(100):
        for train_data in pbar:
            pass
