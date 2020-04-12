# ----------------------------
# !  Copyright(C) 2019
#   All right reserved.
#   文件名称：python_image_processing.py
#   摘   要：基于python编写的图像处理模块
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-7-8
# -----------------------------
import numpy as np


def convert_class_label_to_rgb_label(class_labels, class_num=3,
                                     label_colors=((0, 0, 0), (255, 0, 0),
                                                   (255, 165, 0), (255, 255, 0),
                                                   (0, 255, 0), (0, 127, 255),
                                                   (0, 0, 255), (139, 0, 255))):
    """
    将class的label转为彩色的label
    :param class_labels:数值型标签，输入为4维的numpy ndarray数据,[img_num,height,width,
    channel]
    :param class_num:类别数目,如果前景的类别数为ｎ，此处需加上背景，设置为n+1
    :param label_colors:颜色列表，默认是黑色代表背景，赤、橙、黄、绿、青、蓝、紫代表1到7类，
    当类别数大于8，需要更新该参数
    :return:彩色的标签图像
    """
    if not isinstance(class_labels, np.ndarray) or len(class_labels.shape) != 4:
        raise ValueError('输入必须为4维的ndarray数组！')
    if np.max(class_labels) >= class_num:
        raise ValueError('class_labels中最大类别数大于等于class_num！')

    if class_num > len(label_colors):
        raise ValueError('class_num大于label_colors参数！')

    num_images, height, width = class_labels.shape[:-1]
    outputs = np.zeros((num_images, height, width, 3), dtype=np.uint8)

    for img_index in range(num_images):
        single_class_label = class_labels[img_index]
        label_unique = np.unique(single_class_label)
        for single_class in label_unique:
            class_label_indices = np.where(single_class_label == single_class)
            outputs[img_index, class_label_indices[0], class_label_indices[1],
            :] = label_colors[single_class]
    return outputs
