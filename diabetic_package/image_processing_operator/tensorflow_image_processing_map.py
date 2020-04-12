# ----------------------------
# !  Copyright(C) 2019
#   All right reserved.
#   文件名称：xxx.py
#   摘   要：xxx
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-x-x
# -----------------------------
import tensorflow as tf
from functools import partial
from ..image_processing_operator import tensorflow_image_processing
import numpy as np


def cond_fun(flip_index, image, label, function_true):
    if label is None:
        image = tf.cond(
            tf.equal(flip_index, 1), lambda: function_true(image, label=None),
            lambda: image)
        return image, None
    image, label = tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image, label=label),
        lambda: (image, label))
    return image, label


def random_crop_or_pad_image_and_label_tf_map(image, label, crop_height,
                                              crop_width, rate=0.25):
    """
    使用tensorflow函数以50%概率进行图像进行缩放
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label:,四维[b,h,w,c]的tensor或者None
    :param min_scale: 缩放最小系数
    :param max_scale: 缩放最大系数
    :param rate:进行该操作的概率
    :return: 返回缩放后的图像
    """

    def function_false(image, label):
        image = tf.image.resize_bilinear(image, size=[crop_height, crop_width])
        if label is None:
            return image
        label = tf.image.resize_nearest_neighbor(label,
                                                 size=[crop_height, crop_width])
        return image, label

    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_crop_or_pad_image_and_label_tf,
        crop_width=crop_width, crop_height=crop_height)
    result = tf.cond(tf.equal(flip_index, 1),
                     lambda: function_true(image, label),
                     lambda: function_false(image, label))
    if label is None:
        return result, None
    return result


def flip_image_and_label_tf_map(image, label):
    """
    以50%概率进行翻转图像和标签,包括上下翻转,左右翻转,对角线翻转,和不翻转
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入标签,四维[b,h,w,c]的tensor或者None
    :param rate:进行该操作的概率
    :return:返回翻转后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    flip_index = tf.floor(tf.random.uniform([], 0.0, 5.0))
    if label is None:
        image = tf.cond(tf.equal(flip_index, 1),
                        lambda: tensorflow_image_processing.flip_up_down_image_and_label_tf(
                            image, label=None),
                        lambda: tf.cond(tf.equal(flip_index, 2),
                                        lambda: tensorflow_image_processing.flip_left_right_image_and_label_tf(
                                            image, label=None),
                                        lambda: tf.cond(tf.equal(flip_index, 3),
                                                        lambda: tensorflow_image_processing.transpose_image_image_and_label_tf(
                                                            image, label=None),
                                                        lambda: (image))))
        return image, None

    image, label = tf.cond(tf.equal(flip_index, 1),
                           lambda: tensorflow_image_processing.flip_up_down_image_and_label_tf(
                               image, label),
                           lambda: tf.cond(tf.equal(flip_index, 2),
                                           lambda: tensorflow_image_processing.flip_left_right_image_and_label_tf(
                                               image, label),
                                           lambda: tf.cond(
                                               tf.equal(flip_index, 3),
                                               lambda: tensorflow_image_processing.transpose_image_image_and_label_tf(
                                                   image, label),
                                               lambda: (image, label))))

    return image, label


def flip_up_down_image_and_label_tf_map(image, label, rate=0.25):
    """以50%概率进行上下翻转"""
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.flip_up_down_image_and_label_tf
    return cond_fun(flip_index, image, label, function_true)


def flip_left_right_image_and_label_tf_map(image, label, rate=0.25):
    """以50%概率进行左右翻转"""
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.flip_left_right_image_and_label_tf
    return cond_fun(flip_index, image, label, function_true)


def transpose_image_image_and_label_tf_map(image, label, rate=0.25):
    """以50%概率进行对角线翻转"""
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.transpose_image_image_and_label_tf
    return cond_fun(flip_index, image, label, function_true)


def random_rescale_image_and_label_tf_map(image, label, min_scale=0.5,
                                          max_scale=1, rate=0.25):
    """
    使用tensorflow函数以50%概率进行图像进行缩放
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label:,四维[b,h,w,c]的tensor或者None
    :param min_scale: 缩放最小系数
    :param max_scale: 缩放最大系数
    :param rate:进行该操作的概率
    :return: 返回缩放后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_rescale_image_and_label_tf,
        min_scale=min_scale,
        max_scale=max_scale)
    return cond_fun(flip_index, image, label, function_true)


def rotate_image_and_label_tf_map(image, label, angle, rate=0.25):
    """
    使用tensorflow函数以50%概率进行图像进行旋转
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的角度
    :param rate:进行该操作的概率
    :return: 旋转后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.rotate_image_and_label_tf, angle=angle)
    return cond_fun(flip_index, image, label, function_true)


def random_rotate_image_and_label_tf_map(image, label, max_angle, rate=0.25):
    """
    使用tensorflow函数以50%概率进行图像进行随机旋转
   :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的最大角度
    :param rate:进行该操作的概率
    :return: 旋转后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_rotate_image_and_label_tf,
        max_angle=max_angle)
    return cond_fun(flip_index, image, label, function_true)


def translate_image_and_label_tf_map(image, label, dx, dy, rate=0.25):
    """
    使用tensorflow函数以50%概率进行图像进行平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param dx:水平方向平移,向右为正
    :param dy:垂直方向平移,向下为正
    :param rate:进行该操作的概率
    :return:返回平移后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.translate_image_and_label_tf, dx=dx, dy=dy)
    return cond_fun(flip_index, image, label, function_true)


def random_translate_image_and_label_tf_map(image, label, max_dx, max_dy,
                                            rate=0.25):
    """
    使用tensorflow函数以50%概率进行图像随机平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param max_dx:水平方向平移最大距离,向右为正
    :param max_dy:垂直方向平移最大距离,向下为正
    :param rate:进行该操作的概率
    :return:返回平移后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_translate_image_and_label_tf,
        max_dx=max_dx, max_dy=max_dy)
    return cond_fun(flip_index, image, label, function_true)


def adjust_brightness_image_tf_map(image, delta, rate=0.25):
    """
    使用tensorflow函数以50%概率调整图像亮度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param delta:增加的值
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.adjust_brightness_image_tf, delta=delta)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def random_brightness_image_tf_map(image, max_delta, rate=0.25):
    """
    使用tensorflow函数以50%概率随机调整图像亮度
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param max_delta: 调整的最大值
    :param seed: 种子点
    :param rate:进行该操作的概率
    :return: 调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_brightness_image_tf,
        max_delta=max_delta)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def adjust_contrast_image_tf_map(image, contrast_factor, rate=0.25):
    """
    使用tensorflow函数以50%概率调整图像对比度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param contrast_factor:调整的倍率
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.adjust_contrast_image_tf,
        contrast_factor=contrast_factor)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def random_contrast_image_tf_map(image, lower, upper, seed=None, rate=0.25):
    """
      使用tensorflow函数以50%概率随机调整图像对比度
      :param image: 输入图像,四维[b,h,w,c]的tensor
      :param lower: 调整的最小值
      :param upper: 调整的最大值
      :param seed: 种子点
      :param rate:进行该操作的概率
      :return: 调整后的图像
      """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_contrast_image_tf, lower=lower,
        upper=upper, seed=seed)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def adjust_hue_image_tf_map(image, delta, rate=0.25):
    """
    使用tensorflow函数以50%概率调整图像色相
    :param image:输入图像,四维[b,h,w,c]的tensor,c必须为3
    :param delta:调整颜色通道的增加量
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.adjust_hue_image_tf,
                            delta=delta)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def random_hue_image_tf_map(image, max_delta, rate=0.25):
    """
    使用tensorflow函数以50%概率随机调整图像色相
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param max_delta: 调整颜色通道的最大随机值
    :param seed: 种子点
    :param rate:进行该操作的概率
    :return: 调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.random_hue_image_tf,
                            max_delta=max_delta)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def adjust_saturation_image_tf_map(image, saturation_factor, rate=0.25):
    """
    使用tensorflow函数以50%概率调整图像饱和度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param saturation_factor:调整的倍率
    :param rate:进行该操作的概率
    :return:调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.adjust_saturation_image_tf,
        saturation_factor=saturation_factor)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def random_saturation_image_tf_map(image, lower, upper, seed=None, rate=0.25):
    """
    使用tensorflow函数以50%概率随机调整图像饱和度
    :param lower: 调整的最小值
    :param upper: 调整的最大值
    :param seed: 种子点
    :param rate:进行该操作的概率
    :return: 调整后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.random_saturation_image_tf, lower=lower,
        upper=upper, seed=seed)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def per_image_standardization_tf_map(image, rate=0.25):
    """
    以50%概率进行图像标准化
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param rate:进行该操作的概率
    :return:标准化后的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = tensorflow_image_processing.per_image_standardization_tf

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def add_gaussian_noise_tf_map(image, mean, std, rate=0.25):
    """
    以50%概率增加高斯噪声
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param mean: 噪声均值
    :param std: 噪声标准差
    :param rate:进行该操作的概率
    :return: 添加噪声的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.add_gaussian_noise,
                            mean=mean, std=std)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def add_random_noise_tf_map(image, minval, maxval, rate=0.25):
    """
    以50%概率增加随机噪声
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param minval:噪声的最小值
    :param maxval:噪声的最大值
    :param rate:进行该操作的概率
    :return:添加噪声的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(tensorflow_image_processing.add_random_noise,
                            minval=minval, maxval=maxval)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)


def add_salt_and_pepper_noise_pyfunc_tf_map(image, scale, value, rate=0.25):
    """
    以50%概率图像添加椒盐噪声
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param scale: 噪声的比例系数
    :param value: 噪声的值
    :param rate:进行该操作的概率
    :return: 添加噪声的图像
    """
    flip_index = tf.floor(tf.random.uniform([], 0.0, np.round(1.0 / rate)))
    function_true = partial(
        tensorflow_image_processing.add_salt_and_pepper_noise_pyfunc,
        scale=scale, value=value)

    return tf.cond(
        tf.equal(flip_index, 1), lambda: function_true(image), lambda: image)
