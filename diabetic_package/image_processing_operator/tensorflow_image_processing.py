# ----------------------------
# !  Copyright(C) 2019
#   All right reserved.
#   文件名称：image_preprocessing.py
#   摘   要：基于tensorflow编写的图像处理基础模块
#   当前版本:1.0
#   作   者：崔宗会
#   完成日期：2019-6-27
# -----------------------------
import tensorflow as tf
import numpy as np
from scipy import misc
import cv2

COLOR_VOC = ((0, 0, 0),
             (128, 0, 0),
             (0, 128, 0),
             (128, 128, 0),
             (0, 0, 128),
             (128, 0, 128),
             (0, 128, 128),
             (128, 128, 128),
             (64, 0, 0),
             (192, 0, 0),
             (64, 128, 0),
             (192, 128, 0),
             (64, 0, 128),
             (192, 0, 128),
             (64, 128, 128),
             (192, 128, 128),
             (0, 64, 0),
             (128, 64, 0),
             (0, 192, 0),
             (128, 192, 0),
             (0, 64, 128))


def read_image_and_resize_tf(file_path, channel, image_format, height, width,
                             method, data_type=tf.float32):
    """
    利用tensorflow的函数读取图像
    :param file_path: 图像文件路径
    :param channel: 图像的通道数,灰度图为1,彩色图为3
    :param image_format: 图像的格式,目前支持'bmp','jpg','png','gif'
    :data_type:返回的图像数据的类型
    :return: 图像数据
    """
    decode_image = tf.read_file(file_path)
    image = decode_image_tf(decode_image, channel, image_format)
    image = tf.image.resize_images(image, [height, width], method=method)
    image = tf.cast(image, data_type)
    return image


def read_image_tf(file_path, channel, image_format, data_type=tf.float32):
    """
    利用tensorflow的函数读取图像
    :param file_path: 图像文件路径
    :param channel: 图像的通道数,灰度图为1,彩色图为3
    :param image_format: 图像的格式,目前支持'bmp','jpg','png','gif'
    :data_type:返回的图像数据的类型
    :return: 图像数据
    """
    decode_image = tf.read_file(file_path)
    image = decode_image_tf(decode_image, channel, image_format)
    image = tf.cast(image, data_type)
    return image


def decode_image_tf(contents, channel, image_format):
    """
    利用tensorflow的函数解码图像
    :param contents:需要解码的图像
    :param channel:图像的通道数,灰度图为1,彩色图为3
    :param image_format:图像的格式,目前支持'bmp','jpg','png','gif'
    :return:图像数据
    """
    image_format_lower = image_format.lower()
    if image_format_lower == 'bmp':
        image_decode = tf.image.decode_bmp(contents, channels=channel)
    elif image_format_lower == 'jpg' or image_format_lower == 'jpeg':
        image_decode = tf.image.decode_jpeg(contents, channels=channel)
    elif image_format_lower == 'png':
        image_decode = tf.image.decode_png(contents, channels=channel)
    elif image_format_lower == 'gif':
        image_decode = tf.image.decode_gif(contents, channels=channel)
    else:
        raise ValueError('只支持bmp,jpg,jpeg,png,gif类型的图像!')
    return image_decode


def random_crop_or_pad_image_and_label_tf(image, label, crop_height,
                                          crop_width):
    """
    使用tensorflow函数进行图像crop,如果图像尺寸小于crop尺寸进行padding
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入label,四维[b,h,w,c]的tensor或者None
    :param crop_height:crop的高
    :param crop_width:crop的宽
    :return:crop的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    image_height = tf.shape(image)[1]
    image_width = tf.shape(image)[2]
    if label is None:
        image_crop = tf.image.pad_to_bounding_box(
            image, 0, 0,
            tf.maximum(crop_height, image_height),
            tf.maximum(crop_width, image_width))

        image_crop = tf.random_crop(
            image_crop,
            [image.shape[0], crop_height, crop_width, image.shape[3]])
        return image_crop

    label = tf.to_float(label)
    image_and_label = tf.concat([image, label], axis=3)
    image_and_label_pad = tf.image.pad_to_bounding_box(
        image_and_label, 0, 0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width))
    image_and_label_crop = tf.random_crop(
        image_and_label_pad, [image_and_label_pad.shape[0],
                              crop_height, crop_width,
                              image_and_label_pad.shape[3]])

    image_crop = image_and_label_crop[:, :, :, :image.shape[3]]
    label_crop = image_and_label_crop[:, :, :, image.shape[3]:]

    return image_crop, label_crop


def flip_image_and_label_tf(image, label):
    """
    翻转图像和标签,包括上下翻转,左右翻转,对角线翻转,和不翻转
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param label:输入标签,四维[b,h,w,c]的tensor或者None
    :return:返回翻转后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    flip_index = tf.round(tf.random.uniform([], 0, 3.0))
    if label is None:
        image = tf.cond(tf.equal(flip_index, 1),
                        lambda: flip_up_down_image_and_label_tf(image,
                                                                label=None),
                        lambda: tf.cond(tf.equal(flip_index, 2),
                                        lambda: flip_left_right_image_and_label_tf(
                                            image, label=None),
                                        lambda: tf.cond(tf.equal(flip_index, 3),
                                                        lambda: transpose_image_image_and_label_tf(
                                                            image, label=None),
                                                        lambda: (image))))
        return image

    image, label = tf.cond(tf.equal(flip_index, 1),
                           lambda: flip_up_down_image_and_label_tf(image,
                                                                   label),
                           lambda: tf.cond(tf.equal(flip_index, 2),
                                           lambda: flip_left_right_image_and_label_tf(
                                               image, label),
                                           lambda: tf.cond(
                                               tf.equal(flip_index, 3),
                                               lambda: transpose_image_image_and_label_tf(
                                                   image, label),
                                               lambda: (image, label))))

    return image, label


def flip_up_down_image_and_label_tf(image, label):
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    if label is None:
        return tf.image.flip_up_down(image)

    return tf.image.flip_up_down(image), tf.image.flip_up_down(label)


def flip_left_right_image_and_label_tf(image, label):
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    if label is None:
        return tf.image.flip_left_right(image)
    return tf.image.flip_left_right(image), tf.image.flip_left_right(label)


def transpose_image_image_and_label_tf(image, label):
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    image_transpose = tf.image.transpose_image(image)
    image_transpose = tf.image.resize_bilinear(image_transpose,
                                               size=[image.shape[1],
                                                     image.shape[2]])
    if label is None:
        return image_transpose
    label_transpose = tf.image.transpose_image(label)
    label_transpose = tf.image.resize_nearest_neighbor(label_transpose,
                                                       size=[label.shape[1],
                                                             label.shape[2]])
    return image_transpose, label_transpose


def random_rescale_image_and_label_tf(image, label, min_scale=0.5, max_scale=1):
    """
    使用tensorflow函数对图像进行缩放
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label:,四维[b,h,w,c]的tensor或者None
    :param min_scale: 缩放最小系数
    :param max_scale: 缩放最大系数
    :return: 返回缩放后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    if min_scale <= 0:
        raise ValueError('\'min_scale\' must be greater than 0.')
    elif max_scale <= 0:
        raise ValueError('\'max_scale\' must be greater than 0.')
    elif min_scale >= max_scale:
        raise ValueError('\'max_scale\' must be greater than \'min_scale\'.')

    height = tf.to_float(image.shape[1])
    width = tf.to_float(image.shape[2])
    scale = tf.random_uniform(
        [], minval=min_scale, maxval=max_scale, dtype=tf.float32, seed=1)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(
        image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR)
    if label is None:
        return image

    label = tf.image.resize_images(
        label, [new_height, new_width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image, label


def rotate_image_and_label_tf(image, label, angle):
    """
    使用tensorflow函数对图像进行旋转
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的角度
    :return: 旋转后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    image_rotate = tf.contrib.image.rotate(image, angle)
    if label is None:
        return image_rotate

    label_rotate = tf.contrib.image.rotate(label, angle)
    return image_rotate, label_rotate


def random_rotate_image_and_label_tf(image, label, max_angle):
    """
    使用tensorflow函数对图像进行随机旋转
   :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的最大角度
    :return: 旋转后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    angle = tf.random_uniform(
        [], minval=-max_angle, maxval=max_angle, dtype=tf.float32, seed=1)
    return rotate_image_and_label_tf(image, label, angle)


def random_rotate_image_and_label_tf_pyfunc(image, label, max_angle):
    """
    使用tensorflow函数对图像进行旋转
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的角度
    :return: 旋转后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    def random_rotate_image_and_label_func(img, label, max_angle):
        # 旋转角度范围
        angle = np.random.uniform(low=-max_angle, high=max_angle)
        if label is None:
            return misc.imrotate(img, angle, 'bicubic'), None
        else:
            return misc.imrotate(img, angle, 'bicubic'), misc.imrotate(label,
                                                                       angle,
                                                                       'bicubic')

    image_rotate, label_rotate = tf.py_func(
        random_rotate_image_and_label_func, [image, label, max_angle],
        [tf.uint8, tf.uint8])
    if label_rotate is None:
        return image_rotate
    return image_rotate, label_rotate


def rotate_image_and_label_tf_pyfunc(image, label, angle):
    """
    使用tensorflow函数对图像进行随机旋转
   :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param angle: 旋转的最大角度
    :return: 旋转后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    def rotate_image_and_label_func(img, label, angle):
        if label is None:
            return misc.imrotate(img, angle, 'bicubic')
        else:
            return misc.imrotate(img, angle, 'bicubic'), misc.imrotate(label,
                                                                       angle,
                                                                       'bicubic')

    image_rotate, label_rotate = tf.py_func(
        rotate_image_and_label_func, [image, label, angle],
        [tf.uint8, tf.uint8])
    if label_rotate is None:
        return image_rotate
    return image_rotate, label_rotate


def translate_image_and_label_tf(image, label, dx, dy):
    """
    使用tensorflow函数对图像进行平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param dx:水平方向平移,向右为正
    :param dy:垂直方向平移,向下为正
    :return:返回平移后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    if np.abs(dx) > image.shape[2]:
        raise ValueError('dx超过了图像的宽度!')
    if np.abs(dy) > image.shape[1]:
        raise ValueError('dy超过了图像的高度!')

    image_translate = tf.contrib.image.translate(image, translations=[dx, dy])
    if label is None:
        return image_translate
    label_translate = tf.contrib.image.translate(label, translations=[dx, dy])
    return image_translate, label_translate


def random_translate_image_and_label_tf(image, label, max_dx, max_dy):
    """
    使用tensorflow函数对图像进行随机平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param max_dx:水平方向平移最大距离,向右为正
    :param max_dy:垂直方向平移最大距离,向下为正
    :return:返回平移后的图像
    """

    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    if np.abs(max_dx) > image.shape[2]:
        raise ValueError('max_dx超过了图像的宽度!')
    if np.abs(max_dy) > image.shape[1]:
        raise ValueError('max_dy超过了图像的高度!')

    dx = tf.random_uniform(
        [], minval=-max_dx, maxval=max_dx, dtype=tf.float32, seed=1)
    dy = tf.random_uniform(
        [], minval=-max_dy, maxval=max_dy, dtype=tf.float32, seed=1)
    image_translate = tf.contrib.image.translate(image, translations=[dx, dy])
    if label is None:
        return image_translate

    label_translate = tf.contrib.image.translate(label, translations=[dx, dy])
    return image_translate, label_translate


def random_translate_image_and_label_tf_pyfunc(image, label, max_dx, max_dy):
    """
    使用tensorflow函数对图像进行随机平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param max_dx:水平方向平移最大距离,向右为正
    :param max_dy:垂直方向平移最大距离,向下为正
    :return:返回平移后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    def random_translate_image_and_label_func(img, label, max_dx, max_dy):
        dx = np.random.uniform(low=0, high=max_dx)
        dy = np.random.uniform(low=0, high=max_dy)
        affine_arr = np.float32([[1, 0, dy], [0, 1, dx]])
        img = cv2.warpAffine(img, affine_arr, (img.shape[1], img.shape[0]))
        if label is None:
            return img, None
        label = cv2.warpAffine(label, affine_arr,
                               (label.shape[1], label.shape[0]))
        return img, label

    image_translate, label_translate = tf.py_func(
        random_translate_image_and_label_func, [image, label, max_dx, max_dy],
        [tf.uint8, tf.uint8])
    if label is None:
        return image_translate
    return image_translate, label_translate


def translate_image_and_label_tf_pyfunc(image, label, dx, dy):
    """
    使用tensorflow函数对图像进行平移
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param label: 输入的标签,四维[b,h,w,c]的tensor或者None
    :param dx:水平方向平移,向右为正
    :param dy:垂直方向平移,向下为正
    :return:返回平移后的图像
    """
    if len(image.shape) != 4 or (label is not None and len(label.shape) != 4):
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    def translate_image_and_label_func(img, label, dx, dy):
        affine_arr = np.float32([[1, 0, dy], [0, 1, dx]])
        img = cv2.warpAffine(img, affine_arr, (img.shape[1], img.shape[0]))
        if label is None:
            return img
        label = cv2.warpAffine(label, affine_arr,
                               (label.shape[1], label.shape[0]))
        return img, label

    image_translate, label_translate = tf.py_func(
        translate_image_and_label_func, [image, label, dx, dy],
        [tf.uint8, tf.uint8])
    if label is None:
        return image_translate
    return image_translate, label_translate


def adjust_brightness_image_tf(image, delta):
    """
    使用tensorflow函数调整图像亮度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param delta:增加的值
    :return:调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    return tf.image.adjust_brightness(image, delta)


def random_brightness_image_tf(image, max_delta, seed=None):
    """
    使用tensorflow函数随机调整图像亮度
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param max_delta: 调整的最大值
    :param seed: 种子点
    :return: 调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    return tf.image.random_brightness(image, max_delta, seed)


def adjust_contrast_image_tf(image, contrast_factor):
    """
    使用tensorflow函数调整图像对比度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param contrast_factor:调整的倍率
    :return:调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    return tf.image.adjust_contrast(image, contrast_factor)


def random_contrast_image_tf(image, lower, upper, seed=None):
    """
    使用tensorflow函数随机调整图像对比度
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param lower: 调整的最小值
    :param upper: 调整的最大值
    :param seed: 种子点
    :return: 调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    return tf.image.random_contrast(image, lower, upper, seed)


def adjust_hue_image_tf(image, delta):
    """
    使用tensorflow函数调整图像色相
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param delta:调整颜色通道的增加量
    :return:调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    if image.shape[-1] != 3:
        raise ValueError('必须传入三通道图像!')
    if delta < 0 or delta > 0.5:
        raise ValueError('delta必须为0到0.5之间的数')
    return tf.image.adjust_hue(image, delta)


def random_hue_image_tf(image, max_delta, seed=None):
    """
    使用tensorflow函数随机调整图像色相
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param max_delta: 调整颜色通道的最大随机值
    :param seed: 种子点
    :return: 调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    if image.shape[-1] != 3:
        raise ValueError('必须传入三通道图像!')
    return tf.image.random_hue(image, max_delta, seed)


def adjust_saturation_image_tf(image, saturation_factor, name=None):
    """
    使用tensorflow函数调整图像饱和度
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param saturation_factor:调整的倍率
    :return:调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    if image.shape[-1] != 3:
        raise ValueError('必须传入三通道图像!')
    return tf.image.adjust_saturation(image, saturation_factor, name)


def random_saturation_image_tf(image, lower, upper, seed=None):
    """
    使用tensorflow函数随机调整图像饱和度
    :param lower: 调整的最小值
    :param upper: 调整的最大值
    :param seed: 种子点
    :return: 调整后的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    if image.shape[-1] != 3:
        raise ValueError('必须传入三通道图像!')
    return tf.image.random_saturation(image, lower, upper, seed)


def per_image_standardization_tf(image):
    """
    图像标准化
    :param image:输入图像,四维[b,h,w,c]的tensor
    :return:
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    if image.shape[-1] != 3:
        raise ValueError('必须传入三通道图像!')

    img = tf.reshape(image, [image.shape[1], image.shape[2], image.shape[3]])
    img = tf.image.per_image_standardization(img)
    return tf.reshape(img, [-1, img.shape[0], img.shape[1], img.shape[2]])


def add_gaussian_noise(image, mean, std):
    """
    增加高斯噪声
    :param image:输入图像,四维[b,h,w,c]的tensor
    :param mean: 噪声均值
    :param std: 噪声标准差
    :return: 添加噪声的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    noise = tf.random_normal(shape=tf.shape(image), mean=mean, stddev=std,
                             dtype=tf.float32)
    image = tf.add(image, noise)
    # image = tf.cast(image, tf.uint8)
    return image


def add_random_noise(image, minval, maxval):
    """
    增加随机噪声
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param minval:噪声的最小值
    :param maxval:噪声的最大值
    :return:添加噪声的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')
    noise = tf.random.uniform(tf.shape(image), minval=minval, maxval=maxval)
    image = tf.add(image, noise)
    # image = tf.cast(image, tf.uint8)
    return image


def add_salt_and_pepper_noise_pyfunc(image, scale, value=255):
    """
    图像添加椒盐噪声
    :param image: 输入图像,四维[b,h,w,c]的tensor
    :param scale: 噪声的比例系数
    :param value: 噪声的值
    :return: 添加噪声的图像
    """
    if len(image.shape) != 4:
        raise ValueError('传入的image必须是四维的tensor,label必须是四维的tensor或者None!')

    def add_salt_and_pepper_noise_func(img, height, width, scale):
        img_copy = img.copy()
        noise_num = np.int32(np.round(scale * height * width))
        random_coordinate = np.array(
            [[np.random.randint(height), np.random.randint(width)] for i in
             range(noise_num)])
        img_copy[:, random_coordinate[:, 0], random_coordinate[:, 1], :] = value
        return img_copy

    height = tf.to_float(image.shape[1])
    width = tf.to_float(image.shape[2])

    image_noise = tf.py_func(
        add_salt_and_pepper_noise_func, [image, height, width, scale],
        [tf.float32])
    # image_noise = tf.cast(image_noise, tf.uint8)
    return image_noise


def convert_class_label_to_rgb_label_pyfunc(class_labels, class_num=21,
                                            label_colors=COLOR_VOC):
    def convert_class_label_to_rgb_label(class_labels, class_num, label_colors):
        if not isinstance(class_labels, np.ndarray) or len(
                class_labels.shape) != 4:
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
                class_label_indices = np.where(
                    single_class_label == single_class)
                outputs[img_index, class_label_indices[0],
                class_label_indices[1], :] = label_colors[single_class]
        return outputs

    rgb_label = tf.py_func(
        convert_class_label_to_rgb_label,
        [class_labels, class_num, label_colors], [tf.uint8])
    return rgb_label
