# ---------------------------------
#   !Copyright(C) 2018,北京博众
#   All right reserved.
#   文件名称：data augmentation
#   摘   要：对图像数据进行增强，包括随机旋转，随机裁剪，随机噪声，平移变换，图像反转
#   当前版本:1.0
#   作   者：陈瑞侠
#   完成日期：2018-06-8
# ---------------------------------


import cv2
import numpy as np
from skimage import util


def rotate(image, label=None, center=None, min_angle=0,
           max_angle=360, rotate_angle=None):
    '''图像旋转,angle正负值都可以，正负值代表顺逆时针'''
    height, width = image.shape[:2]
    if rotate_angle is None:
        rotate_angle = np.random.randint(min_angle, max_angle)
    if center is None:
        center = (height / 2, width / 2)
    rotation_matrix2d = cv2.getRotationMatrix2D(center, rotate_angle, scale=1)
    rotated_img = cv2.warpAffine(image, rotation_matrix2d, (width, height))
    if label is not None:
        if image.shape[: 2] != label.shape[: 2]:
            raise ValueError('输入参数image, label的shape不匹配!')
        label = cv2.warpAffine(label, rotation_matrix2d, (width, height))
    return rotated_img, label


def shear_transformation(image, label=None, min=-1,
                         max=1, scale1=None, scale2=None):
    height, width = image.shape[:2]
    rotation_matrix2d = np.ones((2, 3))
    if scale1 is None:
        scale1 = np.random.uniform(min, max)
    if scale2 is None:
        scale2 = np.random.uniform(min, max)
    rotation_matrix2d[0, 1] = scale1
    rotation_matrix2d[1, 0] = scale2
    deltx = -rotation_matrix2d[0, 1] * height / 2
    delty = - rotation_matrix2d[1, 0] * width / 2

    rotation_matrix2d[0, 2] = deltx
    rotation_matrix2d[1, 2] = delty

    transformed_img = cv2.warpAffine(image, rotation_matrix2d, (width, height))
    if label is not None:
        if image.shape[: 2] != label.shape[: 2]:
            raise ValueError('输入参数image, label的shape不匹配!')
        label = cv2.warpAffine(label, rotation_matrix2d, (width, height))
    return transformed_img, label


def scale(image, min_scale, max_scale, scale=None, label=None,
          interpolation=cv2.INTER_LINEAR):
    '''图像缩放'''
    height, width = image.shape[:2]
    if scale is None:
        scale = np.random.uniform(min_scale, max_scale)
    height_n = int(height * scale)
    width_n = int(width * scale)  # 是否加1根据图像的尺寸的奇偶决定
    scaled_img = cv2.resize(
        image, (width_n, height_n),
        interpolation=interpolation)
    if label is not None:
        if image.shape[: 2] != label.shape[: 2]:
            raise ValueError('输入参数image, label的shape不匹配!')
        label = cv2.resize(
            label, (width_n, height_n),
            interpolation=interpolation)
    return scaled_img, label


def flip(image, flip_num=None, label=None):
    '''图像反转（类似镜像，镜子中的自己'''
    if flip_num is None:
        num = np.random.randint(-1, 1)
    else:
        num = flip_num
    flipped_img = image
    flipped_label = label
    # 横向翻转图像
    if num == 1:
        flipped_img = cv2.flip(image, 1)
        if label is not None:
            if image.shape[: 2] != label.shape[: 2]:
                raise ValueError('输入参数image, label的shape不匹配!')
        flipped_label = cv2.flip(flipped_label, 1)
    # 纵向翻转图像
    if num == 0:
        flipped_img = cv2.flip(image, 0)
        if label is not None:
            if image.shape[: 2] != label.shape[: 2]:
                raise ValueError('输入参数image, label的shape不匹配!')
        flipped_label = cv2.flip(flipped_label, 0)
    # 同时在横向和纵向翻转图像
    if num == -1:
        flipped_img = cv2.flip(image, -1)
        if label is not None:
            if image.shape[: 2] != label.shape[: 2]:
                raise ValueError('输入参数image, label的shape不匹配!')
        flipped_label = cv2.flip(flipped_label, -1)
    return flipped_img, flipped_label


def crop(image, crop_size=None, label=None):
    '''图像随机裁剪,考虑到图像大小为(w,h),使用一个大于(w/2,h/2)的窗口进行裁剪'''
    height, width = image.shape[:2]
    crop_max_size = min(height, width)
    if crop_size is None:
        crop_size = np.random.randint(crop_max_size / 2, crop_max_size)
    if crop_size > crop_max_size:
        crop_size = crop_max_size
    roi_min_col = int((width - crop_size) / 2)
    roi_max_col = int((width + crop_size) / 2)
    roi_min_row = int((height - crop_size) / 2)
    roi_max_row = int((height + crop_size) / 2)
    crop_img = image[roi_min_row: roi_max_row, roi_min_col: roi_max_col]
    if label is not None:
        if image.shape[: 2] != label.shape[: 2]:
            raise ValueError('输入参数image, label的shape不匹配!')
        label = label[roi_min_row: roi_max_row, roi_min_col: roi_max_col]
    return crop_img, label


def add_noise(image, mode_list, mean, var):
    '''函数作用:给图片加入噪声
        参数说明：
                mode_list:传入的是要添加的噪声mode的列表
                mode可以选择'gaussian','poisson',
                'salt','pepper','speckle','s&p'其中任意一种噪声
        mean:噪声随机分布的均值
        var:噪声随机分布的方差
    '''
    result = []
    noise_img = image
    for mode in mode_list:
        if mode == 'gaussian' or mode == 'speckle':
            noise_img = util.random_noise(image, mode=mode, mean=mean, var=var)
        if mode == 'salt' or mode == 'pepper' or mode == 'salt & pepper':
            noise_img = util.random_noise(image, mode=mode, amount=var)
        if mode == 's&p':
            noise_img = util.random_noise(image, mode=mode, salt_vs_pepper=var)
        result.append(noise_img)
    return result


def stauration_noise(image, hsv_list=((0.6, 0.6), (0.6, 0.7), (0.4, 0.8))):
    '''加入饱和度光照噪声'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 增加饱和度光照的噪声
    hsv[:, :, 0] = hsv[:, :, 0] * (hsv_list[0]
                                   [0] + np.random.random() * hsv_list[0][1])
    hsv[:, :, 1] = hsv[:, :, 1] * (hsv_list[1]
                                   [0] + np.random.random() * hsv_list[1][1])
    hsv[:, :, 2] = hsv[:, :, 2] * (hsv_list[2]
                                   [0] + np.random.random() * hsv_list[2][1])
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


def translation(image, max_dist, dist=None, label=None):
    '''图像平移变换'''
    if dist is None:
        dist = np.random.randint(-max_dist, max_dist)
    Matrix = np.float32([[1, 0, dist], [0, 1, dist]])
    shifted_img = cv2.warpAffine(
        image, Matrix, (image.shape[1], image.shape[0]))
    if label is not None:
        if image.shape[: 2] != label.shape[: 2]:
            raise ValueError('输入参数image, label的shape不匹配!')
        label = cv2.warpAffine(label, Matrix, (label.shape[1], label.shape[0]))
    return shifted_img, label


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    image_path = \
        '/home/crx/crx/data/dr11-200/000009078_20140605100247_900414_34133.jpg'
    image = cv2.imread(image_path)

    plt.subplot(211)
    plt.imshow(image)
    plt.title("img")

    # 测试data_label_augment
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, label_src = cv2.threshold(gray_img, 80, 1, cv2.THRESH_BINARY)
    plt.subplot(212)
    plt.imshow(label_src)
    plt.title("img")
    plt.show()

    image_rotate, label = rotate(image, label_src)
    plt.subplot(211)
    plt.imshow(image_rotate)
    plt.title("image_rotate")
    plt.subplot(212)
    plt.imshow(label)
    plt.title("label")
    plt.show()

    image_transform, label = shear_transformation(image, label_src)

    plt.subplot(211)
    plt.imshow(image_transform)
    plt.title("image_transform")
    plt.subplot(212)

    plt.imshow(label)
    plt.title("label")
    plt.show()

    image_scale, label = scale(
        image, min_scale=0, max_scale=1, label=label_src)
    plt.subplot(211)
    plt.imshow(image_scale)
    plt.title("image_scale")
    plt.subplot(212)
    plt.imshow(label)
    plt.title("label")
    plt.show()

    image_flip, label = flip(image, 0, label_src)
    plt.subplot(211)
    plt.imshow(image_flip)
    plt.title("image_flip")
    plt.subplot(212)
    plt.imshow(label)
    plt.title("label")
    plt.show()

    image_crop, label = crop(image, label_src)
    plt.subplot(211)
    plt.imshow(image_crop)
    plt.title("image_crop")
    plt.subplot(212)
    plt.imshow(label)
    plt.title("label")
    plt.show()

    mode_list = ['gaussian', 'poisson', 'salt']
    gaussian, poisson, salt = add_noise(image, mode_list, mean=0.01, var=0.001)
    plt.subplot(311)
    plt.imshow(gaussian)
    plt.title("gaussian")
    plt.subplot(312)
    plt.imshow(poisson)
    plt.title("poisson")
    plt.subplot(313)
    plt.imshow(salt)
    plt.title("salt")
    plt.show()

    mode_list = ['pepper', 'speckle', 's&p']
    pepper, speckle, s_p = add_noise(image, mode_list, mean=0.01, var=0.001)

    plt.subplot(311)
    plt.imshow(pepper)
    plt.title("pepper")
    plt.subplot(312)
    plt.imshow(speckle)
    plt.title("speckle")
    plt.subplot(313)
    plt.imshow(s_p)
    plt.title("s_p")
    plt.show()

    image_transform, label = translation(image, 10, label_src)
    plt.subplot(211)
    plt.imshow(image_transform)
    plt.title("image_transform")
    plt.subplot(212)
    plt.imshow(label)
    plt.title("label")
    plt.show()

    image_tfactor = stauration_noise(image)
    plt.imshow(image_tfactor)

    plt.show()
