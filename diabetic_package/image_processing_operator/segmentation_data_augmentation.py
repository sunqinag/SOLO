import numpy as np
import cv2
from ..image_processing_operator import region_operator
from ..file_operator import bz_path
from ..model_training import data_augmentation


def get_defect_difference_from_background(defect_img, defect_label):
    """
    :param defect_img: 缺陷图像
    :param defect_label: 缺陷图像标签
    :return:
    """

    if defect_img.shape[:2] != defect_label.shape:
        raise ValueError('缺陷图像与缺陷标签图像高宽必须一致！')
    binary = (defect_label > 0).astype(np.int32)
    binary_inv = 1 - binary
    if len(defect_img.shape) != 3:
        defect_img = np.expand_dims(defect_img, axis=2)
    mean_intensity_list = []
    for i in range(defect_img.shape[2]):
        regions = region_operator.region_props(
            binary_image=binary_inv, intensity_img=defect_img[:, :, i])
        all_area = 0
        all_intensity = 0
        for region in regions:
            area = region.area
            mean_intensity = region.mean_intensity
            all_area += area
            all_intensity += mean_intensity * area
        if all_area == 0:
            raise ZeroDivisionError('除数all_area不能为0！')
        all_mean_intensity = all_intensity / all_area
        mean_intensity_list.append(all_mean_intensity)
    return np.squeeze(defect_img - np.array(mean_intensity_list))


def add_noise_to_defect(defect_img,
                        gauss_params=(0.0001, 0.0001),
                        local_params=(0.0001, 0.00006),
                        salt_params=(0.0001, 0.00015)):
    """
    :param defect_img: 缺陷与背景的差值
    :param gauss_params: gauss噪声params
    :param local_params: local噪声params
    :param salt_params: salt噪声params
    :return:
    """
    random_noise = np.random.randint(0, 4)
    if random_noise == 0:
        return defect_img
    elif random_noise == 1:
        defect_noise = data_augmentation.add_noise(defect_img.copy(),
                                                   ['gaussian'],
                                                   mean=gauss_params[0],
                                                   var=gauss_params[1])
    elif random_noise == 2:
        defect_noise = data_augmentation.add_noise(defect_img.copy(),
                                                   ['localvar'],
                                                   mean=local_params[0],
                                                   var=local_params[1])
    else:
        defect_noise = data_augmentation.add_noise(defect_img.copy(),
                                                   ['salt'],
                                                   mean=salt_params[0],
                                                   var=salt_params[1])
    return np.uint8(defect_noise[0] * 255)


# 在目标图像上平移缺陷
def random_locate_defect_to_img(img,
                                label,
                                defect_difference,
                                defect_label):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    height, width = np.shape(img)[:2]
    defect_height, defect_width = np.shape(defect_difference)[:2]
    img_convert = img.astype(np.float32)
    # 增加while循环
    while (1):
        location_height = np.random.randint(0, height - defect_height)
        location_width = np.random.randint(0, width - defect_width)
        img_convert_copy = img_convert.copy()
        img_convert[location_height: location_height + defect_height,
        location_width: location_width + defect_width] += \
            defect_difference
        img_convert[img_convert < 0] = 0
        img_convert[img_convert > 255] = 255
        label_copy = label.copy()
        label[location_height: location_height + defect_height,
        location_width: location_width + defect_width] = defect_label
        # 判断缺陷标签之间是否有重合
        if (np.sum(label > 0) - np.sum(label_copy > 0)) == \
                np.sum(defect_label > 0):
            break
        img_convert = img_convert_copy
        label = label_copy
    img_fuzzy = fuzzy_defect_boundary_from_img(img_convert,
                                               defect_difference,
                                               location_height,
                                               location_width,
                                               10,
                                               10)

    return np.uint8(img_fuzzy), label


# 增加差值图像后做模糊（单独写成函数）
def fuzzy_defect_boundary_from_img(img,
                                   defect_difference,
                                   row,
                                   column,
                                   extend_height=10,
                                   extend_width=10):
    """
    :param img: 目标图像
    :param defect_difference: 缺陷与背景的差值
    :param column: 缺陷图像在目标图像的起始行
    :param row: 缺陷图像在目标图像的起始列
    :param extend_height: 扩张高度
    :param extend_width: 扩张宽度
    :return:
    """
    height, width = img.shape[:2]
    defect_height, defect_width = defect_difference.shape[:2]
    # 考虑边界问题
    left_column = np.maximum(0, row - extend_height)
    right_column = np.minimum(row + defect_height + extend_height, height)
    left_row = np.maximum(0, column - extend_width)
    right_row = np.minimum(column + defect_width + extend_width, width)
    extend_defect_img = img[left_column: right_column, left_row: right_row]
    extend_defect_height, extend_defect_width = extend_defect_img.shape[:2]
    extend_defect_img_copy = extend_defect_img.copy()
    fuzzy_defect_img = cv2.blur(extend_defect_img_copy, (3, 3))

    # 将扩张后的模糊缺陷放回到目标图像上
    img[left_column: right_column, left_row: right_row] = fuzzy_defect_img
    # 将原始缺陷图像放回到目标图像上
    img[left_column + extend_height: right_column - extend_height,
    left_row + extend_width: right_row - extend_width] = \
        extend_defect_img[extend_height: extend_defect_height - extend_height,
        extend_width: extend_defect_width - extend_width]
    return img


# 在目标图像上翻转缺陷
def random_flip_defect_to_img(img,
                              label,
                              defect_difference,
                              defect_label):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    flip_type = np.random.randint(-1, 3)
    if flip_type == 2:
        return random_locate_defect_to_img(img,
                                           label,
                                           defect_difference,
                                           defect_label)
    flip_defect = cv2.flip(defect_difference, flip_type)
    flip_defect_label = cv2.flip(defect_label, flip_type)
    return random_locate_defect_to_img(
        img, label, flip_defect, flip_defect_label)


# 在目标图像上缩放缺陷
def random_zoom_defect_to_img(img,
                              label,
                              defect_difference,
                              defect_label=None,
                              min_scale=0.5,
                              max_scale=2):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景的差值
    :param defect_label: 缺陷图像标签
    :param min_scale: 缩放最小值比例
    :param max_scale: 缩放最大值比例
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if min_scale < 0 or max_scale < 0:
        raise ValueError('min_scale, max_scale必须大于0！')
    if min_scale > max_scale:
        raise ValueError('min_angle必须大于max_angle!')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    level_scale, vertial_scale = np.random.uniform(min_scale, max_scale, [2])
    defect_height, defect_width = np.shape(defect_difference)[:2]
    scale_height = (np.round(defect_height * level_scale)).astype(np.int32)
    scale_width = (np.round(defect_width * vertial_scale)).astype(np.int32)
    scale_defect = cv2.resize(defect_difference,
                              (scale_width, scale_height),
                              interpolation=cv2.INTER_LINEAR)
    scale_defect_label = cv2.resize(defect_label,
                                    (scale_width, scale_height),
                                    interpolation=cv2.INTER_NEAREST)
    return random_locate_defect_to_img(
        img, label, scale_defect, scale_defect_label)


# 在目标图像旋转缺陷
def random_rotate_defect_to_img(img,
                                label,
                                defect_difference,
                                defect_label=None,
                                min_angle=0,
                                max_angle=360):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景差值
    :param defect_label: 缺陷图像标签
    :param min_angle: 旋转最小角度
    :param max_angle: 旋转最大角度
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if min_angle > max_angle:
        raise ValueError('min_angle必须大于max_angle!')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    # 随机旋转缺陷
    rotation_angle = np.random.uniform(min_angle, max_angle)

    def get_extension_img(img):
        height, width = np.shape(img)[:2]
        max_length = np.maximum(height, width)
        if len(img.shape) != 3:
            extension_img = np.zeros([max_length, max_length])
        else:
            extension_img = np.zeros([max_length, max_length, img.shape[2]])
        extension_img[np.int32(max_length / 2) - np.int32(height / 2):
                      np.int32(max_length / 2) + np.int32(height / 2),
        np.int32(max_length / 2) - np.int32(width / 2):
        np.int32(max_length / 2) + np.int32(width / 2)] = \
            img[: height, : width]
        return extension_img

    extension_defect = get_extension_img(defect_difference)
    extension_height, extension_width = np.shape(extension_defect)[:2]
    rotation_center = (extension_height / 2, extension_width / 2)
    rotation_matrix2d = cv2.getRotationMatrix2D(
        rotation_center, rotation_angle, scale=1)
    rotation_defect = cv2.warpAffine(extension_defect,
                                     rotation_matrix2d,
                                     (extension_width, extension_height))
    extension_defect_label = get_extension_img(defect_label)
    rotation_defect_label = cv2.warpAffine(extension_defect_label,
                                           rotation_matrix2d,
                                           (extension_width, extension_height))
    return random_locate_defect_to_img(
        img, label, rotation_defect, rotation_defect_label)


# 在目标图像上拉伸缺陷
def random_shear_defect_to_img(img,
                               label,
                               defect_difference,
                               defect_label=None,
                               min=-1,
                               max=1):
    """
    :param img: 目标图像
    :param label: 目标图像标签
    :param defect_difference: 缺陷与背景差值
    :param defect_label: 缺陷图像标签
    :param min1: 最小值
    :param max1: 最大值
    :return:
    """
    if img.shape[0] < defect_difference.shape[0] \
            or img.shape[1] < defect_difference.shape[1]:
        raise ValueError('目标图像shape必须大于缺陷图像shape！')
    if img.shape[:2] != label.shape[:2]:
        raise ValueError('目标图像和标签图像高宽必须必须一致！')
    if min > max:
        raise ValueError('min_angle必须小于max_angle!')
    if defect_difference.shape[:2] != defect_label.shape:
        raise ValueError('缺陷差值图像与缺陷标签图像高宽必须一致！')
    level_scale, vertical_scale, level_angle, vertical_angle = \
        np.random.uniform(min, max, [4])
    M = np.array(
        [[level_scale, level_angle, 0], [vertical_angle, vertical_scale, 0]],
        dtype=np.float32)
    shear_defect = cv2.warpAffine(defect_difference, M, (
        np.max(defect_difference.shape), np.max(defect_difference.shape)))

    shear_defect_label = cv2.warpAffine(defect_label, M, (
        np.max(defect_label.shape), np.max(defect_label.shape)))
    return random_locate_defect_to_img(
        img, label, shear_defect, shear_defect_label)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os

    defect_img_dir = './DEFECT/Untitled Folder'
    defect_label_dir = './Defect_Label/'
    img_dir = './Image'
    label_dir = './Label'
    location_result_dir = './locate_img'
    rotation_result_dir = './rotate_img'
    flip_result_dir = './flip_img'
    shear_result_dir = './stretch_img'
    for img_path in bz_path.get_file_path(img_dir, ret_full_path=True):
        img_name = os.path.splitext(os.path.split(img_path)[1])[0]
        img = cv2.imread(img_path, 0)
        label = cv2.imread(label_dir + '/' + img_name + '.png', 0)
        flip_img = img.copy()
        flip_label = label.copy()
        for defect_img_path in bz_path.get_file_path(
                defect_img_dir, ret_full_path=True):
            defect_img = cv2.imread(defect_img_path, 0)
            defect_img_name = \
                os.path.splitext(os.path.split(defect_img_path)[1])[0]
            defect_label = cv2.imread(
                defect_label_dir + '/' + defect_img_name + '.png', 0)
            defect_label = defect_label
            difference_img = get_defect_difference_from_background(
                defect_img, defect_label)
            # noise_img = add_noise_to_defect(defect_img)

            # # 在目标图像上平移缺陷
            # location_img, location_label = random_locate_defect_to_img(
            #     img, label, difference_img, defect_label)
            # cv2.imwrite(location_result_dir + '/' + img_name + '.png', img)
            # cv2.imwrite(location_result_dir + '/' + img_name + '_.png',
            #             location_img)
            # cv2.imwrite(location_result_dir + '/' + img_name + '__.png',
            #             location_label)

            # 在目标图像上翻转缺陷
            flip_img, flip_label = random_flip_defect_to_img(
                img, label, difference_img, defect_label)


            # cv2.imwrite(flip_result_dir + '/' + img_name + '.png', img)
            # cv2.imwrite(flip_result_dir + '/' + img_name + '_.png',
            #             flip_img)
            # cv2.imwrite(translate_result_dir + '/' + img_name + '__.png',
            #             translation_img2)

            # # 在目标图像上旋转缺陷
            # rotation_img, rotation_label = rotate_defect_to_img(
            #     img, difference_img, defect_label)
            # cv2.imwrite(rotation_result_dir + '/' + img_name + '.png', img)
            # cv2.imwrite(rotation_result_dir + '/' + img_name + '_.png',
            #             rotation_img)
            # cv2.imwrite(rotation_result_dir + '/' + img_name + '__.png',
            #             rotation_label)
            #
            # 在目标图像上缩放缺陷
            # scale_img, scale_label = random_zoom_defect_to_img(
            #     img, label, difference_img, defect_label, 0.5, 2)
            # cv2.imwrite(scale_result_dir + '/' + img_name + '.png', img)
            # cv2.imwrite(scale_result_dir + '/' + img_name + '_.png',
            #             scale_img)
            # cv2.imwrite(scale_result_dir + '/' + img_name + '__.png',
            #             scale_label)

            # 在目标图像上拉伸缺陷
            # shear_img, shear_label = shear_defect_to_img(
            #     img, label, difference_img, defect_label)
            # cv2.imwrite(shear_result_dir + '/' + img_name + '.png', img)
            # cv2.imwrite(shear_result_dir + '/' + img_name + '_.png',
            #             shear_img)
            # cv2.imwrite(shear_result_dir + '/' + img_name + '__.png',
            #             shear_label)

            def draw_contours(img, binary, color, thickness):
                max_val = np.max(binary)
                if max_val == 0:
                    return img
                _, contours, hierarchy = cv2.findContours(
                    binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                img_draw = img.copy()
                img_draw = cv2.drawContours(img_draw, contours, -1, color,
                                            thickness)
                return img_draw


            binary = (flip_label > 200).astype(np.uint8)
            translation_img_copy = flip_img.copy()
            translation_img_draw = draw_contours(
                translation_img_copy, binary, [255, 0, 0], 2)
            plt.subplot(131)
            plt.imshow(img, cmap=plt.gray())
            plt.title("img")

            plt.subplot(132)
            plt.imshow(flip_img, cmap=plt.gray())
            plt.title("translate_img")

            plt.subplot(133)
            plt.imshow(flip_label * 255, cmap=plt.gray())
            plt.title("translate_label")

            # plt.subplot(144)
            # plt.imshow(translation_img_draw, cmap=plt.gray())
            # plt.title("translate_img_draw")
            plt.show()
