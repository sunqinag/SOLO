import numpy as np
import os
import shutil
from diabetic_package.dataset_common import dataset
from alexnet.cross_val import corss_val_data
import json


class AlexnetDataSet(dataset.ImageLabelDataSetBaseClass):
    def preprocess_image_map(self, data_dict):
        """
        图像处理的map函数
        :param data_dict:
        :param is_mirroring:
        :return:
        """
        image = data_dict['img']
        label = data_dict['label']
        # image, label = tensorflow_image_processing_map.random_rescale_image_and_label_tf_map(
        #     image, label, min_scale=0.5, max_scale=2.0)
        #
        # image, label = tensorflow_image_processing_map.random_crop_or_pad_image_and_label_tf_map(
        #     image, label=label, crop_height=self.crop_height_width[0], crop_width=self.crop_height_width[1])
        # # image,label=tensorflow_image_processing_map.flip_image_and_label_tf_map(image,label)
        # image, label = tensorflow_image_processing_map.random_rotate_image_and_label_tf_map(image, label,
        #                                                                                     max_angle=40)
        # image, label = tensorflow_image_processing_map.flip_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.flip_up_down_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.flip_left_right_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.transpose_image_image_and_label_tf_map(image, label)
        # # # image, label = tensorflow_image_processing_map.translate_image_and_label_tf_map(image, label, dx=10, dy=-60.0)
        # image, label = tensorflow_image_processing_map.random_translate_image_and_label_tf_map(image, label,
        #                                                                                        max_dx=30,
        #                                                                                        max_dy=30)
        # #
        # # # image = tensorflow_image_processing_map.random_brightness_image_tf_map(image, 10)
        # # # image = tensorflow_image_processing_map.random_contrast_image_tf_map(image, 0, 200)
        # # # image = tensorflow_image_processing_map.random_hue_image_tf_map(image, 0.5)
        # # # image = tensorflow_image_processing_map.random_saturation_image_tf_map(image, 0.5,3.0)
        # image = tensorflow_image_processing_map.add_random_noise_tf_map(image, 0.5, 100)
        # # image = tensorflow_image_processing_map.add_gaussian_noise_tf_map(image, 0.5, 3.0)
        #
        # # image = tensorflow_image_processing_map.add_salt_and_pepper_noise_pyfunc_tf_map(image, 0.001, 255)
        return image, label


class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


class Classifier:
    def __init__(self,
                 estimator_obj,
                 model_dir='./model_dir',
                 best_checkpoint_dir='./best_checkpoint_dir',
                 height_width=(227, 227),
                 crop_height_width=(227, 227),
                 k_fold=10,
                 channels_list=[3],
                 img_format_list=['bmp'],
                 batch_size=1,
                 epoch_num=1
                 ):
        """

        :param model_dir: checkpoint保存目录
        :param best_checkpoint_dir: 最佳checkpoint保存目录
        :param estimator_obj: estimator对象
        :param height_width: 输入dataset后resize图像目标高宽
        :param crop_height_width: crop图像目标高宽
        :param k_fold: 交叉验证折数，正整数，值为1时不做交叉验证
        :param channels_list: list形式，分类任务传一个元素，为图像通道数
        :param img_format_list: list形式，分类任务传一个元素，为图像格式
        :param batch_size:
        :param epoch_num:
        """
        self.model_dir = model_dir
        self.estimator = estimator_obj
        self.height_width = height_width
        self.crop_height_width = crop_height_width
        self.k_fold = k_fold
        self.channels_list = channels_list
        self.img_format_list = img_format_list
        self.batch_size = batch_size
        self.epoch_num = epoch_num
        self.best_checkpoint_dir = best_checkpoint_dir
        self.best_model_info_path = self.best_checkpoint_dir + '/best_model_info.json'
        self.value_judgment = - np.inf
        self.__init_value_judgment()

    def cal_value_judgment(self, eval_result):
        """
        计算判决值，这个判决值大小决定是否保存这一模型。
        :param eval_result: eval模式下输出，字典形式
        :return:
        """
        loss = eval_result['loss']
        # accuracy = eval_result['accuracy']
        label_weight = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        accuracy_weight_sum = 0
        for class_index in range(len(label_weight)):
            accuracy_weight_sum += eval_result['class' + str(
                class_index) + '_accuracy_evaluate'] \
                                   * label_weight[class_index]
        value_judgment = accuracy_weight_sum - loss
        return value_judgment

    def __init_value_judgment(self):
        if os.path.exists(self.best_model_info_path):
            with open(self.best_model_info_path, 'r') as f:
                best_model_info_dict = json.load(f)
            self.value_judgment = best_model_info_dict['value_judgment']

    def __save_best_checkpoint(self, eval_result):
        """

        :param eval_result: eval模式下输出，字典形式
        :return:
        """
        print('Saving checkpoint to' + self.best_checkpoint_dir)
        if os.path.exists(self.best_checkpoint_dir):
            shutil.rmtree(self.best_checkpoint_dir)
        os.makedirs(self.best_checkpoint_dir)
        (_, checkpoint_name) = os.path.split(
            self.estimator.latest_checkpoint())
        for root, dirs, files in os.walk(self.model_dir):
            for path in files:
                if checkpoint_name in path:
                    shutil.copy(self.model_dir + '/' + path,
                                self.best_checkpoint_dir)
        with open(self.best_model_info_path, 'w') as f:
            json.dump(eval_result, f, cls=MyEncoder)

    def fit(self, train_images_path, train_labels, eval_images_path,
            eval_labels, val_epoch_num):
        """

        :param train_images_path: nd.array形式，元素为训练集图片全路径
        :param train_labels: nd.array形式，训练集对应labels
        :param eval_images_path: nd.array形式，元素为验证集图片全路径，交叉验证时为None
        :param eval_labels: nd.array形式，训练集对应labels，交叉验证时为None
        :param val_epoch_num: 非交叉验证时，训练几个eopch验证一次
        :return:
        """
        if self.k_fold > 1:
            for i in range(self.epoch_num):
                dataset_dict = corss_val_data(self.k_fold, train_images_path,
                                              train_labels)
                eval_result = {}
                cross_eval_result = {}
                for j in range(self.k_fold):
                    print('交叉验证第%d/%d次训练开始' % (j, self.k_fold))
                    train_images_path, train_labels, eval_images_path, eval_labels = \
                    dataset_dict[j]
                    self.estimator.train(input_fn=lambda: self.__train_input_fn(
                        train_images_path, train_labels))
                    cross_eval_result = self.estimator.evaluate(
                        input_fn=lambda: self.__eval_input_fn(eval_images_path,
                                                              eval_labels))
                    for key, value in cross_eval_result.items():
                        if key not in ['global_step']:
                            if j == 0:
                                eval_result[key] = value / self.k_fold
                            else:
                                eval_result[key] += value / self.k_fold
                eval_result['global_step'] = cross_eval_result['global_step']
                print(eval_result)
                value_judgment = self.cal_value_judgment(eval_result)
                if value_judgment > self.value_judgment:
                    self.value_judgment = value_judgment
                    eval_result['value_judgment'] = value_judgment
                    self.__save_best_checkpoint(eval_result)
        else:
            if not eval_images_path or not eval_labels:
                raise ValueError("非交叉验证训练时，应输入验证集")
            index_shuffle = np.arange(train_images_path.shape[0])
            np.random.shuffle(index_shuffle)
            train_images_path = train_images_path[index_shuffle]
            train_labels = train_labels[index_shuffle]
            for i in range(self.epoch_num):
                self.estimator.train(
                    input_fn=lambda: self.__train_input_fn(train_images_path,
                                                           train_labels))
                if not i % val_epoch_num:
                    eval_result = self.estimator.evaluate(
                        input_fn=lambda: self.__eval_input_fn(eval_images_path,
                                                              eval_labels))
                    print(eval_result)
                    value_judgment = self.cal_value_judgment(eval_result)
                    if value_judgment > self.value_judgment:
                        self.value_judgment = value_judgment
                        eval_result['value_judgment'] = value_judgment
                        self.__save_best_checkpoint(eval_result)

    def predict(self, features):
        predictions = self.estimator.predict(
            lambda: self.__predict_input_fn(features))
        return np.array([pre['classes'] for pre in predictions])

    def __train_input_fn(self, features, labels):
        train_dataset = AlexnetDataSet(
            img_list=features,
            label_list=labels,
            channels_list=self.channels_list,
            img_format_list=self.img_format_list,
            height_width=self.height_width,
            crop_height_width=self.crop_height_width,
            batch_size=self.batch_size,
            num_epochs=1,
            shuffle=True)
        return train_dataset.create_dataset()

    def __eval_input_fn(self, features, labels):
        eval_dataset = AlexnetDataSet(
            img_list=features,
            label_list=labels,
            channels_list=self.channels_list,
            img_format_list=self.img_format_list,
            height_width=self.height_width,
            crop_height_width=self.crop_height_width,
            batch_size=self.batch_size,
            num_epochs=1,
            mode='evaluate')
        return eval_dataset.create_dataset()

    def __predict_input_fn(self, features):
        predict_dataset = AlexnetDataSet(
            img_list=features,
            label_list=None,
            channels_list=self.channels_list,
            img_format_list=self.img_format_list,
            height_width=self.height_width,
            mode='predict')
        return predict_dataset.create_dataset()
