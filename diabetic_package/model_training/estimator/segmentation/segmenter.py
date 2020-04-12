import tensorflow as tf
import numpy as np
import os
import shutil
from ....file_operator import bz_path
from ....dataset_common import dataset
import cross_val


def cal_class_weighted_accuracy(result, accuracy_weight):
    if isinstance(result, dict) is False:
        raise ValueError('result必须是字典类型！')
    weighted_accuracy = 0
    i = 0
    for key, value in result.items():
        if 'evaluate' in key:
            weighted_accuracy += value * accuracy_weight[i]
            i += 1
    return weighted_accuracy


class Segmenter():
    def __init__(self,
                 model_dir,
                 estimator_obj,
                 height_and_width,
                 crop_height_and_width=(400, 400),
                 channels_list=(1, 1),
                 img_format_list=('bmp', 'png'),
                 class_num=2,
                 accuracy_fn=cal_class_weighted_accuracy,
                 accuracy_weight=(0.98, 0.02),
                 k_fold=1,
                 eval_frequency=1,
                 batch_size=1,
                 epoch_num=1):
        """
        :param model_dir: 模型路径
        :param estimator_obj: estimator对象
        :param height_and_width: 图像高度和宽度
        :param crop_height_and_width: 图像crop高度和宽度
        :param channels_list: 图像通道list
        :param img_format_list: 图像类型list
        :param class_num: 分类个数
        :param accuracy_weight: 精度权重
        :param k_fold: 交差验证折数
        :param batch_size: batch大小
        :param epoch_num: epoch数量
        """
        self.img_height = height_and_width[0]
        self.img_width = height_and_width[1]
        self.img_crop_height = crop_height_and_width[0]
        self.img_crop_width = crop_height_and_width[1]
        self.channel_list = channels_list
        self.img_format_list = img_format_list
        self.model_dir = model_dir
        self.class_num = class_num
        self.accuracy_fn = accuracy_fn
        self.accuracy_weight = accuracy_weight
        self.estimator_obj = estimator_obj
        self.k_fold = k_fold
        self.eval_frequency = eval_frequency
        self.batch_size = batch_size
        self.epoch_num = epoch_num

    def fit(self,
            train_features,
            train_labels,
            eval_features=None,
            eval_labels=None):
        """
        :param train_features: 训练图像路径
        :param train_labels: 训练标签路径
        :param eval_features: 验证图像路径
        :param eval_labels: 验证标签路径
        :return:
        """
        # 交叉验证
        accuracy = 0
        eval_result = {}
        for i in range(self.epoch_num):
            print('epoch:', i)
            if self.k_fold > 1:
                data_list = cross_val.corss_val_data(self.k_fold,
                                                     train_features,
                                                     train_labels)
                eval_result = {}
                for j in range(self.k_fold):
                    sub_train_features, \
                    sub_train_labels, \
                    sub_eval_features, \
                    sub_eval_labels = data_list[j]
                    self.estimator_obj.train(lambda: self.__train_input_fn(
                        sub_train_features, sub_train_labels))
                    cross_eval_result = self.estimator_obj.evaluate(
                        lambda: self.__eval_input_fn(
                            sub_eval_features, sub_eval_labels))
                    print(cross_eval_result)
                    for key, value in cross_eval_result.items():
                        if j == 0:
                            eval_result[key] = value
                        else:
                            eval_result[key] += value
                for key, value in eval_result.items():
                    eval_result[key] /= self.k_fold
                print(eval_result)
            else:
                if eval_features is None or eval_labels is None:
                    raise ValueError('非交叉验证时必须输入验证集！')
                self.estimator_obj.train(
                    lambda: self.__train_input_fn(train_features, train_labels))
                if i % self.eval_frequency == 0:
                    eval_result = self.estimator_obj.evaluate(
                        lambda: self.__eval_input_fn(eval_features,
                                                     eval_labels))
                    print(eval_result)
            weighted_accuracy = self.accuracy_fn(
                eval_result, self.accuracy_weight)
            # 模型保存的条件
            if weighted_accuracy > accuracy:
                export_model_dir = self.model_dir + '/export_model_dir'
                best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir'
                with open('accuracy.txt', 'w') as f:
                    for key, value in eval_result.items():
                        f.write(key + ':' + str(value))
                self.export_model(export_model_dir=export_model_dir)
                self.__save_best_checkpoint(best_checkpoint_dir)
                accuracy = weighted_accuracy

    def predict(self, features):
        """
        :param features: 预测图像路径
        :return:
        """
        predictions = self.estimator_obj.predict(
            lambda: self.__predict_input_fn(features))
        return np.array([pre['classes'] for pre in predictions])

    def export_model(self, export_model_dir):
        """
        :param export_model_dir: export模型路径
        :return:
        """
        best_checkpoint_dir = self.model_dir + '/best_checkpoint_dir/'
        self.__save_best_checkpoint(best_checkpoint_dir)
        best_checkpoint = best_checkpoint_dir + os.path.split(
            self.estimator_obj.latest_checkpoint())[1]
        return self.estimator_obj.export_model(export_model_dir,
                                               best_checkpoint)

    def __save_best_checkpoint(self, best_checkpoint_dir):
        """
        :param best_checkpoint_dir: best_checkpoint路径
        :return:
        """
        if (os.path.exists(best_checkpoint_dir)):
            shutil.rmtree(best_checkpoint_dir)
        os.mkdir(best_checkpoint_dir)
        checkpoint_files = bz_path.get_file_path(
            self.model_dir, ret_full_path=True)
        (_, checkpoint_name) = os.path.split(
            self.estimator_obj.latest_checkpoint())
        for path in checkpoint_files:
            if checkpoint_name in path:
                shutil.copy(path, best_checkpoint_dir)

    def __train_input_fn(self, features, labels):
        """
        :param features: 训练图像路径
        :param labels: 训练标签路径
        :return:
        """
        train_dataset = dataset.ImageLabelDataSet(
            img_list=features,
            label_list=labels,
            channels_list=self.channel_list,
            img_format_list=self.img_format_list,
            height_width=(self.img_height, self.img_width),
            crop_height_width=(self.img_crop_height, self.img_crop_width),
            batch_size=self.batch_size,
            num_epochs=self.eval_frequency,
            shuffle=False,
            mode='train',
            task='segmentation')
        return train_dataset.create_dataset()

    def __eval_input_fn(self, features, labels):
        """
        :param features: 验证图像路径
        :param labels: 验证标签路径
        :return:
        """
        eval_dataset = dataset.ImageLabelDataSetBaseClass(
            img_list=features,
            label_list=labels,
            channels_list=self.channel_list,
            img_format_list=self.img_format_list,
            height_width=(self.img_height, self.img_width),
            crop_height_width=(self.img_crop_height, self.img_crop_width),
            batch_size=self.batch_size,
            num_epochs=1,
            mode='evaluate',
            task='segmentation')
        return eval_dataset.create_dataset()

    def __predict_input_fn(self, features):
        """
        :param features: 预测图像路径
        :return:
        """
        predict_dataset = dataset.ImageLabelDataSetBaseClass(
            img_list=features,
            label_list=None,
            channels_list=self.channel_list[0],
            img_format_list=self.img_format_list[0],
            height_width=(self.img_height, self.img_width),
            crop_height_width=(self.img_crop_height, self.img_crop_width),
            mode='predict',
            task='segmentation')
        return predict_dataset.create_dataset()
