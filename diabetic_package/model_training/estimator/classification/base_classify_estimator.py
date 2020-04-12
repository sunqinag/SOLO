import shutil
import os
import tensorflow as tf


class BaseClassifyEstimator(tf.estimator.Estimator):
    def __init__(self,
                 class_num=6,
                 model_dir='./',
                 img_shape=(227, 227, 3),
                 regularizer_scale=(0, 0),
                 label_smoothing=0,
                 label_weight=None,
                 optimizer_fn=tf.train.AdadeltaOptimizer,
                 learning_rate=1e-3,
                 tensors_to_log={'probabilities': 'softmax:0'}
                 ):
        """

            :param class_num： 分类类别数
            :param model_dir: 存储checkpoint的路径
            :param regularizer_scale: l1,l2正则项系数,tumple格式
            :param label_smoothing:使用soft label时,统计的标签存在不准确的概率
            :param label_weight: 不同类别样本加权
            :param optimizer_fn： 激活函数，双元素tumple格式，分别表示卷积层和全连接层的激活函数。
            :param learning_rate： 学习率
            :param tensors_to_log： train时输出的log，字典形式
            :return
        """
        self.model_path = model_dir
        self.regularizer_scale = regularizer_scale
        self.img_shape = img_shape
        self.label_smoothing = label_smoothing
        self.label_weight = label_weight
        self.optimizer_fn = optimizer_fn
        self.learning_rate = learning_rate
        self.tensors_to_log = tensors_to_log
        self.class_num = class_num
        self.__check_param()
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=5)
        super(BaseClassifyEstimator, self).__init__(
            model_fn=self.__model_fn,
            model_dir=self.model_path,
            config=run_config)

    def network(self, features, is_training):
        """

        :param features: network输入层
        :param is_training: 是否在train模式下调用
        :return:
        """
        raise NotImplementedError('请重写子类network方法')

    def cal_class_accuracy(self, predictions, labels, mode):
        """
        计算准确度以及分类别准确度
        :param predictions: 预测类别结果
        :param labels: 标签
        :param mode: 'train' or 'evaluate'
        :return:
        """
        # 计算每个类别的精度
        accuracy_dict = {}
        accuracy_dict['accuracy'] = tf.metrics.accuracy(labels, predictions)
        for i in range(self.class_num):
            # 每一个类别的精度
            class_indices = tf.squeeze(tf.where(tf.equal(labels, i)), 1)
            class_labels = tf.gather(labels, class_indices)
            class_predictions = tf.gather(predictions, class_indices)
            accuracy_dict[
                'class' + str(i) + '_accuracy_' + mode] = tf.metrics.accuracy(
                class_labels,
                class_predictions)
        return accuracy_dict

    def regularizer(self):
        """

        :param weights: tensorflow 自行传递
        :return:
        """
        return tf.contrib.layers.l1_l2_regularizer(
            self.regularizer_scale[0], self.regularizer_scale[1])

    def export_model(self, export_model_dir, checkpoint_path):
        """

        :param export_model_dir: 输出model路径
        :param checkpoint_path: checkpoint地址
        :return:
        """
        if os.path.exists(export_model_dir):
            shutil.rmtree(export_model_dir)
            os.mkdir(export_model_dir)
        self.estimator.export_savedmodel(
            export_dir_base=export_model_dir,
            serving_input_fn=self.__serving_input_receiver_fn,
            checkpoint_path=checkpoint_path)

    def __check_param(self):
        if len(self.label_weight) != self.class_num:
            raise ValueError('输入的label_weight列表长度应与类别个数一致')
        if self.regularizer_scale[0] < 0 or self.regularizer_scale[1] < 0:
            raise ValueError('输入的正则项系数必须大于等于0')

    def __generate_estimator_spec(self, mode, logits, labels):
        """

        :param mode: 调用模式
        :param logits: feature层
        :param labels: 用于验证的labels
        :return:
        """
        softmax = tf.nn.softmax(logits, axis=-1, name='softmax')
        predictions = {'classes': tf.argmax(logits, axis=1, name='classes'),
                       'probabilities': softmax}
        if (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL):
            penalty = tf.losses.get_regularization_loss()
            labels = tf.cast(labels, tf.int32)
            labels_weighted = None
            if self.label_weight:
                labels_weighted = tf.zeros_like(labels, dtype=tf.float32)
                for i in range(len(self.label_weight)):
                    labels_weighted += tf.where(tf.equal(labels, i),
                                                tf.multiply(tf.ones_like(labels,
                                                                         dtype=tf.float32),
                                                            self.label_weight[
                                                                i]),
                                                tf.zeros_like(labels,
                                                              dtype=tf.float32))
            loss = tf.losses.sparse_softmax_cross_entropy(
                labels, logits, weights=labels_weighted,
                reduction=tf.losses.Reduction.MEAN) + penalty
        else:
            loss = None
        if mode == tf.estimator.ModeKeys.TRAIN:
            accuracy_dict = self.cal_class_accuracy(predictions['classes'],
                                                    labels, 'train')
            optimizer = self.optimizer_fn(learning_rate=self.learning_rate)
            update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_op):
                train_op = optimizer.minimize(
                    loss=loss, global_step=tf.train.get_global_step())
            for key, value in accuracy_dict.items():
                tf.identity(value[1], name=key)
                tf.summary.scalar(key, value[1])
        else:
            train_op = None
        eval_metric_ops = None
        if mode == tf.estimator.ModeKeys.EVAL:
            accuracy_dict = self.cal_class_accuracy(predictions['classes'],
                                                    labels, 'evaluate')
            eval_metric_ops = accuracy_dict
        if mode == tf.estimator.ModeKeys.PREDICT:
            export_outputs = {'result': tf.estimator.export.PredictOutput(
                predictions['classes'])}
        else:
            export_outputs = None
        logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log,
            every_n_iter=50)

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            eval_metric_ops=eval_metric_ops,
            training_hooks=[logging_hook],
            export_outputs=export_outputs)

    def __model_fn(self, features, mode):
        """

        :param features:
        :param mode: 调用模式
        :return:
        """
        input_layer = features['img']
        if input_layer.get_shape()[-3:] != self.img_shape:
            raise ValueError('输入model_fn的图像shape有误！')
        input_layer = tf.reshape(input_layer,
                                 [-1, self.img_shape[0], self.img_shape[1],
                                  self.img_shape[2]])
        if mode == tf.estimator.ModeKeys.PREDICT:
            labels = None
        else:
            labels = features['label']
        logits = self.network(
            input_layer, mode == tf.estimator.ModeKeys.TRAIN)
        return self.__generate_estimator_spec(mode, logits, labels)
