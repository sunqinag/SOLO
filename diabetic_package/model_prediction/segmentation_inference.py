import tensorflow as tf
from . import segmentation


class SegmentationModelLoader(segmentation.INNSegmentation):
    def __init__(self, model_dir):
        self.sess = tf.Session()
        try:
            tf.saved_model.loader.load(
                self.sess,
                [tf.saved_model.tag_constants.SERVING],
                model_dir)
            self.feature_placeholder = self.sess.graph.get_tensor_by_name(
                'input_img:0')
            self.predict_tensor = self.sess.graph.get_tensor_by_name(
                'classes:0')
        except IOError:
            raise ValueError(model_dir + '中不存在模型或者模型错误！')

    def __del__(self):
        if self.sess is not None:
            self.sess.close()

    def segment(self, img):
        if len(img.shape) != 4:
            raise ValueError('输入img的shape必须是4维！')
        return self.sess.run(self.predict_tensor,
                             feed_dict={self.feature_placeholder: img})
