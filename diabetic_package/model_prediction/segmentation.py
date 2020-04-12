# -------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称 ：
#   摘   要 ：病变分割
#   当前版本 : 0.0
#   作   者 ：于川汇 陈瑞侠
#   完成日期 : 2018-5-4
# -------------------------------------------------------------
from abc import ABCMeta, abstractmethod


class INNSegmentation(metaclass=ABCMeta):
    @abstractmethod
    def segment(self, img):
        '''
            输入待分割图像img，返回经网络分割后的二值图
        '''
        pass
