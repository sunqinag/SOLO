# ------------------------------------------------------------------
#   !Copyright(C) 2018, 北京博众
#   All right reserved.
#   文件名称：
#   摘   要：病变分类
#   当前版本: 0.0
#   作   者：于川汇 陈瑞侠
#   完成日期: 2018-1-26
# ------------------------------------------------------------------
from abc import ABCMeta, abstractmethod


class IClassifier(metaclass=ABCMeta):
    @abstractmethod
    def classify(self, input_data):
        pass
