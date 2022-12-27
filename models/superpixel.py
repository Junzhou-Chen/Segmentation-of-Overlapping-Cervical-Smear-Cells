import cv2
import numpy as np
import matplotlib.pyplot as plt


class Superpixels:
    """
    超像素建立和相关信息处理操作
    """

    def __init__(self, img=None, iteration=40, region_size=20, ruler=30.0):
        if img is None:
            img = []
        self.img = img
        slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=region_size, ruler=ruler)
        slic.iterate(iteration)  # 迭代次数
        self.mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
        self.label_slic = slic.getLabels()  # 获取超像素标签
        self.number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
        self.f_shap = []  # 形状特征
        self.f_text = []  # 纹理特征
        self.f_bound = []  # 边界特征

    def getFShap(self):
        """
        获取形状特征 f_shap
        计算小轴长度l_m和偏心EC = sqrt(1-(l_m/l_x)^2)
        :return: NULL
        """
        # 花1个小时写的十行代码，不许删！啊呜呜呜呜
        for i in range(1, self.number_slic + 1):
            # 获取单个超像素点
            mask = np.zeros(self.img.shape[:2], dtype="uint8")
            mask[self.label_slic == i] = 1
            # 获取超像素点的最小外接矩形
            ind = np.argwhere(mask == 1)
            _, (l, w), _ = cv2.minAreaRect(ind)
            if not l:
                self.f_shap.append([0, 0])
                continue
            EC = pow((1 - pow((w / l), 2)), 0.5)
            EC = format(EC, '.9f')
            self.f_shap.append([min(l, w), EC])

    def getFText(self):
        """

        需要检索目标超像素点周围的超像素点，能力有限，只检索上下左右、左下、
        左上、右上、右下八个方向
        :return:
        """
        for i in range(1, self.number_slic + 1):
            # 获取单个超像素点
            mask = np.zeros(self.img.shape[:2], dtype="uint8")
            mask[self.label_slic == i] = 1
            # 获取外接矩形
            ind = np.argwhere(mask == 1)
            x, y, w, h = cv2.boundingRect(ind)
            # 获取超像素周围元素值
            surround = self._getSurround_(x, y, w, h)
            print(surround)

    def _getSurround_(self, x, y, w, h):
        point = [[x - 1, y - 1], [x + w + 1, y - 1], [x - 1, y + h + 1],
                 [x + w + 1, y + h + 1], [x + w // 2, y - 1], [x - 1, y + h // 2],
                 [x + w + 1, y + h // 2], [x + w // 2, y + h + 1]]
        ix, iy = self.mask_slic.shape
        num = []
        for i in point:
            if 0 < i[0] < ix and 0 < i[1] < iy:
                number = self.label_slic[i[0]][i[1]]
                if number not in num:
                    num.append(number)
        return num

    def getFBound(self):
        pass


def SLIC(img: np.ndarray, iteration=40):
    """
    :depict 输入图像数据，返回超像素Mask，超像素标签和标签数量
    :param iteration: 迭代次数,default 40, int
    :param img: image data, numpy.ndarray
    :return: Mask array, 超像素标签 array, 超像素数目 int
    """
    slic = cv2.ximgproc.createSuperpixelSLIC(img, region_size=15, ruler=30.0)
    slic.iterate(iteration)  # 迭代次数
    mask_slic = slic.getLabelContourMask()  # 获取Mask，超像素边缘Mask==1
    label_slic = slic.getLabels()  # 获取超像素标签
    number_slic = slic.getNumberOfSuperpixels()  # 获取超像素数目
    return mask_slic, label_slic, number_slic
