import cv2
import numpy
import matplotlib.pyplot as plt
import csv
from models import *


def PLT_SHOW(img):
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    test_img_path = r'./img_path/42.png'
    test_img = cv2.imread(test_img_path)
    PLT_SHOW(test_img)
    img_1 = Superpixels(test_img)
    PLT_SHOW(img_1.mask_slic)
    img_1.getFText()
