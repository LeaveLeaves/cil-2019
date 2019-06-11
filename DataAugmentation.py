# -*- coding: utf-8 -*-
import cv2
import numpy as np
import os

def ImageRotation(path):
    for file in os.listdir(path):
        img = cv2.imread(path + "/" + file)
        img90 = np.rot90(img)
        cv2.imwrite("90_" + file, img90)
        img180 = np.rot90(img90)
        cv2.imwrite("180_"+ file, img180)
        img270 = np.rot90(img180)
        cv2.imwrite("270_" + file, img270)

ImageRotation("./images")
ImageRotation("./groundtruth")   