#-*- coding:utf-8 -*-
import os
import sys
import cv2
from math import *
import numpy as np
import mahotas
from PIL import Image

#sys.path.append(os.getcwd() + '/ctpn')
#from ctpn.text_detect import text_detect
#from lib.fast_rcnn.config import cfg_from_file
from densenet.model import predict as keras_densenet


def sort_box(box):
    """ 
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box

def adjustGray(cropImage):
    gray = cv2.cvtColor(cropImage, cv2.COLOR_BGR2GRAY)
    thresh, img_bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    his = np.histogram(img_bw)
    if his[0][0] >= his[0][-1]:
        print(his[0][0], his[0][-1])
        for i in range(img_bw.shape[0]):
            for j in range(img_bw.shape[1]):
                gray[i][j] = 255 - gray[i][j]
                img_bw[i][j] = 255 - img_bw[i][j]
    return gray, img_bw

def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])) : min(ydim - 1, int(pt3[1])), max(1, int(pt1[0])) : min(xdim - 1, int(pt3[0]))]

    return imgOut

def charRec(img, text_recs, adjust=False):
   """
   加载OCR模型，进行字符识别
   """
   results = {}
   xDim, yDim = img.shape[1], img.shape[0]
    
   for index, rec in enumerate(text_recs):
       xlength = int((rec[6] - rec[0]) * 0.1)
       ylength = int((rec[7] - rec[1]) * 0.2)
       if adjust:
           pt1 = (max(1, rec[0] - xlength), max(1, rec[1] - ylength))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6] + xlength, xDim - 2), min(yDim - 2, rec[7] + ylength))
           pt4 = (rec[4], rec[5])
       else:
           pt1 = (max(1, rec[0]), max(1, rec[1]))
           pt2 = (rec[2], rec[3])
           pt3 = (min(rec[6], xDim - 2), min(yDim - 2, rec[7]))
           pt4 = (rec[4], rec[5])
           
       degree = degrees(atan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))  # 图像倾斜角度

       partImg = dumpRotateImage(img, degree, pt1, pt2, pt3, pt4)
	    #进行阈值
       thresh = cv2.cvtColor(partImg,cv2.COLOR_BGR2GRAY)
	    #Otsu's threshold法
       T= mahotas.thresholding.otsu(thresh)
       thresh[thresh >T] = 255#矩阵thresh中>T的值赋值为255
       thresh[thresh <= T] = 0#矩阵thresh中<255的值赋值为0
       #确认背景和前景
       nmx = np.sum(thresh[0,:]==255)
       nmy = np.sum(thresh[:,1]==255)
       bk = 255
       if nmx<thresh.shape[1]/2 or nmy<thresh.shape[0]/2:
           bk = 0
       if bk==0:
           partImg = 255-partImg
       '''partImg = 255 - thresh
       else:
           partImg = thresh'''
       '''gray , img_bw= adjustGray(partImg)
       partImg = img_bw'''

       if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0]/partImg.shape[1]>2:  # 过滤异常图片
           continue

       image = Image.fromarray(partImg).convert('L')
       #print(partImg.shape)
       #image = Image.fromarray(partImg)
       text = keras_densenet(image)
       
       if len(text) > 0:
           results[index] = [rec]
           results[index].append(text)  # 识别文字
 
   return results

def model(img, adjust=False):
    """
    @img: 图片
    @adjust: 是否调整文字识别结果
    """
    cfg_from_file('./ctpn/ctpn/text.yml')
    text_recs, img_framed, img = text_detect(img)
    text_recs = sort_box(text_recs)
    result = charRec(img, text_recs, adjust)
    return result, img_framed

