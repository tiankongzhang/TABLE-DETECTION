#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 24 08:37:45 2018

@author: zhouwenzhang
"""
import json
from collections import OrderedDict

import cv2
import time
import math
import os
import ocr
import copy
import mahotas
import numpy as np
from PIL import Image
import tensorflow as tf

import locality_aware_nms as nms_locality
import lanms

#输入相关目录
tf.app.flags.DEFINE_string('test_data_path', 'test_data', '')
tf.app.flags.DEFINE_string('gpu_list', '2', '')
tf.app.flags.DEFINE_string('checkpoint_path', 'east_resnet_v1_50_rbox', '')

#输出一级目录
tf.app.flags.DEFINE_string('out_chart_path', 'Chart', '')
tf.app.flags.DEFINE_string('out_nochart_path', 'NoChart', '')
tf.app.flags.DEFINE_string('out_detect_path', 'Detect', '')
tf.app.flags.DEFINE_string('out_recog_path', 'Recongnition.txt', '')

#输出二级目录
tf.app.flags.DEFINE_string('out_chart_origin', 'origin', '')
tf.app.flags.DEFINE_string('out_chart_view', 'view', '')
tf.app.flags.DEFINE_string('out_detect_cut', 'cut', '')
tf.app.flags.DEFINE_string('out_detect_view', 'view', '')

tf.app.flags.DEFINE_bool('no_write_images', False, 'do not write images')

import detect_model
from icdar import restore_rectangle

FLAGS = tf.app.flags.FLAGS
mrc = ['{','}','|',':','[',']','!',',','\'','’','‘','_','\"','"']              #需要过滤的异常识别结果

def get_path():
    #新建一级目录
    isExists=os.path.exists(FLAGS.out_chart_path)
    if not isExists:
        os.makedirs(FLAGS.out_chart_path)
    
    isExists=os.path.exists(FLAGS.out_nochart_path)
    if not isExists:
        os.makedirs(FLAGS.out_nochart_path)
    
    isExists=os.path.exists(FLAGS.out_detect_path)
    if not isExists:
        os.makedirs(FLAGS.out_detect_path)

    #新建二级目录
    FLAGS.out_chart_origin = os.path.join(FLAGS.out_chart_path,FLAGS.out_chart_origin)
    FLAGS.out_chart_view = os.path.join(FLAGS.out_chart_path,FLAGS.out_chart_view)
    FLAGS.out_detect_cut = os.path.join(FLAGS.out_detect_path,FLAGS.out_detect_cut)
    FLAGS.out_detect_view = os.path.join(FLAGS.out_detect_path,FLAGS.out_detect_view)
    
    
    isExists=os.path.exists(FLAGS.out_chart_origin)
    if not isExists:
        os.makedirs(FLAGS.out_chart_origin)

    isExists=os.path.exists(FLAGS.out_chart_view)
    if not isExists:
        os.makedirs(FLAGS.out_chart_view)

    isExists=os.path.exists(FLAGS.out_detect_cut)
    if not isExists:
        os.makedirs(FLAGS.out_detect_cut)
    
    isExists=os.path.exists(FLAGS.out_detect_view)
    if not isExists:
        os.makedirs(FLAGS.out_detect_view)

    return


def get_images():
    '''
    find image files in test data path
    :return: list of files found
    '''
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(FLAGS.test_data_path):
        
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files


def resize_image(im, max_side_len=2400):
    '''
    resize image to a size multiple of 32 which is required by the network
    :param im: the resized image
    :param max_side_len: limit of max image size to avoid out of memory in gpu
    :return: the resized image and the resize ratio
    '''
    h, w, _ = im.shape

    resize_w = w
    resize_h = h

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % 32 == 0 else (resize_h // 32 - 0) * 32
    resize_w = resize_w if resize_w % 32 == 0 else (resize_w // 32 - 0) * 32
    im = cv2.resize(im, (int(resize_w), int(resize_h)))

    ratio_h = resize_h / float(h)
    ratio_w = resize_w / float(w)

    return im, (ratio_h, ratio_w)

#获取外接矩形框
def bbox(points):
    res = np.zeros((2,2),dtype=np.int32)
    res[0,:] = np.min(points, axis=0)
    res[1,:] = np.max(points, axis=0)
    return res

#最大类间方差
def Max_Var(data):
    bm = copy.deepcopy(data)
    bm.sort()
    mx_t = 0
    mx_sm = 0
    stx = bm[0];
    ste = bm[len(bm)-1]
    for i in range(stx,ste):
        t = i
        xt = 0
        dt = 0
        xt_sum = 0
        dt_sum = 0
        
        for j in range(len(data)):
            if data[j]>t:
                dt = dt + 1
                dt_sum = dt_sum + data[j]
            else:
                xt = xt + 1
                xt_sum = xt_sum + data[j]
        if xt>0:
            xt_sum = xt_sum / xt
            
        if dt>0:
            dt_sum = dt_sum / dt
            
        xt = xt / len(data)
        dt = dt / len(data)
        m_mx_sm = xt*xt_sum*xt_sum+dt*dt_sum*dt_sum;
        if m_mx_sm > mx_sm:
            mx_t = t
            mx_sm = m_mx_sm
    return mx_t

#最大类间方差
def Multi_Max_Var(data,n):
    bm = copy.deepcopy(data)
    bm.sort()
    mx_t = 0
    mx_sm = 0
    mx_av = 0
    th_rt = {}
    ndata = []
    if len(bm)==0:
        return ndata
    for i in range(len(bm)):
        mm = np.sqrt(bm[i][0]*bm[i][0]+bm[i][1]*bm[i][1])
        ndata.append(mm)
        mx_av = mx_av + mm
        
    mx_av = mx_av*1.0/len(bm)
    for i in range(len(ndata)):
        t = i
        xt = 0
        dt = 0
        xt_sum = 0
        dt_sum = 0
        
        for j in range(len(ndata)):
            if j!=i:
                if ndata[j]>=t:
                    dt = dt + 1
                    dt_sum = dt_sum + ndata[j]
                else:
                    xt = xt + 1
                    xt_sum = xt_sum + ndata[j]
        if xt>0:
            xt_sum = xt_sum / xt
            
        if dt>0:
            dt_sum = dt_sum / dt
            
        xt = xt / len(data)
        dt = dt / len(data)
        m_mx_sm = xt*(xt_sum-mx_av)*(xt_sum-mx_av)+dt*(dt_sum-mx_av)*(dt_sum-mx_av);
        if m_mx_sm > mx_sm:
            mx_t = t
            mx_sm = m_mx_sm
    num_mx = 0
    for i in range(len(bm)):
        if i>=mx_t and mx_sm>n:
            th_rt[bm[i][2]] = 1
            num_mx = num_mx + 1
        else:
            th_rt[bm[i][2]] = 0
            
    if num_mx>5:
        for i in range(len(bm)):
            th_rt[bm[i][2]] = 0
        
    return th_rt

def extract_box(im,boxes,th_rt):
    T = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
    for box in boxes: 
        cv2.rectangle(T,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(255,255,255),-1)
    
    #水平投影
    hhist = []
    mark=0
    for i in range(T.shape[0]):
        nums = 0
        mark=0
        for j in range(T.shape[1]):
            if T[i,j]>100 and mark==0:
                mark=1
                nums = nums + 1
            elif T[i,j]<100 and mark==1:
                mark = 0
        hhist.append(nums)
    
    #垂直投影
    vhist = []
    mark=0
    for j in range(T.shape[1]):
        nums = 0
        mark=0
        for i in range(T.shape[0]):
            if T[i,j]>100 and mark==0:
                mark=1
                nums = nums + 1
            elif T[i,j]<100 and mark==1:
                mark = 0
        vhist.append(nums)
    
    
    nboxes = []
    MY = max(hhist)
    MX = max(vhist)
    for ii in range(len(boxes)):
        cy = int((boxes[ii][1,1]+boxes[ii][0,1])/2)
        cx = int((boxes[ii][0,0]+boxes[ii][1,0])/2)
        if th_rt[ii]==1:
            if hhist[cy] > MY/2 and vhist[cx]>MX/2:
                nboxes.append(boxes[ii])
        else:
            nboxes.append(boxes[ii])
    #nboxes = np.array(nboxes)
    return nboxes


#获取垂直和水平投影
def Extract_getLC(im,ld_dt):
    #水平投影
    hhist = []
    mark=0
    for i in range(im.shape[0]):
        nums = 0
        mark=0
        for j in range(im.shape[1]):
            if im[i,j]>100 and mark==0:
                mark=1
                nums = nums + 1
            elif im[i,j]<100 and mark==1:
                mark = 0
        hhist.append(nums)
    
    #去除非零
    non_hhist = []
    for i in range(im.shape[0]):
        if hhist[i]>0:
            non_hhist.append(hhist[i])
    #T = Max_Var(non_hhist)
    T = max(non_hhist)
    for i in range(im.shape[0]):
        if  hhist[i]>2 and hhist[i]>T*0.3:#and
            hhist[i] = 255
        else:
            hhist[i] = 0
    
    #获取间隔
    start = 0
    end = 0
    mark=0
    h_gas = []
    for i in range(im.shape[0]):
        if hhist[i]>0 and mark==0:
            start = i
            mark=1
        elif hhist[i]==0 and mark==1:
            end =i
            mark = 0
            h_gas.append([start,end,end-start])
        elif i==im.shape[0]-1 and mark==1:
            end =i
            mark = 0
            h_gas.append([start,end,end-start])

    ll_h = len(h_gas)
    if ll_h>0:
        #T = Max_Var(np.array(h_gas)[:,2])
        T= np.min(np.array(h_gas)[:,2])
        h_dist = []
        for el_index in range(1,len(h_gas)):
            if h_gas[el_index][0] - h_gas[el_index-1][1]>0:
                h_dist.append(h_gas[el_index][0] - h_gas[el_index-1][1])
        T1 = np.median(np.array(h_dist))
        T = max([T,T1])

        for el_index in range(ll_h-1):
            ei = ll_h - 1 - el_index
            el = h_gas[ei]
            if el_index<ll_h-1:
                #T = max(T,h_gas[el_index][2])
                if el[0] - h_gas[ei-1][1]< T*3:  #h_gas[el_index][2]
                    for k in range(h_gas[ei-1][0],el[0]+1):
                        hhist[k]=255
    
    nh_gas = []
    mgas = 0
    start = 0
    end = 0
    for i in range(im.shape[0]):
        if hhist[i]>0 and mark==0:
            start = i
            mark=1
        elif hhist[i]==0 and mark==1:
            end =i
            mark = 0
            #if end-start>mgas:
            #mgas = end-start
            nh_gas.append([start,end])
        elif i==im.shape[0]-1 and mark==1:
            end =i
            mark = 0
            #if end-start>mgas:
            #mgas = end-start
            nh_gas.append([start,end])
    
    #垂直投影
    vhist = []
    mark=0
    for j in range(im.shape[1]):
        nums = 0
        mark=0
        for i in range(im.shape[0]):
            if im[i,j]>100 and mark==0:
                mark=1
                nums = nums + 1
            elif im[i,j]<100 and mark==1:
                mark = 0
        vhist.append(nums)
    
    bvhist = copy.deepcopy(vhist)
    #去除非零
    non_vhist = []
    for i in range(im.shape[1]):
        if vhist[i]>0:
            non_vhist.append(vhist[i])
    #T = Max_Var(non_vhist)
    T = max(non_vhist)
    for i in range(im.shape[1]):
        if vhist[i]>1 and vhist[i]>T*0.2:
            vhist[i] = 255
        else:
            vhist[i] = 0
    
    #获取间隔
    start = 0
    end = 0
    mark=0
    v_gas = []
    for i in range(im.shape[1]):
        if vhist[i]>0 and mark==0:
            start = i
            mark=1
        elif vhist[i]==0 and mark==1:
            end =i
            mark = 0
            v_gas.append([start,end,end-start])
        elif i==im.shape[1]-1 and mark==1:
            end =i
            mark = 0
            v_gas.append([start,end,end-start])
            
    lvs = len(v_gas)
    if lvs > 1:
        #T = Max_Var(np.array(v_gas)[:,2])
        T1 = np.median(np.array(v_gas)[:,2])
        v_dist = []
        for el_index in range(1,lvs):
            if(v_gas[el_index][0] - v_gas[el_index-1][1] > 0):
                v_dist.append(v_gas[el_index][0] - v_gas[el_index-1][1])
        T = Max_Var(np.array(v_dist))
        T = max([T,T1])
        for el_index in range(lvs-1):
            ei = lvs -1 - el_index
            el = v_gas[ei]
            if el_index<lvs-1:
                if el[0] - v_gas[ei-1][1]<T*4:
                    for k in range(v_gas[ei-1][1],el[0]+1):
                        vhist[k]=255
    nv_gas = []
    mgas = 0
    start = 0
    end = 0
    mT = max(bvhist)
    for i in range(im.shape[1]):
        if vhist[i]>0 and mark==0 and bvhist[i]>mT*0.3:
            start = i
            mark=1
        elif vhist[i]==0 and mark==1:
            end =i
            mark = 0
            #if end-start>mgas:
            #mgas = end-start
            nv_gas.append([start,end])
        elif i==im.shape[1]-1 and mark==1:
            end =i
            mark = 0
            #if end-start>mgas:
            #mgas = end-start
            nv_gas.append([start,end])
        
    return hhist, vhist, nh_gas, nv_gas

def Extract_Cut_Rect(im,mask,rect_dict):
    startx = 0
    endx = 0
    starty = 0
    endy = 0
    
    #统计每行数字的个数
    
    #行裁剪
    for i in range(im.shape[0]):
        mark=0
        nums = 0
        ri = im.shape[0]-1-i
        mri = ri
        for j in range(im.shape[1]):
            if im[ri,j] ==255:
                if rect_dict[mask[ri,j]][0][1,1] > mri:
                        mri = rect_dict[mask[ri,j]][0][1,1]
            if im[ri,j]==255 and mark==0:
                mark=1
                if rect_dict[mask[ri,j]][2] == 1:
                    nums = nums + 1
                
            elif im[ri,j]==0 and mark==1:
                mark = 0
        if nums > 2:
            endy = mri
            break
    
    #列裁剪
    for i in range(im.shape[1]):
        mark=0
        nums = 0
        ri = im.shape[1]-1-i
        mri = ri
        for j in range(endy):
            if im[j,ri]==255:
                if rect_dict[mask[j,ri]][0][1,0] > mri:
                    mri = rect_dict[mask[j,ri]][0][1,0]
            if im[j,ri]==255 and mark==0:
                mark=1
                if rect_dict[mask[j,ri]][2] == 1:
                    nums = nums + 1
                
            elif im[j,ri]==0 and mark==1:
                mark = 0
        if nums > 0:
            endx = mri
            break
    
    #增加像素修正
    endx = endx
    endy = endy
        
    return startx, endx, starty, endy


#更新边界函数
def Extract_Bdct(startx,endx,starty,endy,im,mask,rect_dict):
    
    xmi = startx
    xma = endx
    
    ymi = starty
    yma = endy
    
    for i in range(im.shape[0]):
        if im[i,startx]>100:
            cx = (rect_dict[mask[i,startx]][0][0,0] + rect_dict[mask[i,startx]][0][1,0])/2
            if cx>xmi and rect_dict[mask[i,startx]][0][0,0] <= xmi:
                xmi = rect_dict[mask[i,startx]][0][0,0]
        if im[i,endx]>100:
            if rect_dict[mask[i,endx]][0][1,0] > xma:
                xma = rect_dict[mask[i,endx]][0][1,0]
    
    mark_yi = 0
    mark_ya = 0          
    for i in range(im.shape[1]):
        if im[starty,i]>100:
            if rect_dict[mask[starty,i]][0][0,1] < ymi:
                ymi = rect_dict[mask[starty,i]][0][0,1]
        if im[endy,i]>100:
            if rect_dict[mask[endy,i]][0][1,1] > yma:
                yma = rect_dict[mask[endy,i]][0][1,1]
    _, mvx = divmod(xma-xmi,32)
    _, mvy = divmod(yma-ymi,32)
    mvx = int(mvx/2)+1
    mvx = int(mvy/2)+1
    xmi = xmi - mvx
    xma = xma + mvx
    ymi = ymi - mvy
    yma = yma + mvy
    return xmi, xma, ymi, yma

#规则n*m表格分割
def Reg_getLC(im,ld_dt):
    #水平投影
    hhist = []
    mark=0
    start = 0
    end = 0
    for i in range(im.shape[0]):
        nmi = np.sum(im[i,:]==255)  
        if nmi>0 and mark==0:
            start = i
            mark = 1
        elif nmi==0 and mark==1:
            end = i
            mark=0
            hhist.append([start,end])
    #垂直投影
    vhist = []
    mark=0
    start = 0
    end = 0
    for i in range(im.shape[1]):
        nmi = np.sum(im[:,i]==255)  
        if nmi>0 and mark==0:
            start = i
            mark = 1
        elif nmi==0 and mark==1:
            end = i
            mark=0
            vhist.append([start,end])
    nld_dt = []
    for mi in range(len(ld_dt)):
        m = ld_dt[mi]
        for i in range(len(hhist)):
            mkk = 0
            for j in range(len(vhist)):
                cx = (m[0,0]+m[1,0])/2
                cy = (m[1,1]+m[0,1])/2
                if cx>=vhist[j][0] and cx <= vhist[j][1] and cy>=hhist[i][0] and cy <= hhist[i][1]:
                    lm = {'e':m,'y':i,'x':j}
                    nld_dt.append(lm)
                    mkk=1
                    break
            if mkk==1:
                break
        
    return len(hhist), len(vhist), nld_dt


def detect(score_map, geo_map, timer, score_map_thresh=0.6, box_thresh=0.05, nms_thres=0.1):
    '''
    restore text boxes from score map and geo map
    :param score_map:
    :param geo_map:
    :param timer:
    :param score_map_thresh: threshhold for score map
    :param box_thresh: threshhold for boxes
    :param nms_thres: threshold for nms
    :return:
    '''
    if len(score_map.shape) == 4:
        score_map = score_map[0, :, :, 0]
        geo_map = geo_map[0, :, :, ]
    # filter the score map
    xy_text = np.argwhere(score_map > score_map_thresh)
    # sort the text boxes via the y axis
    xy_text = xy_text[np.argsort(xy_text[:, 0])]
    # restore
    start = time.time()
    text_box_restored = restore_rectangle(xy_text[:, ::-1] * 4, geo_map[xy_text[:, 0], xy_text[:, 1], :])  # N*4*2
    print('{} text boxes before nms'.format(text_box_restored.shape[0]))
    boxes = np.zeros((text_box_restored.shape[0], 9), dtype=np.float32)
    boxes[:, :8] = text_box_restored.reshape((-1, 8))
    boxes[:, 8] = score_map[xy_text[:, 0], xy_text[:, 1]]
    timer['restore'] = time.time() - start
    # nms part
    start = time.time()
    # boxes = nms_locality.nms_locality(boxes.astype(np.float64), nms_thres)
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thres)
    timer['nms'] = time.time() - start

    if boxes.shape[0] == 0:
        return None, timer

    # here we filter some low score boxes by the average score map, this is different from the orginal paper
    for i, box in enumerate(boxes):
        mask = np.zeros_like(score_map, dtype=np.uint8)
        cv2.fillPoly(mask, box[:8].reshape((-1, 4, 2)).astype(np.int32) // 4, 1)
        boxes[i, 8] = cv2.mean(score_map, mask)[0]
    boxes = boxes[boxes[:, 8] > box_thresh]

    return boxes, timer

def extract_chart(im,ld_dt,step=0):
     orect_dict = {}
     charts = []
     abml = ['(',')','（','）','\"\"']
     
     xr_le = []
     yr_le = []
     for inc in range(len(ld_dt)):
         box = bbox(ld_dt[inc])
         if box[1,1]-box[0,1]>(box[1,0]-box[0,0])*2:
             box[0,0] = box[0,0]-5
             box[1,0] = box[1,0]+5
             
             if box[0,1]<0:
                 box[0,1] = 0
             if box[0,0]<0:
                 box[0,0] = 0
             if box[1,0]>im.shape[1]:
                 box[1,0]=im.shape[1]-1
             if box[1,1]>im.shape[0]:
                 box[1,1]=im.shape[0]-1
             
          
         recs = [[box[0,0],box[0,1],box[1,0],box[0,1],
                  box[0,0],box[1,1],box[1,0],box[1,1]]]
         
         #进行阈值
         result = ocr.charRec(img = im,text_recs=recs,adjust=False)
         if len(result)>0:
             astr =[]
             mk=0
             asc = []
             
             #对异常识别结果进行过滤
             for rt in range(len(result[0][1])):
                 if result[0][1][rt] in mrc:                                                #异常的处理
                     continue
                 elif result[0][1][rt] is '.' and mk==0:                                    #第一个'.'
                     continue
                 elif ord(result[0][1][rt]) == 9675 or ord(result[0][1][rt]) == 9312:       #异常O
                     astr.append('0')
                     mk=1
                 elif result[0][1][rt]=='o' or result[0][1][rt]=='O':
                     o_0_mark = 0
                     if rt >0 and result[0][1][rt-1]>='0' and result[0][1][rt-1]<='9':
                         o_0_mark=1
                     if rt <len(result[0][1])-1 and result[0][1][rt+1]>='0' and result[0][1][rt+1]<='9':
                         o_0_mark=1
                     if o_0_mark==1:
                          astr.append('0')
                     else:
                          astr.append('O')
                         
                 else:
                     astr.append(result[0][1][rt])
                     asc.append(ord(result[0][1][rt]))
                     mk=1
             
             if ''.join(astr) in abml or len(astr)==0 or len(astr)>10:
                 continue
             
             #判断是否为数字
             isdgt = 0
             isdgt_m = 0
             tzc = 0
             mtzc = ['x','l','m','s']
             mtzc_num = 0
             mtzc_mark = 0
             dgt_num = 0
             for el in astr:
                 if (el>='0' and el<='9') or (el=='.' and isdgt_m==1):
                     isdgt_m = 1
                     #isdgt = 1
                     #break
                     dgt_num = dgt_num + 1
                 if el.lower() in mtzc:
                     mtzc_num = mtzc_num + 1
             if dgt_num >= len(astr)*1.0/3:
                 isdgt = 1
             if mtzc_num >= len(astr)*1.0/2:
                 mtzc_mark=1
             
             if ''.join(astr).replace(' ','') == '/' or ''.join(astr).replace(' ','').lower() == 'i' or ''.join(astr).replace(' ','') == '-'  or ''.join(astr).replace(' ','')=='一':
                 tzc = 1
             
             if step==0:
                 if isdgt>0 or tzc==1 or mtzc_mark==1:
                     orect_dict[inc] = [box,astr,1]
                 elif  box[1,1] - box[0,1] < im.shape[0]/3 and box[1,0] - box[0,0]<im.shape[1]/3: #ßand box[1,0] - box[0,0]>box[1,1] - box[0,1]:
                     orect_dict[inc] = [box,astr,0]
             else:
                 if isdgt>0 or tzc==1 or mtzc_mark==1:
                     orect_dict[inc] = [box,astr,1]
                 else:
                     orect_dict[inc] = [box,astr,0]
            
             if len(astr)>0:
                 yr = int((box[1,1]-box[0,1])/1)
                 xr = int((box[1,0]-box[0,0])/len(astr))
                 
                 xr_le.append(xr)
                 yr_le.append(yr)
                 
     #获取众数
     counts = np.bincount(xr_le)
     zwx = np.argmax(counts)
     counts = np.bincount(yr_le)
     zwy = np.argmax(counts)
    
     rect_dict = {}
     for inc in orect_dict:
        box = orect_dict[inc][0]
        lastr = orect_dict[inc][1]
        yr = (box[1,1]-box[0,1])/1
        xr = (box[1,0]-box[0,0])/len(lastr)
        if xr<zwx*5 and yr<zwy*4:
           rect_dict[inc] = orect_dict[inc]
     if rect_dict is  None:
        return charts

     imbk = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
     mask = np.zeros((im.shape[0],im.shape[1]),dtype=np.int32)
     for inc in rect_dict:
         box = rect_dict[inc][0]
         cv2.rectangle(imbk,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(255,255,255),-1)
         cv2.rectangle(mask,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(inc,inc,inc),-1)

     #去掉线边缘
     startx,endx,starty,endy = Extract_Cut_Rect(imbk,mask,rect_dict)
     if startx==endx or starty == endy:
         return charts
     
     #去掉上边缘
     nimbk = imbk[starty:endy,startx:endx]
     new_im = im[starty:endy,startx:endx,:]
     hhist, vhist, nh_gas, nv_gas = Extract_getLC(nimbk,ld_dt)
     #charts.append([startx,endx,starty,endy])
     cut_index = 0
     for vn in nv_gas:
         for hn in nh_gas:
             startx,endx,starty,endy = Extract_Bdct(vn[0],vn[1],hn[0],hn[1],imbk,mask,rect_dict)#[vn[0],vn[1],hn[0],hn[1]]
             if startx <0:
                 startx = 0
             if starty<0:
                 starty = 0
             if endx>im.shape[1]:
                 endx=im.shape[1]
             if endy>im.shape[0]:
                 endy = im.shape[0]
             charts.append([startx,endx,starty,endy])
             cut_index = cut_index + 1
     return charts
 
def chart_class(im,boxes):
    #识别判断是不是表格
    number_rect_num = 0        #记录矩形框包含数字的信息
    for inc in range(len(boxes)):
         box = boxes[inc]
         #边框修正
         lim = im[box[0,1]:box[1,1],box[0,0]:box[1,0],:]
         thresh = cv2.cvtColor(lim,cv2.COLOR_BGR2GRAY)
         T= mahotas.thresholding.otsu(thresh)
         thresh[thresh >T] = 255#矩阵thresh中>T的值赋值为255
         thresh[thresh <= T] = 0#矩阵thresh中<255的值赋值为0
         
         #确认背景和前景
         nm = np.sum(thresh==255)
         bk = 255
         if nm>(box[1,0]-box[0,0])*(box[1,1]-box[0,1])/2:
             bk = 0
         
         startx = 0
         endx = 0
         marks = 0
         marke = 0
         lt = box[1,0]-box[0,0]
         for i in range(lt):
             nmi = np.sum(thresh[:,i]==bk)
             if nmi>0:
                 startx = box[0,0] + i-7
                 marks =  1
                 break
        
         for i in range(lt):
             nmi = np.sum(thresh[:,lt-i-1]==bk)
             if nmi>0:
                 marke = 1
                 endx = box[1,0]-i + 6
                 break
         if marks ==0:
             startx = box[0,0]
         if marke ==0:
             endx = box[1,0]
         
         starty = 0
         endy = 0
         marks = 0
         marke = 0
         lt = box[1,1]-box[0,1]
         for i in range(lt):
             nmi = np.sum(thresh[i,:]==bk)
             if nmi>0:
                 starty = box[0,1] + i-4
                 marks =  1
                 break
        
         for i in range(lt):
             nmi = np.sum(thresh[lt-i-1:]==bk)
             if nmi>0:
                 marke = 1
                 endy = box[1,1]-i + 4
                 break
         if marks ==0:
             starty = box[0,1]
         if marke ==0:
             endy = box[1,1]
          
         recs = [[startx,starty,endx,starty,
                  startx,endy,endx,endy]]
         
         if endy-starty<lt/2:
             starty = int(starty - (endx-startx-endy+starty)/2)
             endy = int(endy + (endx-startx-endy+starty)/2)
         
         if endy-starty > endx - startx:
             ccx = (startx + endx)/2
             ccy = (endy-starty)/2
             startx = int(ccx-ccy)
             endx = int(ccx+ccy)
             
         #进行阈值
         result = ocr.charRec(img = im,text_recs=recs,adjust=False)
         if len(result)>0:
             astr =[]
             mk=0
             asc = []
             
             #对异常识别结果进行过滤
             for rt in range(len(result[0][1])):
                 if result[0][1][rt] in mrc:                                                #异常的处理
                     continue
                 elif result[0][1][rt] is '.' and mk==0:                                    #第一个'.'
                     continue
                 elif ord(result[0][1][rt]) == 9675 or ord(result[0][1][rt]) == 9312:       #异常O
                     astr.append('0')
                     mk=1
                 elif result[0][1][rt]=='o' or result[0][1][rt]=='O':
                     o_0_mark = 0
                     if rt >0 and result[0][1][rt-1]>='0' and result[0][1][rt-1]<='9':
                         o_0_mark=1
                     if rt <len(result[0][1])-1 and result[0][1][rt+1]>='0' and result[0][1][rt+1]<='9':
                         o_0_mark=1
                     if o_0_mark==1:
                          astr.append('0')
                     else:
                          astr.append('O')
                         
                 else:
                     astr.append(result[0][1][rt])
                     asc.append(ord(result[0][1][rt]))
                     mk=1
             
             #判断是否为数字
             if len(astr)>0:
                 sastr = ''.join(astr)
                 mz = ['.','*']
                 
                 isdgt = 0
                 isdgt_m = 0
                 tzc = 0
                 mtzc = ['x','l','m','s']
                 mtzc_num = 0
                 mtzc_mark = 0
                 dgt_num = 0
                 for el in astr:
                     if (el>='0' and el<='9') or (el=='.' and isdgt_m==1):
                         dgt_num = dgt_num + 1
                     if el.lower() in mtzc:
                         mtzc_num = mtzc_num + 1
                 if dgt_num >= len(astr)*1.0/2:
                     isdgt = 1
                 if mtzc_num >= len(astr)*1.0/2:
                     mtzc_mark=1
                         
                 for el_mz in mz:
                     if el_mz in astr:
                         sastr.replace(el_mz,'')
                     
                 if sastr.isdigit() or (astr[0].isdigit() and not astr[len(astr)-1].isdigit()) or sastr=='/' or isdgt==1:
                     c_n = 0
                     if sastr.isdigit():
                         for el in astr:
                             if ord(el)>200:
                                 c_n=1
                                 break
                     if c_n ==0:
                         number_rect_num = number_rect_num + 1
    return number_rect_num
 
def recog(im,load_dict):     
     #对检测的文字外框进行修正
     ld_dt = []
     imbk = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
     for box in load_dict:
         cv2.rectangle(imbk,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(255,255,255),1)
         ld_dt.append(box)
         
     
     #表格判断
     rs, cs, nld_dt = Reg_getLC(imbk,ld_dt)
     xs = []
     for ii in range(rs):
         m = []
         for jj in range(cs):
             if jj==0:
                 m.append('')
             else:
                 m.append(',')
         xs.append(m)
     

     for inc in nld_dt:
         box = inc['e']
         yy = inc['y']
         xx = inc['x']
         #边框修正
         lim = im[box[0,1]:box[1,1],box[0,0]:box[1,0],:]
         thresh = cv2.cvtColor(lim,cv2.COLOR_BGR2GRAY)
         T= mahotas.thresholding.otsu(thresh)
         thresh[thresh >T] = 255#矩阵thresh中>T的值赋值为255
         thresh[thresh <= T] = 0#矩阵thresh中<255的值赋值为0
         
         #确认背景和前景
         nm = np.sum(thresh==255)
         bk = 255
         if nm>(box[1,0]-box[0,0])*(box[1,1]-box[0,1])/2:
             bk = 0
         
         startx = 0
         endx = 0
         marks = 0
         marke = 0
         lt = box[1,0]-box[0,0]
         for i in range(lt):
             nmi = np.sum(thresh[:,i]==bk)
             if nmi>0:
                 startx = box[0,0] + i-7
                 marks =  1
                 break
        
         for i in range(lt):
             nmi = np.sum(thresh[:,lt-i-1]==bk)
             if nmi>0:
                 marke = 1
                 endx = box[1,0]-i + 6
                 break
         if marks ==0:
             startx = box[0,0]
         if marke ==0:
             endx = box[1,0]
         
         starty = 0
         endy = 0
         marks = 0
         marke = 0
         lt = box[1,1]-box[0,1]
         for i in range(lt):
             nmi = np.sum(thresh[i,:]==bk)
             if nmi>0:
                 starty = box[0,1] + i-4
                 marks =  1
                 break
        
         for i in range(lt):
             nmi = np.sum(thresh[lt-i-1:]==bk)
             if nmi>0:
                 marke = 1
                 endy = box[1,1]-i + 4
                 break
         if marks ==0:
             starty = box[0,1]
         if marke ==0:
             endy = box[1,1]
          
         recs = [[startx,starty,endx,starty,
                  startx,endy,endx,endy]]

         if endy-starty<lt/2:
             starty = int(starty - (endx-startx-endy+starty)/2)
             endy = int(endy + (endx-startx-endy+starty)/2)
         
         #进行阈值
         result = ocr.charRec(img = im,text_recs=recs,adjust=False)
         if len(result)>0:
             astr =[]
             mk=0
             asc = []
             
             #对异常识别结果进行过滤
             for rt in range(len(result[0][1])):
                 if result[0][1][rt] in mrc:                                                #异常的处理
                     continue
                 elif result[0][1][rt] is '.' and mk==0:                                    #第一个'.'
                     continue
                 elif ord(result[0][1][rt]) == 9675 or ord(result[0][1][rt]) == 9312:       #异常O
                     astr.append('0')
                     mk=1
                 elif result[0][1][rt]=='o' or result[0][1][rt]=='O':
                     o_0_mark = 0
                     if rt >0 and result[0][1][rt-1]>='0' and result[0][1][rt-1]<='9':
                         o_0_mark=1
                     if rt <len(result[0][1])-1 and result[0][1][rt+1]>='0' and result[0][1][rt+1]<='9':
                         o_0_mark=1
                     if o_0_mark==1:
                          astr.append('0')
                     else:
                          astr.append('O')
                         
                 else:
                     astr.append(result[0][1][rt])
                     asc.append(ord(result[0][1][rt]))
                     mk=1
             
             if xs[yy][xx] == ',' or xs[yy][xx] == '':
                 xs[yy][xx] = xs[yy][xx]+''.join(astr).lower()
             else:
                 xs[yy][xx] = xs[yy][xx]+','+''.join(astr).lower()
                 
         #cv2.rectangle(org_im,(startx,starty),(endx,endy),(55,255,155),1)
     #cv2.imwrite(FLAGS.det_data_path+'/'+file,org_im)
     
     return rs, cs, xs


def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis + 1) % 4, (min_axis + 2) % 4, (min_axis + 3) % 4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]


def main(argv=None):
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu_list
    
    #目前查询并新建
    get_path()

    graph=tf.Graph()
    #with tf.get_default_graph().as_default():
    with graph.as_default():
        input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
        #global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        global_step = tf.train.get_or_create_global_step()
        f_score, f_geometry = detect_model.detect_model(input_images, is_training=False)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            ckpt_state = tf.train.get_checkpoint_state(FLAGS.checkpoint_path)
            model_path = os.path.join(FLAGS.checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
            print('Restore from {}'.format(model_path))
            saver.restore(sess, model_path)
    fin = open(FLAGS.out_recog_path,'w')                #记录识别结果txt

    im_fn_list = get_images()
    for im_fn in im_fn_list:
        print('--'+im_fn)
        ft_file_path = im_fn.split('/')                                    #父级目录
        ft_file_path = ft_file_path[len(ft_file_path)-2]
        im = cv2.imread(im_fn)
        charts = []
        if im is not None and im.shape[0]>=32 and im.shape[1]>=32:
            charts.append([0,im.shape[1],0,im.shape[0]])
            for itr in range(3):
                ncharts = []
                for cht in charts:
                    if cht[1]-cht[0]<32 or cht[3]-cht[2]<32:
                        continue
                    
                    #检测
                    cim = im[cht[2]:cht[3],cht[0]:cht[1],:]
                    
                    if cim.shape[0]<32 and cim.shape[1]<32:
                        cim = cv2.resize(cim,(32,32))
                    elif cim.shape[0]<32:
                        rte = 32//cim.shape[0]
                        cim = cv2.resize(cim,(cim.shape[1]*rte,32))
                    elif cim.shape[1]<32:
                        rte = 32//cim.shape[1]
                        cim = cv2.resize(cim,(32,cim.shape[0]*rte))
                    
                    start_time = time.time()
                    im_resized, (ratio_h, ratio_w) = resize_image(cim)
    
                    timer = {'net': 0, 'restore': 0, 'nms': 0}
                    start = time.time()
                    score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                    timer['net'] = time.time() - start
    
                    boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                    print('{} : net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
                        im_fn, timer['net'] * 1000, timer['restore'] * 1000, timer['nms'] * 1000))
                    
                    if boxes is not None:
                        boxes = boxes[:, :8].reshape((-1, 4, 2))
                        boxes[:, :, 0] /= ratio_w
                        boxes[:, :, 1] /= ratio_h
                    else:
                        continue
                            
                    
    
                    duration = time.time() - start_time
                    print('[timing] {}'.format(duration))
                    
                    nboxes = []
                    cl_inputs = []
                    nindex_i = 0
                    for ii in range(len(boxes)):
                         ps = boxes[ii]
                         box = bbox(ps)
                         if box[0,1]<0:
                             box[0,1] = 0
                         if box[0,0]<0:
                             box[0,0] = 0
                         if box[1,0]>cim.shape[1]:
                             box[1,0]=cim.shape[1]
                         if box[1,1]>cim.shape[0]:
                             box[1,1]=cim.shape[0]
                             
                         if box[1,0] - box[0,0]>0 and box[1,1]-box[0,1]>0:
                             nboxes.append(box)
                             cl_inputs.append([box[1,0]-box[0,0],box[1,1]-box[0,1],nindex_i])
                             nindex_i = nindex_i + 1
                    th_data = []
                    if itr>0:
                        th_rt = Multi_Max_Var(cl_inputs,80)
                        th_data = extract_box(cim,nboxes,th_rt)
                        '''for ii in range(len(cl_inputs)):
                            if th_rt[cl_inputs[ii][2]] == 0:
                                th_data.append(nboxes[ii])'''
                    else:
                        th_data = nboxes
                
                    if th_data is not None:
                        number_rect_num = chart_class(cim,th_data)
                        if itr<3:
                            #根据数字框的个数判断是否存在表格
                            if number_rect_num>4:
                                 u_charts = extract_chart(cim,th_data,itr)
                                 
                                 for ui in range(len(u_charts)):
                                     ut = u_charts[ui]
                                     mtsu = [ut[0]+cht[0],ut[1]+cht[0],ut[2]+cht[2],ut[3]+cht[2]]
                                     if mtsu[0]<0:
                                         mtsu[0] = 0
                                     if mtsu[1]>im.shape[1]:
                                         mtsu[1] = im.shape[1]
                                     if mtsu[2]<0:
                                         mtsu[2] = 0
                                     if mtsu[3]>im.shape[0]:
                                         mtsu[3] = im.shape[0]
                                     ncharts.append(mtsu)
                        else:
                            if number_rect_num>4:
                                 ncharts.append(cht)
                
                del charts
                charts = copy.deepcopy(ncharts)
            
            #进行重新检测和识别
            chart_number = 0
            bim = copy.deepcopy(im)
            num_charts = -1
            #eim = copy.deepcopy(im)
            result_charts = OrderedDict()
            charts_coord_index = 0
            for itr in charts:
                if cht[1]-cht[0]<32 or cht[3]-cht[2]<32:
                    continue
                fin.writelines(ft_file_path+'_'+os.path.basename(im_fn).split('.')[0]+'_'+str(chart_number)+'.jpg'+'\n')
                chart_number = chart_number + 1
                
                
                #保存表格提取结果
                cv2.rectangle(bim,(itr[0],itr[2]),(itr[1],itr[3]),(0,0,255),1)
                
                #检测
                dim = copy.deepcopy(im[itr[2]:itr[3],itr[0]:itr[1],:])
                dim = dim[:, :, ::-1]
                eim = copy.deepcopy(dim)
                
                if dim.shape[0]<32 or dim.shape[1]<32:
                    continue
                im_resized, (ratio_h, ratio_w) = resize_image(dim)
                score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
                boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
                
                if boxes is not None:
                    boxes = boxes[:, :8].reshape((-1, 4, 2))
                    boxes[:, :, 0] /= ratio_w
                    boxes[:, :, 1] /= ratio_h
                
                
                #进行聚类删选
                
                # save to file
                if boxes is not None:
                    #记录表格坐标
                    lb = (itr[0], itr[2])
                    rb = (itr[1], itr[2])
                    rt = (itr[1], itr[3])
                    lt = (itr[0], itr[3])
                    result_charts[str(charts_coord_index)] = OrderedDict({
                           'coordinate': {
                           'p_1': OrderedDict({
                                              'y': float(lb[1]),
                                              'x': float(lb[0])
                                              }),
                           'p_2': OrderedDict({
                                              'y': float(lt[1]),
                                              'x': float(lt[0])
                                              }),
                           'p_3': OrderedDict({
                                              'y': float(rt[1]),
                                              'x': float(rt[0])
                                              }),
                           'p_4': OrderedDict({
                                              'y': float(rb[1]),
                                              'x': float(rb[0])
                                              }),
                           },
                           'content': ''
                           })
                    charts_coord_index = charts_coord_index + 1

                    #表格中的文本
                    result = OrderedDict()
                    coord_index = 0
                    num_charts = num_charts + 1
                    for box in boxes:
                        # to avoid submitting errors
                        box = sort_poly(box.astype(np.int32))
                        if np.linalg.norm(box[0] - box[1]) < 0 or np.linalg.norm(box[3] - box[0]) < 0:
                            continue
                        lb = (box[0, 0], box[0, 1])
                        rb = (box[1, 0], box[1, 1])
                        rt = (box[2, 0], box[2, 1])
                        lt = (box[3, 0], box[3, 1])
                        result[str(coord_index)] = OrderedDict({
                            'coordinate': {
                                'p_1': OrderedDict({
                                    'y': float(lb[1]),
                                    'x': float(lb[0])
                                }),
                                'p_2': OrderedDict({
                                    'y': float(lt[1]),
                                    'x': float(lt[0])
                                }),
                                'p_3': OrderedDict({
                                    'y': float(rt[1]),
                                    'x': float(rt[0])
                                }),
                                'p_4': OrderedDict({
                                    'y': float(rb[1]),
                                    'x': float(rb[0])
                                }),
                            },
                            'content': ''
                        })
                        coord_index += 1
                        cv2.polylines(eim, [box.astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 255, 0), thickness=1)
                
                
                    if not FLAGS.no_write_images:
                        file_name = os.path.basename(im_fn)
                        name_len = len(file_name)
                        img_path = os.path.join(FLAGS.out_detect_cut, ft_file_path+'_'+file_name[0:name_len-4] + '_' + str(num_charts) + file_name[len(file_name)-4:name_len])
                        cv2.imwrite(img_path, dim)
                        
                        img_path = os.path.join(FLAGS.out_detect_view, ft_file_path+'_'+file_name[0:name_len-4] + '_' + str(num_charts) + file_name[len(file_name)-4:name_len])
                        cv2.imwrite(img_path, eim)
                    
                    
                    
                        
                    txt_id = ft_file_path+'_'+os.path.basename(im_fn).split('.')[0] + '_' + str(num_charts)
                    outputs_dir = os.path.join(FLAGS.out_detect_cut, 'jsondata')
                    if not os.path.exists(outputs_dir):
                        os.mkdir(outputs_dir)
                    with open(os.path.join(outputs_dir, '{}.json'.format(txt_id)), 'w') as f:
                        json.dump(result, f, indent=4)
                    
                    nboxes = []
                    for ps in boxes:
                         box = bbox(ps)
                         if box[0,1]<0:
                             box[0,1] = 0
                         if box[0,0]<0:
                             box[0,0] = 0
                         cv2.rectangle(eim,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(0,0,255),1)
                         if box[1,0] - box[0,0]>0 and box[1,1]-box[0,1]>0:
                             nboxes.append(box)
                    rs, cs, xs = recog(dim,nboxes)
                    
                    
                    #文本输出修正
                    for ii in range(rs):
                         mk = 0
                         for jj in range(cs):
                             if jj==0:
                                 if xs[ii][jj]!='':
                                     mk=1
                                 fin.writelines(xs[ii][jj])
                             else:
                                 fin.writelines(xs[ii][jj])
                         fin.writelines('\n')
                    fin.writelines('\n')
                    
                    
    
                    
            if chart_number>0:
                #保存原始图片
                cv2.imwrite(FLAGS.out_chart_origin+'/'+ft_file_path+'_'+os.path.basename(im_fn),im)
                cv2.imwrite(FLAGS.out_chart_view+'/'+ft_file_path+'_'+os.path.basename(im_fn),bim)
                #保存表格的坐标到json
                txt_id = ft_file_path+'_'+os.path.basename(im_fn).split('.')[0]
                outputs_dir = os.path.join(FLAGS.out_chart_origin, 'jsondata')
                if not os.path.exists(outputs_dir):
                    os.mkdir(outputs_dir)
                with open(os.path.join(outputs_dir, '{}.json'.format(txt_id)), 'w') as f:
                    json.dump(result_charts, f, indent=4)
            else:
                cv2.imwrite(FLAGS.out_nochart_path+'/'+ft_file_path+'_'+os.path.basename(im_fn),im)

                
    fin.close()

if __name__ == '__main__':
    tf.app.run()
