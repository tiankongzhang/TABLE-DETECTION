# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 20:39:52 2018

@author: Zhou WenZhang
"""

# -*- coding: utf-8 -*
import os

import json

import ocr
import cv2
import mahotas
import numpy as np
import tensorflow as tf


tf.app.flags.DEFINE_string('test_data_path', 'test_data', '')
tf.app.flags.DEFINE_string('reg_data_path', 'reg_data', '')


FLAGS = tf.app.flags.FLAGS


#获取外接矩形框
def bbox(points):
    res = np.zeros((2,2),dtype=np.int32)
    res[0,:] = np.min(points, axis=0)
    res[1,:] = np.max(points, axis=0)
    return res


#规则n*m表格分割
def getLC(im,ld_dt):
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
    #print(hhist)
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


mrc = ['{','}','|',':','[',']','!',',','\'','’','‘','_']              #需要过滤的异常识别结果
nondigit = ['⑦']
def main(argv=None):
    #创建目录
    isExists=os.path.exists(FLAGS.reg_data_path)
    if not isExists:
        os.makedirs(FLAGS.reg_data_path)
    
    fin = open(FLAGS.reg_data_path+'/reg.txt','w')                #记录识别结果txt
    
    #开始读取文件目录图片识别并保存结果
    for root, dirs, files in os.walk(FLAGS.test_data_path):
        
        #进行排序，顺序识别和检测
        file_dicts = {}
        kindex = []
        for fnx in range(len(files)):
            if files[fnx].endswith('.DS_Store'):
                continue
            file = files[fnx]               
            kindex.append(int(os.path.splitext(file)[0]))   
            file_dicts[int(os.path.splitext(file)[0])] = file
        kindex.sort()
        
        #开始识别
        for fnx in range(len(kindex)):   
            iix = kindex[fnx]
            file = file_dicts[iix]
            print(file)
            
            if os.path.splitext(file)[1] == '.png' or os.path.splitext(file)[1] == '.jpg':
                if os.path.exists(FLAGS.test_data_path+'/jsondata/'+os.path.splitext(file)[0]+".json"):
                    with open(FLAGS.test_data_path+'/jsondata/'+os.path.splitext(file)[0]+".json",'r') as load_f:
                         fin.writelines(file+'\n')
                         
                         load_dict = json.load(load_f)
                         im = np.array(cv2.imread(FLAGS.test_data_path+'/'+file))
                         
                         #对检测的文字外框进行修正
                         ld_dt = []
                         imbk = np.zeros((im.shape[0],im.shape[1]),dtype=np.uint8)
                         for inc in load_dict:
                             ps = []
                             for p in load_dict[inc]['coordinate']:  
                                 m = [int(load_dict[inc]['coordinate'][p]['x']),
                                      int(load_dict[inc]['coordinate'][p]['y'])]
                                 ps.append(m)
                             box = bbox(ps)
                             if box[0,1]<0:
                                 box[0,1] = 0
                             if box[0,0]<0:
                                 box[0,0] = 0
                             cv2.rectangle(imbk,(box[0,0],box[0,1]),(box[1,0],box[1,1]),(255,255,255),1)
                             ld_dt.append(box)
                             
                         
                         #表格判断
                         rs, cs, nld_dt = getLC(imbk,ld_dt)
                         xs = []
                         for ii in range(rs):
                             m = []
                             for jj in range(cs):
                                 if jj==0:
                                     m.append('')
                                 else:
                                     m.append(',')
                             xs.append(m)
                         
        
                         number_rect_num = 0        #记录矩形框包含数字的信息
                         for inc in nld_dt:
                             box = inc['e']
                             yy = inc['y']
                             xx = inc['x']
                             #print(box)
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
                             #print(recs)
                             if endy-starty<lt/2:
                                 starty = int(starty - (endx-startx-endy+starty)/2)
                                 endy = int(endy + (endx-startx-endy+starty)/2)
                             
                             #进行阈值
                             #lim = im[box[0,1]:box[1,1],box[0,0]:box[1,0],:]
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
                                 if ''.join(astr).isdigit():
                                     c_n = 0
                                     for el in astr:
                                         if ord(el)>200:
                                             c_n=1
                                             break
                                     if c_n ==0:
                                         number_rect_num = number_rect_num + 1
                                 if xs[yy][xx] == ',' or xs[yy][xx] == '':
                                     xs[yy][xx] = xs[yy][xx]+''.join(astr).lower()
                                 else:
                                     xs[yy][xx] = xs[yy][xx]+','+''.join(astr).lower()
                         
                         
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
                         
                     
    fin.close()

if __name__ == '__main__':
    tf.app.run()
