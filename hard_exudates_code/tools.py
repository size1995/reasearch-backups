import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from mxnet import nd

#dictionary list获取,输入dictionary.txt file name,(格式str,txt文件在根目录下),
def get_dictionary(item):
    dic_list=[]
    f=open(item)
    line = f.readline()
    while line:
        dic=line.strip('\n')
        dic_list.append(dic)
        line = f.readline()
    return dic_list

#生成lst文件,输入xxx.csv,xxx.lst
def lst_file_creator(csv_filename,lst_name):
        df = pd.read_csv(csv_filename, header=0)
        f = open(lst_name, 'w')
        num = 0
        for (name, series) in df.iterrows():
            name = name
            series = series
            item_num=len(series)
            for i in range(item_num):
                if i ==0:
                    s = str(series[0]) + '\t'
                else:
                    s=s+str(series[i]) + '\t'
            s=s+str(series[0])+'.jpg'+ '\n'
            f.write(s)
            num+=1
            print(num)
        f.close()

#根据输出向量生成对应的label,输入对应的dictionary.txt文件，待转换0,1列表
def label_out(dictionary,bool_list):
    dic_list=get_dictionary(dictionary)
    index_list = []
    label_out_list=[]
    for i in range(len(bool_list)):
        if bool_list[i] == 1:
            index_list.append(i)
    for index in index_list:
        label_out_list.append(dic_list[index])

    return label_out_list

#评价指标，计算TP，输入预测结果与真实值标签，均为numpy格式
def TP_cal(pred,label):
    tp = pred * label
    TP=np.sum(tp)
    return TP
#评价指标，计算FP，输入预测结果与真实值标签，均为numpy格式
def FP_cal(pred,label):
    tp=pred * label
    fp=pred-tp
    FP=np.sum(fp)
    return FP
#评价指标，计算FN，输入预测结果与真实值标签，均为numpy格式
def FN_cal(pred,label):
    tp=pred * label
    fn=label-tp
    FN=np.sum(fn)
    return FN
#评价指标，计算TN，输入预测结果与真实值标签，均为numpy格式
def TN_cal(pred,label):
    tn=(1-pred)*(1-label)
    TN=np.sum(tn)
    return TN

# 评价指标，计算Precision = (预测为1且正确预测的样本数)/(所有预测为1的样本数) = TP/(TP+FP)
def Precision_cal(pred,label):
    TP=TP_cal(pred,label)
    FP=FP_cal(pred,label)
    precision=TP/(TP+FP+1e-10)
    return precision
#评价指标，计算Recall = (预测为1且正确预测的样本数)/(所有真实情况为1的样本数) = TP/(TP+FN)
def Recall_cal(pred,label):
    TP=TP_cal(pred,label)
    FN=FN_cal(pred, label)
    recall=TP/(TP+FN+1e-10)
    return recall
#评价指标，F-score计算，输入B值，pred，label
def FB_score_cal(B,pred,label):
    precision=Precision_cal(pred,label)
    recall=Recall_cal(pred,label)
    FB_score=((1+B**2)*(precision*recall))/((B**2)*precision+recall+1e-10)
    return FB_score
#评价指标，F_score计算，输入B值，precision，recall
def FB_score_cal_PR(B,precision,recall):
    FB_score = ((1 + B ** 2) * (precision * recall)) / ((B ** 2) * precision + recall+1e-10)
    return FB_score
#随机乱序列表生成
def random_list(lenth):
    normal_list=[]
    for i in range(lenth):
        normal_list.append(i)
    random.shuffle(normal_list)
    return normal_list
#生成训练集、验证集文件夹，csv表格
def generate_train_val(val_num,sum_num,datalist,train_path,val_path):
    image_list=random_list(sum_num)
    list_train=image_list[0:sum_num-val_num]
    list_val=image_list[sum_num-val_num:]
    train_folder_exists=os.path.exists(train_path)
    val_folder_exists=os.path.exists(val_path)
    if not train_folder_exists:
        os.makedirs(train_path)
    if not val_folder_exists:
        os.makedirs(val_path)
    data_list=[]
    with open(datalist) as f:
        line=f.readline()
        while line:
            data_list.append(line)
            line=f.readline()

    train_list=[]
    val_list=[]

    for i in list_train:
        train_list.append(data_list[i])
    for i in list_val:
        val_list.append(data_list[i])

    f1 = open('train.txt', 'w')
    for i in train_list:
        f1.write(i)
        f1.write("\n")
    f1.close()

    f2 = open('val.txt','w')
    for i in val_list:
        f2.write(i)
        f2.write('\n')
    f2.close
#测试数据显示，显示一个batch的图片和标签,img为3*h*w的0-255 ndarray,pred为网络输出nd格式，label为nd格式的标签,best_ths为最佳阈值列表
def image_multi_label_out(img,pred,label,best_ths,dictionary):
    image=nd.transpose(img,axes=(1,2,0)).asnumpy().astype('uint8')
    best_ths=best_ths.asnumpy()
    label=label.asnumpy().tolist()
    label_true=label_out(dictionary,label)
    pred=nd.sigmoid(pred)

    best_ths=nd.array(best_ths)
    label_pred=pred>best_ths
    label_pred=label_pred.asnumpy().tolist()
    label_pred=label_out(dictionary,label_pred)
    print('label_pred=',label_pred)
    print('label_true=',label_true)
    plt.imshow(image)
def image_multi_label_out1(img,pred,label,dictionary):
    image=nd.transpose(img,axes=(1,2,0)).asnumpy().astype('uint8')
    label=label.asnumpy().tolist()
    label_true=label_out(dictionary,label)

    label_pred=pred.asnumpy().tolist()
    label_pred=label_out(dictionary,label_pred)
    print('label_pred=',label_pred)
    print('label_true=',label_true)
    plt.imshow(image)

#试验区域
