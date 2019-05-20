#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 18 21:12:23 2019

@author: qimindeng
"""

import random
import os 
import jieba 
import math

stopw=['记者','年月日','报道','违法','招聘广告','京备','编号','安备','发布','版权所有','网站','信息','代码','好友',
       '分享','内容','版权','报警','本站','首页','日日','视频','客户端','管网','一个','本网','公网','本报讯','手机',
       '许可证','在线','转载','微信','作者','新闻网','邮箱','扫一扫','新华社','许可','这是','评论','广告','根据','登录',
       '二维码','报社','违法行为','举报电话','互联网','电话','不良信息','新闻','安全监管','法律法规','联系','注册','日电',
       '什么','网络','责任编辑','朋友圈','好友','备案号','联系方式','微博','邮编','举报','友情链接','公司','通报','未经许可',
       '违规','稿件','图片','来源','要闻','正文','编辑','日报','标题','中国','新浪','服务','我们','点击数','诚聘','客服热线',
       '不良信息','举报电话','扫描','官方','发布','有限公司','声明','地址','时间','通讯员','下载','热线','报道','百度']
    

def textParse(line):    #input is line, #output is word list
    seglist = jieba.cut(line,cut_all=False,HMM=True) 
    newword=[] 
    for i in seglist: 
        if i not in stopw and 1 < len(i) < 8:
            newword.append(i)        
            word_list = ' '.join(list(newword))        
    return word_list

def get_filename(filep,test_ratio):
    L=os.listdir(filep)
    newfile=[] 
    for file in L:
        filename = filep + file
        if os.path.splitext(filename)[1]==".txt":
            newfile.append(filename)
    trainfile= random.sample(newfile, math.floor(test_ratio*len(L)))
    testfile=list(set(newfile).difference(set(trainfile)))  
    return trainfile,testfile    

def get_train_test(filepath):
    data,datatarget=[],[]
    testdata,testtarget=[],[]
    L=os.listdir(filepath)
    for i in L:
        if os.path.isfile(i) : 
            pass
        else:
            filename=filepath+i+'/'
            trainfile,testfile=get_filename(filename,0.75)  
            for n in trainfile:
                with open(n,'r',encoding = 'utf-8') as f:
                    for line in f.readlines():
                        word_list = textParse(line)       
                data.append(word_list)
                datatarget.append(i)
            for n in testfile:
                with open(n,'r',encoding = 'utf-8') as f:
                    for line in f.readlines():
                        word_list = textParse(line) 
                testdata.append(word_list)
                testtarget.append(i)
    return data,datatarget,testdata,testtarget

filepath='/Users/qimindeng/climate_nlp/all/情感极性/'
data,datatarget,testdata,testtarget=get_train_test(filepath)

from sklearn.naive_bayes import MultinomialNB,GaussianNB
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

vt=TfidfVectorizer(min_df=0.1,max_df=0.8)
v=CountVectorizer(min_df=0.1,max_df=0.8)
train_data=vt.fit_transform(data)
test_data=vt.transform(testdata)

clf=MultinomialNB(alpha=0.1)
clf.fit(train_data,datatarget)
pred=clf.predict(test_data)

count=0
for l,r in zip(pred,testtarget):
    if l==r:
        count+=1
print(count/len(testtarget))
