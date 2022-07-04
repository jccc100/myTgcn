# _*_ coding: utf-8 _*_
# @time     :2022/7/4 15:08
# @Author   :jc
# @File     :test.py

import csv
import pandas as pd
# data=csv.reader(open("sz_speed.csv"))
# # print(data)
# for line in data:
#     print(line)
# data=pd.read_csv("sz_speed.csv")
# print(data.shape)
# data2=pd.read_csv("pems08.csv")
# print(data2.shape)
data=pd.read_csv("los_adj.csv")
print(data.shape)
data2=pd.read_csv("pems08_adj.csv")
print(data2.shape)