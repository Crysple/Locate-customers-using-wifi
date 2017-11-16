#-*- coding:utf-8 -*-

"""
大体思路：信号强的相关联
@author: lmy
"""

import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split

def ttt():
    return lambda : defaultdict(lambda : defaultdict(lambda :0))
class Wifi_clf(object):
    def __init__(self):
        self.signal_dis = 7
        self.wifi_to_shops = defaultdict(ttt)
        self.has_train = False
        self.wifi_info_index = 5
    def fit(self,X_train, y_train):
        self.has_train = False
        self.wifi_to_shops = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :0)))
        for line,ans in zip(X_train,y_train):
            wifi = sorted([wifi.split('|') for wifi in line[self.wifi_info_index].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
            self.wifi_to_shops[wifi[0]][wifi[1]][ans] += 1

        self.has_train = True
    def predict(self,X_test):
        if not self.has_train:
            return
        mall = X_test[0][1]
        user_shop_hehavior = pd.read_csv('unclean_mall/'+mall+'.csv')
        shop_val_list = user_shop_hehavior['shop_id'].values.tolist()
        #找出最大的shop
        dic = defaultdict(lambda :0)    
        max_num, most_shop_id = 0, None
        for shop in shop_val_list:
            dic[shop]+=1
            if dic[shop]>max_num:
                max_num = dic[shop]
                most_shop_id = shop


        preds = []
        for line in X_test:
            index = 0
            while True:
                try:
                    if index==5:
                        pred_one = most_shop_id
                        break
                    wifi = sorted([wifi.split('|') for wifi in line[self.wifi_info_index].split(';')],key=lambda x:int(x[1]),reverse=True)[index]
                    counter = defaultdict(lambda : 0)
                    for signal,shop_dict in self.wifi_to_shops[wifi[0]].items():
                        if abs(int(signal)-int(wifi[1]))<self.signal_dis:
                            for shop_id, cnt in self.wifi_to_shops[wifi[0]][signal].items():
                                counter[shop_id] += cnt
                    pred_one = sorted(counter.items(),key=lambda x:x[1],reverse=True)[0][0]
                    break
                except:
                    index+=1
            preds.append(pred_one)
        return preds


    
