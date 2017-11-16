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
class WIFI_SS_clf(object):
    def __init__(self):
        self.signal_dis = 18 #12
        self.wifi_to_shops = defaultdict(ttt)
        self.has_train = False
        self.wifi_info_index = 5
    def fit(self,X_train, y_train):
        self.has_train = False
        self.connect = defaultdict(lambda :set())
        self.wifi_to_shops = defaultdict(lambda : defaultdict(lambda : defaultdict(lambda :0)))
        for line,ans in zip(X_train,y_train):
            wifi = sorted([wifi.split('|') for wifi in line[self.wifi_info_index].split(';')],key=lambda x:int(x[1]),reverse=True)[0]
            self.wifi_to_shops[wifi[0]][wifi[1]][ans] += 1
            wifi_list = [wifi.split('|') for wifi in line[self.wifi_info_index].split(';')]
            for wifi in wifi_list:
                if wifi[2] == 'true' or wifi[2] == True:
                    self.connect[wifi[0]].add(ans)
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
            pred_one = set()
            fir = 6
            for wifi in [wifi.split('|') for wifi in line[self.wifi_info_index].split(';')]:
                if wifi[2]=='true' or wifi[2]==True:
                    pred_one = pred_one|self.connect[wifi[0]]
            while True:
                try:
                    if index==11:
                        if fir==6:
                            pred_one.add(most_shop_id)
                        break
                    wifi = sorted([wifi.split('|') for wifi in line[self.wifi_info_index].split(';')],key=lambda x:int(x[1]),reverse=True)[index]
                    counter = defaultdict(lambda : 0)
                    for signal,shop_dict in self.wifi_to_shops[wifi[0]].items():
                        if abs(int(signal)-int(wifi[1]))<self.signal_dis:
                            for shop_id, cnt in self.wifi_to_shops[wifi[0]][signal].items():
                                counter[shop_id] += cnt
                    pred_one_list = sorted(counter.items(),key=lambda x:x[1],reverse=True)
                    if pred_one_list==[]:
                        raise BaseException()
                    for i,j in pred_one_list:
                        pred_one.add(i)
                    if fir>0:
                        fir-=1
                        index+=1
                        continue
                    break
                except:
                    index+=1
            preds.append(list(pred_one))
        return preds
    def calcu_acc(self,X_train,y_train,X_test,y_test):
        mall = X_test[0][1]
        self.fit(X_train,y_train)
        wifi_preds = self.predict(X_test)
        nnn = 0
        for i,j in list(zip(wifi_preds,y_test)):
            if j not in i:
                nnn+=1
        merror = (nnn)/len(y_test)
        print(mall,1-merror,nnn,len(y_test))
        return nnn


    
