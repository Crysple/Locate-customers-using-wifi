import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib
from sklearn.ensemble import RandomForestClassifier
#from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing


class RF_clf(object):
	def __init__(self):
		self.clf = RandomForestClassifier(n_jobs=-1, random_state=0,n_estimators=670,max_features=6)#n_estimators=670 max_features = 6
		self.has_fit = False
		self.wifi_set = set()
		self.user_set = set()
		self.WIFI_NUM = 5
	def pro_wifi(self,wifirow):
		WIFI_NUM = self.WIFI_NUM
		res = []
		fakewifi = dict()
		fakewifi['ssid'] = 0
		fakewifi['signal'] = 0
		wifi_row_list = list()
		wifi_str_list = wifirow.split(';')
		for wifi_str in wifi_str_list:
			onewifi_list = wifi_str.split('|')
			wifi_dict = dict()
			wifi_dict['ssid'] = onewifi_list[0][2:]
			wifi_dict['signal'] = onewifi_list[1]
			wifi_dict['connect'] = 1 if onewifi_list[2]=='True' else 0
			if not self.has_fit:
				self.wifi_set.add(wifi_dict['ssid'])
			else:
				if wifi_dict['ssid'] not in self.wifi_set:
					continue
			wifi_row_list.append(wifi_dict)
		wifi_row_list = sorted(wifi_row_list,key = lambda item:100 if item['connect']==1 else item['signal'])
		wlen = len(wifi_row_list)
		if wlen<WIFI_NUM:
			for i in range(WIFI_NUM-wlen):
				wifi_row_list.append(fakewifi)
		for i in range(WIFI_NUM):
			res.append(wifi_row_list[i]['ssid'])
			res.append(wifi_row_list[i]['signal'])
		return res
	def process_data(self,x):
		self.wifi_set.add(0)
		#self.user_set.add(-1)
		x = np.delete(x,[0,1,2],1)
		# longitude,latitude,wifi_infos
		res = []
		for i,row in enumerate(x):
			#row[0] = row[0][2:]
			ret = self.pro_wifi(row[2])
			now_row = []
			#hour = int(row[1].split(' ')[1][:2])
			#date = int(row[1].split(' ')[0][-2:])
			#now_row.append(date)
			#now_row.append(hour)
			for j in range(2):
				now_row.append(row[j])
			now_row.extend(ret)
			res.append(now_row)
		res = np.array(res)
		if not self.has_fit:
			self.le = preprocessing.LabelEncoder()
			self.le.fit(list(self.wifi_set))

		for i in range(self.WIFI_NUM):
			i = i*2+2
			res[:,i] = self.le.transform(res[:,i].tolist())
		#print(res[:3])
		#input()
		return res
	# def test(self,X,Y):
	# 	X = self.process_data(X)
	# 	param_grid = {'n_estimators': [750], 'max_features': [8]}
	# 	model = GridSearchCV(estimator=self.clf, param_grid=param_grid, n_jobs=1, cv=None, verbose=20, scoring='accuracy')
	# 	model.fit(X, Y)
	# 	df = pd.DataFrame(model.cv_results_)
	# 	print(df)
	# 	return
		
	def fit(self,X_train,y_train):
		self.has_fit = False
		self.wifi_set = set()
		X_train = self.process_data(X_train)
		#print(X_train[:3])
		self.clf.fit(X_train,y_train)
		self.has_fit = True
		
	def predict(self,X_test):
		X_test = self.process_data(X_test)
		ret = self.clf.predict(X_test)
		return ret
