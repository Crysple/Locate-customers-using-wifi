import pandas as pd
import os
#Import libraries:
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from wifi_select_shop_clf import WIFI_SS_clf
from sklearn.preprocessing import LabelEncoder
import time
from collections import defaultdict

class BINARY_XGB_clf(object):
	def __init__(self,opt="dirty_wifi15.csv"):
		self.param = {
			'booster': 'gbtree',
			'objective':'binary:logistic',
			'max_depth':14, #!14
			'eta':0.1, 
			'subsample':0.85,#!
			'colsample_bytree':0.8,#!
			#'updater':'grow_gpu',
			#'tree_method':'hist',
			'silent':1, 
			'verbose':0,
			'seed':0
			#'missing':-999
			 }
		self.mall = None
		self.wifi_ss_clf = WIFI_SS_clf()
		self.has_fit = False
		dirty_wifi = pd.read_csv(opt)
		self.dirty_dict = defaultdict(lambda :0)
		for row in dirty_wifi.values:
			self.dirty_dict[row[0]] = 1
		self.wifi_set_dict = dict()

		shop_info = pd.read_csv('shop_info.csv')
		self.le = LabelEncoder()
		self.le.fit(shop_info.shop_id.values)
		self.shop_dict = {}
		self.shop_price_dict = {}
		for idx, row in shop_info.iterrows():
			la,lo = row['latitude'],row['longitude']
			self.shop_dict[row['shop_id']] = (lo,la)
			self.shop_price_dict[row['shop_id']] = row['price']
		self.last_time = None

	def cnt(self,text):
		now_time = time.time()
		if self.last_time is None:
			self.last_time = now_time
			last_time = now_time
		else:
			last_time = self.last_time
			self.last_time = now_time
		print(text, now_time-last_time)


	def init(self):
		self.last_time = None
		self.has_fit = False
		self.wifi_set_dict = defaultdict(lambda :0)
		self.id2candidate = {}
	def process_data(self,x):
		#39 row[0,1],,,40 row[2]
		x = np.delete(x,[0,1],1)
		#'longitude'  latitude  wifi_infos
		res = []
		for row in x:
			new_row = {'hour':int(row[0].split()[1][:2]),'longitude':float(row[1]),'latitude':float(row[2])}
			wifi_list = [wifi.split('|') for wifi in row[3].split(';')]
			for wifi in wifi_list:
				if self.has_fit:
					if self.wifi_set_dict[wifi[0]]==1:
						new_row[wifi[0]] = int(wifi[1])
				elif self.dirty_dict[wifi[0]]==0:
					new_row[wifi[0]] = int(wifi[1])
					self.wifi_set_dict[wifi[0]] = 1
			res.append(new_row)
		if not self.has_fit:
			res = pd.DataFrame(res)
			self.train_df = res[:1]
			self.train_df = self.train_df.reset_index(drop=True)
		else:
			temp = self.train_df.append(res)
			temp.reset_index(drop=True,inplace=True)
			temp.drop(0,axis=0,inplace=True)
			res = temp
		return res
	def getCandiDate(self,Id):
		if not self.has_fit:
			candidate_data = pd.read_csv('candidate/'+self.mall+'.csv').values
			self.id2candidate = {}
			for line in candidate_data:
				self.id2candidate[line[-1]] = line[:-1]
		res = []
		for line in Id:
			res.append(self.id2candidate[line])
		return res

	def process_candidate(self,X,shop_candidate,y_train=None):

		y_label = []
		X_new = []
		X = X.values
		num = X.shape[0]
		for i in range(num):
			x = X[i]
			if y_train is not None:
				y = y_train[i]
			for shop in shop_candidate[i]:
				my_x = np.append(x,[shop,self.shop_price_dict[self.le.inverse_transform(shop)]])
				my_x = np.append(my_x,self.shop_dict[self.le.inverse_transform(shop)])
				X_new.append( my_x ) 
				if y_train is not None:
					y_label.append( int(shop==y) )
		if y_train is not None:
			return X_new,y_label			
		return X_new	

	def fit(self,X_train,y_train,alldata=False):
		self.init()
		# 得到候选集
		self.cnt('ALL start')
		shop_candidate = self.getCandiDate(X_train[:,-1])
		self.cnt('candidate finish')

		X_train = self.process_data(X_train)
		y_train = self.le.transform(y_train)
		shop_candidate = [self.le.transform(shop_list) for shop_list in shop_candidate]
		self.cnt('process finish')

		X_new_train,y_label = self.process_candidate(X_train,shop_candidate,y_train)
		y_np = np.array(y_label)
		self.cnt('process candidate finish')


		self.param['scale_pos_weight'] = (y_np.shape[0]-np.sum(y_np)) / np.sum(y_np)
		### 需要加scale_pos_weight吗？
		
		num_round = 100
		#print(X_new_train[:1])

		dtrain = xgb.DMatrix(X_new_train,label=y_label)

		self.bst =xgb.train(self.param,dtrain,num_round) 


		self.has_fit = True


		# self.bst.save_model(FileName)
	def predict(self,X_test):
		shop_candidate = self.getCandiDate(X_test[:,-1])
		num = X_test.shape[0]
		shop_candidate = [self.le.transform(shop_list) for shop_list in shop_candidate]
		res = []
		X_test = self.process_data(X_test)

		X_new_test = self.process_candidate(X_test,shop_candidate)
		dtest = xgb.DMatrix(X_new_test)	
		pred = self.bst.predict(dtest)
		now = 0
		for i in range(num):
			try:
				shop_candidate_i = shop_candidate[i]
				pred_i = pred[now:now+len(shop_candidate_i)]
				max_index = np.where(pred_i==np.max(pred_i))
				ans_shop = self.le.inverse_transform(shop_candidate_i[max_index])
				res.append(ans_shop[0])
				now += len(shop_candidate_i)
			except:
				print(pred_i)
				print(now,len(shop_candidate_i))
				print(len(pred))
				input()
		return res



	# def test(self,X,Y):
	# 	X_dtrain, X_deval, y_dtrain, y_deval = train_test_split(X,Y,test_size=0.15, random_state=22)
	# 	dtrain = xgb.DMatrix(X_dtrain, y_dtrain)
	# 	deval = xgb.DMatrix(X_deval, y_deval)
	# 	watchlist = [(deval, 'eval')]
	# 	params = {
	# 		'booster': 'gbtree',
	# 		'objective': 'reg:linear',
	# 		'subsample': 0.8,
	# 		'colsample_bytree': 0.85,
	# 		'eta': 0.1,
	# 		'max_depth': 7,
	# 		'seed': 2016,
	# 		'silent': 0,
	# 		'eval_metric': 'rmse'
	# 	}
	# 	clf = xgb.train(params, dtrain, 500, watchlist, early_stopping_rounds=50)
	# 	pred = clf.predict(xgb.DMatrix(df_test))
