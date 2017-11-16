#原版
import pandas as pd
import os
#Import libraries:
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn.model_selection import train_test_split
from computeLLD import haversine
from sklearn import preprocessing
from matplotlib import pyplot
from xgboost import plot_importance
class MB_XGB_0_clf(object):
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
			'verbose':0
			#'missing':-999
			 }
		self.has_fit = False
		dirty_wifi = pd.read_csv(opt)
		self.dirty_dict = dict()
		for row in dirty_wifi.values:
			self.dirty_dict[row[0]] = 1
		self.wifi_set_dict = dict()

		shop_info = pd.read_csv('shop_info.csv')
		self.shop_dict = {}
		for idx, row in shop_info.iterrows():
			la,lo = row['latitude'],row['longitude']
			price = row['price']
			self.shop_dict[row['shop_id']] = (lo,la,price)
	def init(self):
		self.has_fit = False
		self.wifi_set_dict = dict()
	def process_one(self,X,shop):
		#x = X.copy()
		#################################################################################
		# X['olongitude'] = self.shop_dict[shop][0]
		# X['olatitude'] = self.shop_dict[shop][1]
		# X['price'] = self.shop_dict[shop][2]
		#################################################################################
		#x['dis'] = x.apply(lambda r:float(haversine(r['longitude'],r['latitude'],self.shop_dict[shop][0],self.shop_dict[shop][1])),axis=1)
		#x['dlong'] = x.apply(lambda r:float(abs(r['longitude']-self.shop_dict[shop][0]))*10000,axis=1)
		#x['dlati'] = x.apply(lambda r:float(abs(r['latitude']-self.shop_dict[shop][1]))*10000,axis=1)
		#print(x[:3])
		#input()
		#x.pop('longitude')
		#x.pop('latitude')
		return X
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
					if wifi[0] in self.wifi_set_dict:
						new_row[wifi[0]] = int(wifi[1])
				elif wifi[0] not in self.dirty_dict:
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
	def fit(self,X_train,y_train,alldata=False):
		self.init()
		self.binaryClfList = {}
		X_train = self.process_data(X_train)
		y_total_num = y_train.shape[0]
		self.y_unique_train = np.unique(y_train)

		for y in self.y_unique_train:
			num_round = 120 if y=="m_7800" else 150
			#num_round = 1
			X_train_i = self.process_one(X_train,y)
			y_binary_train = (y_train==y).astype(int)#0,1
			y_pos_num = np.sum(y_binary_train)
			self.param['scale_pos_weight'] = (y_total_num-y_pos_num) / y_pos_num
			dtrain = xgb.DMatrix(X_train_i,label = y_binary_train)
			#num_round = 105
			bst = xgb.train(self.param, dtrain, num_round)
			self.binaryClfList[y]=bst
			# plot_importance(bst)
			# pyplot.show()
		self.has_fit = True
		

		# self.bst.save_model(FileName)
	def select_shop(self,lo,la):
		res = []
		for y in self.y_unique_train:
			if haversine(lo,la,self.shop_dict[y][0],self.shop_dict[y][1])<200:
				res.append(y)
		return res
	def predict(self,X_test,return_predict = False):
		res=[]
		Sy = np.zeros((X_test.shape[0],self.y_unique_train.shape[0]),dtype="float")
		X_test = self.process_data(X_test)
		for i,y in enumerate(self.y_unique_train):
			X_test_i = self.process_one(X_test,y)
			dtest = xgb.DMatrix(X_test_i)
			#preds = bst.predict(dtrain,num_round)
			Sy[:,i] = self.binaryClfList[y].predict(dtest)
		res = []
		for i in range(X_test.shape[0]):
			row = Sy[i,:]
			max_index = np.where(row==np.max(row))
			res.append(self.y_unique_train[max_index][0])
		##################################
		if return_predict:
			df = pd.DataFrame(Sy)
			df.columns = self.y_unique_train
			return res,df
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
