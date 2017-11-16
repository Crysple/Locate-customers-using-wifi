
#作为模型的单元测试代码以及总的调用Ensemble的代码
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from wifi_clf import Wifi_clf
from rf_clf import RF_clf
from xgb_clf import XGB_clf

from extra_clf import EXTRA_clf
from bayes_clf import BAYES_clf
from xgb_stacker import XGB_stacker
from Ensemble import Ensemble
#temp
from xgb_clf000 import XGB_clf000
from xgb_clf0001 import XGB_clf0001
from xgb_clf0002 import XGB_clf0002
from multibinary_xgb import MB_XGB_clf
from multibinary_xgb_0 import MB_XGB_0_clf
from multibinary_xgb_1 import MB_XGB_1_clf
from wifi_select_shop_clf import WIFI_SS_clf
from binary_xgb import BINARY_XGB_clf
from multiprocessing import Pool

def switch_col(df,c1,c2):
	temp = df[c1].copy()
	df[c1] = df[c2]
	df[c2] = temp
	return df.rename(index=str, columns={c1: c2, c2: c1})
#是否为测试用
TEST = False

def calcu_acc():
	clf = WIFI_SS_clf()
	shop = pd.read_csv('shop_info.csv')
	mall_list = shop['mall_id'].unique()
	ll = mall_list
	all_len = 0
	all_err = 0
	for ii,mall in enumerate(mall_list):
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		Y = user_shop['shop_id']
		# user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		X = switch_col(user_shop,'mall_id','user_id')
		# X_train,X_test,y_train,y_test = train_test_split(X.values.tolist(),Y.values.tolist(),test_size=0.15, random_state=22)
		# all_err+=clf.calcu_acc(X_train,y_train,X_test,y_test)
		# all_len+=len(y_test)
		X,X_test,Y,y_test = train_test_split(X.values,Y.values,test_size=0, random_state=22)
		
		Y_candidate = pd.read_csv('candidate/'+mall+'.csv').values
		for i in range(Y_candidate.shape[0]):
			if Y[i] != Y_candidate[i][0]:
				all_err+= 1
		all_len+=Y.shape[0]

	print("ALL acc:",1-all_err/all_len)
def unit_test(clf,mod,save_name = "",mp=False,proba_file_name=""):
	#分mall训练
	shop = pd.read_csv('shop_info.csv')
	mall_list = shop['mall_id'].unique()

	total_right,total_len = 0.0,0.0
	gross_row_list = []
	gross_pred_list = []
	ll = ['m_6587','m_7800','m_1293','m_2224','m_7168','m_2123','m_6167','m_622','m_5529','m_2907']
	for ii,mall in enumerate(ll):
		if ii%4!=mod and mp:
			continue
		#读取数据
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		# shop_id,category_id,price,mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
		Y = user_shop['shop_id']
		user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		#mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
		#switch mall_id and user_id
		X = switch_col(user_shop,'mall_id','user_id')
		#user_id,mall_id,time_stamp,longitude,latitude,wifi_infos

		### weizhao test binary
		X['id'] = range(len(X))
		clf.mall = mall

		#分隔数据
		X_train,X_test,y_train,y_test = train_test_split(X.values,Y.values,test_size=0.15, random_state=25)
		
		#print(X_train[:2])
		#input()
		clf.fit(X_train, y_train)
		preds = clf.predict(X_test)
		acc = accuracy_score(preds,y_test)
		right = 0
		for i,j in zip(preds, y_test):
			if i==j:
				right+=1
		total_right+=right
		total_len+=len(y_test)
		print(ii,"mall:",mall,"acc: ",acc)
	print("ALLLL acc:",total_right/total_len)
def MP(clf,func):
	save_name = "mb_xgb_0_minute"
	if func.__name__ != 'unit_test':
		#pan duan you mei you !!!!!!!!!!!!!!!!!!!!!!
		os.mkdir('proba_file/'+save_name)
		#pass
	p = Pool(4)
	for i in range(4):
		p.apply_async(func,args = (clf,i,(str(i)+".csv"),True,save_name))
		#result = p.apply_async(func,args = (clf,i,(str(i)+".csv")))
		#result.get()
	p.close()
	p.join()
	#return
	if func.__name__ == 'unit_test':
		return
	df = pd.read_csv("0.csv")
	os.remove('0.csv')
	for i in range(1,4):
		now_df = pd.read_csv(str(i)+".csv")
		df = pd.concat([df,now_df])
		os.remove(str(i)+".csv")
	df.to_csv("ans/"+save_name+".csv",index=False)
def get_result(clf,mod,save_name = "",get_probs_file=False,proba_file_name=""):
	
	#分mall训练
	shop = pd.read_csv('shop_info.csv')
	evalution = pd.read_csv('evaluation_public.csv')
	mall_list = shop['mall_id'].unique()

	total_right,total_len = 0.0,0.0
	gross_row_list = []
	gross_pred_list = []
	for ii,mall in enumerate(mall_list):
		if ii%4!=mod:
			continue
		#读取数据
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		# shop_id,category_id,price,mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
		Y = user_shop['shop_id']
		user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		#mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
		#switch mall_id and user_id
		X = switch_col(user_shop,'mall_id','user_id')
		#user_id,mall_id,time_stamp,longitude,latitude,wifi_infos
		X,X_test,Y,y_test = train_test_split(X.values,Y.values,test_size=0, random_state=22)
		#print(X_train[:2])
		#input()
		this_mall = evalution[evalution.mall_id==mall]
		row_list = this_mall.row_id.values.tolist()
		##############
		# df = pd.read_csv("proba_file/"+proba_file_name+"/"+mall+".csv")
		# df['row_id'] = row_list
		# df.drop("Unnamed: 0",axis=1,inplace=True)
		# df.to_csv("proba_file/"+proba_file_name+"/"+mall+".csv",index=False)
		# print(mall)
		# continue
		##############
		this_mall.pop('row_id')

		clf.fit(X, Y)
		if not get_probs_file:
			preds = clf.predict(this_mall.values)
		else:
			preds, filedf = clf.predict(this_mall.values,True)
			filedf['row_id'] = row_list
			filedf.to_csv("proba_file/"+proba_file_name+"/"+mall+".csv",index=False)

			tpreds, tfiledf = clf.predict(X,True);
			tfiledf.to_csv("proba_file/"+proba_file_name+"/"+mall+"t.csv",index=False)
		gross_row_list.extend(row_list)
		gross_pred_list.extend(preds)

		print(ii,"mall:",mall)

	result = pd.DataFrame({'row_id':gross_row_list,'shop_id':gross_pred_list})
	result.to_csv(save_name,index=None)


def search(clf):
	#分mall训练

	total_right,total_len = 0.0,0.0
	gross_row_list = []
	gross_pred_list = []
	for mall in ['m_7800','m_690','m_3839','m_3739']:

		#读取数据
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		# shop_id,category_id,price,mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
		Y = user_shop['shop_id']
		user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		#mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
		#switch mall_id and user_id
		X = switch_col(user_shop,'mall_id','user_id')
		#user_id,mall_id,time_stamp,longitude,latitude,wifi_infos

		#分隔数据
		X_train,X_test,y_train,y_test = train_test_split(X.values,Y.values,test_size=0.1, random_state=22)
		
		#print(X_train[:2])
		#input()
		paralist = []
		acclist = []
		for para in [0.02,0.05,0.07,0.1]:
			clf.test(X_train,y_train,para)
			preds = clf.predict(X_test)
			acc = accuracy_score(preds,y_test)
			right = 0
			for i,j in zip(preds, y_test):
				if i==j:
					right+=1
			total_right+=right
			total_len+=len(y_test)
			paralist.append(para)
			acclist.append(acc)
			print("para:",para,"acc: ",acc)
		for i,j in zip(paralist,acclist):
			print("Mall",mall,"para:",i,"acc: ",j)
	print("ALLLL acc:",total_right/total_len)
def test_ensemble(ensemble,has = False):
#分mall训练
	# shop = pd.read_csv('shop_info.csv')
	# mall_list = shop['mall_id'].unique()

	total_right,total_len = 0.0,0.0
	gross_row_list = []
	gross_pred_list = []
	for mall in ['m_4422','m_690']:
		ensemble.mall = mall
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		Y = user_shop['shop_id']
		user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		X = switch_col(user_shop,'mall_id','user_id')
		#user_id,mall_id,time_stamp,longitude,latitude,wifi_infos

		X_train,X_test,y_train,y_test = train_test_split(X.values.tolist(),Y.values.tolist(),test_size=0.1, random_state=22)
		#ensemble.process_data(X_train,X_test)
		#return
		
		if not has:
			S_train = ensemble.fit_predict(X_train,y_train,X_test)
			for i in range(len(ensemble.base_models)):
				acc = accuracy_score(S_train[:,i].tolist(),y_train)
				print("model",i,"\t",acc)
		

		# preds = ensemble.fit_predict_stacker(y_train)
		# acc = accuracy_score(preds,y_test)
		# right = 0
		# for i,j in zip(preds, y_test):
		# 	if i==j:
		# 		right+=1
		# total_right+=right
		# total_len+=len(y_test)
		# print("mall:",mall,"acc: ",acc)


def get_ensemble_result(ensemble,has = False):
	#分mall训练
	shop = pd.read_csv('shop_info.csv')
	evalution = pd.read_csv('evaluation_public.csv')
	mall_list = shop['mall_id'].unique()

	total_right,total_len = 0.0,0.0
	gross_row_list = []
	gross_pred_list = []
	cnt = 0
	for mall in mall_list:
		ensemble.mall = mall
		cnt+=1
		#读取数据
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		Y = user_shop['shop_id']
		user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		X = switch_col(user_shop,'mall_id','user_id')
		X = X.values.tolist()
		Y = Y.values.tolist()
		#user_id,mall_id,time_stamp,longitude,latitude,wifi_infos
		X,X_test,Y,y_test = train_test_split(X,Y,test_size=0, random_state=22)
		#print(X_train[:2])
		#input()

		this_mall = evalution[evalution.mall_id==mall]
		row_list = this_mall.row_id.values.tolist()
		this_mall.pop('row_id')
		this_mall = this_mall.values.tolist()
		has = os.path.isfile("ENSEMBLE/"+str(mall)+"TEST.csv")
		##############THE following is true
		# if not has:
		# 	S_train = ensemble.fit_predict(X,Y,this_mall)
		
		# 	for i in range(len(ensemble.base_models)):
		# 		acc = accuracy_score(S_train[:,i].tolist(),Y)
		# 		print("model",i,"\t",acc)

		##############################
		# S_train = ensemble.fit_predict(X,Y,this_mall)
		
		# for i in range(len(ensemble.base_models)):
		# 	acc = accuracy_score(S_train[:,i].tolist(),Y)
		# 	print("model",i,"\t",acc)
		####################################
		preds = ensemble.fit_predict_stacker(Y)

		gross_row_list.extend(row_list)
		gross_pred_list.extend(preds)

		print(cnt,"mall:",mall)
	result = pd.DataFrame({'row_id':gross_row_list,'shop_id':gross_pred_list})
	result.to_csv('ensemble1.1.csv',index=None)


def train_staker():
	clf = XGB_stacker()
	for mall in ['m_690','m_6587','m_625','m_3839','m_1293']:

		#读取数据
		user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
		Y = user_shop['shop_id']
		user_shop.drop(['shop_id','category_id','price'],axis=1,inplace=True)
		X = switch_col(user_shop,'mall_id','user_id')

		#分隔数据
		X_train,X_test,y_train,y_test = train_test_split(X.values.tolist(),Y.values.tolist(),test_size=0, random_state=22)
		for p in range(4,22,2):
			clf.test(y_train,mall,p)
def main():
	x1 = XGB_clf000("dirty_wifi_final_15.csv")
	x2 = XGB_clf000("dirty_wifi_final_20.csv")
	x3 = XGB_clf0001("dirty_wifi_final_15.csv")
	x4 = XGB_clf0001("dirty_wifi_final_20.csv")
	x5 = XGB_clf0002("dirty_wifi_final_15.csv")
	x6 = XGB_clf0002("dirty_wifi_final_20.csv")
	model_list = [x1,x2,x3,x4,x5,x6]
	ensemble = Ensemble(XGB_stacker(),model_list)
	get_ensemble_result(ensemble)
	#test_ensemble(ensemble)
# for mall in ['m_4422']:
# 	print(mall)
# 	user_shop = pd.read_csv('unclean_mall/'+mall+'.csv')
# 	# shop_id,category_id,price,mall_id,user_id,time_stamp,longitude,latitude,wifi_infos
# 	Y = user_shop['shop_id'].values.tolist()
# 	ans = pd.read_csv("ENSEMBLE/"+mall+"TRAIN.csv").values
# 	for i in range(5):
# 		acc = accuracy_score(ans[:,i],Y)
# 		print("model",i,"\t",acc)
#main()
#train_staker()
# rf_clf = RF_clf()
#unit_test(MB_XGB_COPY_clf())
#wifi = Wifi_clf()

unit_test(BINARY_XGB_clf(),4)


#unit_test(XGB_clf000("dirty_wifi_final_15.csv"))
#unit_test(XGB_clf000("dirty_wifi_final_20.csv"))
#unit_test(XGB_clf0001())
# dt_clf = DT_clf()
# unit_test(dt_clf)
# xgb = XGB_clf()
#search(XGB_clf000())
# bayes_clf = BAYES_clf()
#unit_test(XGB_clf0002("dirty_wifi_final_15.csv"))
#get_result(MB_XGB_COPY_clf())
# MP(MB_XGB_clf(),get_result)
#MP(BINARY_XGB_clf(),unit_test)
#MP(MB_XGB_1_clf(),unit_test)
#unit_test(MB_XGB_clf(),3,"",False)
