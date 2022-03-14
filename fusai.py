import pandas as pd
pd.options.display.max_columns=None
pd.options.display.max_rows=1000
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.cluster import DBSCAN
import lightgbm as lgb
import shap
import math
from stdbscan import STDBSCAN
import itertools
from scipy import stats
from scipy.spatial.distance import cdist, euclidean
from datetime import datetime
import shapely.geometry as geom
from multiprocessing import Pool,cpu_count
from datetime import timedelta
from gensim.models import Word2Vec,Doc2Vec
import xgboost as xgb

# test_path='./hy_round1_testA_20200102/'
# test_b_path='./hy_round1_testB_20200221/'
train_path='./hy_round2_train_20200225/'
type_map={'拖网':0,'刺网':1,'围网':2}
type_map_rev={0:'拖网',1:'刺网',2:'围网'}
splits=10
base=[25.173215180565872, 119.27677013681826]

def cut_data(data,num,col):
	data_part_list=[]
	col_list=list(set(data[col]))
	if col=='ship':
		col_list.sort()
	size=int(math.ceil(len(col_list)/num))
	for i in range(num):
		start=i*size
		end=(i+1)*size if (i+1)*size<len(col_list) else len(col_list)
		col_part=col_list[start:end]
		data_part=data[data[col].isin(col_part)]
		data_part_list.append(data_part)
	return data_part_list

def multi_func(num,data,col,func):
	data_part_list=cut_data(data,num,col)
	p=Pool(num)
	rlist=p.map(func,data_part_list)
	p.close()
	p.join()
	if func==prepare_data_3:
		for i in range(num-1):
			rlist[i+1]['group']=rlist[i+1]['group']+rlist[i]['group'].max()
	data=pd.concat(rlist,ignore_index=True)
	return data
		
def prepare_data(path,data_name):
	if os.path.exists('./data/{}.pkl'.format(data_name)):
		data=pd.read_pickle('./data/{}.pkl'.format(data_name))
	else:
		filenames=os.listdir(path)
		data=pd.DataFrame()
		for filename in filenames:
			print(filename)
			df=pd.read_csv(path+filename)
			data=pd.concat([df,data],axis=0)
		data=data.rename(columns={'渔船ID':'ship','速度':'speed','方向':'direction'})
		data.to_pickle('./data/{}.pkl'.format(data_name))
		print(data.shape)
		print(data.head())
	return data

def get_r(lat):
	'''
	https://planetcalc.com/7721/
	'''
	a=6378137
	b=6356752
	lat=np.radians(lat)
	up=(a**2*np.cos(lat))**2+(b**2*np.sin(lat))**2
	down=(a*np.cos(lat))**2+(b*np.sin(lat))**2
	r=np.sqrt(up/down)
	return r

def get_haversine_dis(lat1,lon1,lat2,lon2):
	r=get_r(lat2)
	# for item in [lat1,lon1,lat2,lon2]:
		# item=np.radians(item)
	lat1,lon1,lat2,lon2=map(np.radians,[lat1,lon1,lat2,lon2])
	lat_gap=lat2-lat1
	lon_gap=lon2-lon1
	a=np.sin(lat_gap/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(lon_gap/2)**2
	c=2*np.arcsin(np.sqrt(a))
	dis=r*c
	return dis
	
def get_xy(lat1,lon1,lat2,lon2):
	x=get_haversine_dis(lat1,lon1,lat1,lon2)
	y=get_haversine_dis(lat1,lon1,lat2,lon1)
	if lon2<lon1:
		x=-1*x
	if lat2<lat1:
		y=-1*y
	return [x,y]
	
def get_label(data,name):
	if os.path.exists('./data/label_{}.pkl'.format(name)):
		label=pd.read_pickle('./data/label_{}.pkl'.format(name))
	else:
		temp=data.copy()
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
		temp.sort_values(['ship','group','time'],inplace=True)
		label=temp[['ship','group','type','stop']].drop_duplicates()
		label.to_pickle('./data/label_{}.pkl'.format(name))
	print('label done_{}'.format(name))
	return label

def diff_abs_mean(x):
	return np.mean(np.abs(np.diff(x)))

def cid_ce(x):#时序数据复杂度
	x=x.values
	x=(x-np.mean(x))/np.std(x)
	x=np.diff(x)
	return np.sqrt(np.dot(x,x))
	
def longest_above_median(x):
	x=x.values
	x=(x>np.median(x))
	x=x.astype(int)
	res=[len(list(group)) for value,group in itertools.groupby(x) if value==1]	
	if len(res)>0:
		return np.max(res)
	else:
		return 0

def angle(x_gap_1,x_gap_2,y_gap_1,y_gap_2):
	angle1=math.atan2(y_gap_1,x_gap_1)*180/math.pi
	angle2=math.atan2(y_gap_2,x_gap_2)*180/math.pi
	if angle1*angle2>=0:
		angle=abs(angle1-angle2)
	else:
		angle=abs(angle1)+abs(angle2)
		if angle>180:
			angle=360-angle
	return angle

def corr(x,y):
	return stats.pearsonr(x,y)[0]
		
def p(x,y):
	return stats.pearsonr(x,y)[1]

def min_dis_range(x):
	x=x[['x','y']].values
	nn=NearestNeighbors(n_neighbors=2,algorithm='ball_tree').fit(x)
	distances,indices=nn.kneighbors(x)
	return distances[:,1]

def longest_below_median(x):#中位值以下最大连续数
	x=x.values
	x=(x<np.median(x))
	x=x.astype(int)
	res=[len(list(group)) for value,group in itertools.groupby(x) if value==1]
	if len(res)>0:
		return np.max(res)
	else:
		return 0
			
def peak_nums(x):
	x=x.values
	x=(x>np.median(x))
	x=x.astype(int)
	res=[len(list(group)) for value,group in itertools.groupby(x) if value==1]	
	if len(res)>0:
		return len(res)
	else:
		return 0
		
def geometric_median(x,eps=1):
	x=x[['x','y']].values
	y=np.mean(x,0)
	while True:
		D=cdist(x,[y])
		nonzeros=(D!=0)[:,0]
		Dinv=1/D[nonzeros]
		Dinvs=np.sum(Dinv)
		W=Dinv/Dinvs
		T=np.sum(W*x[nonzeros],0)
		num_zeros=len(x)-np.sum(nonzeros)
		if num_zeros==0:
			y1=T
		elif num_zeros==len(x):
			return y
		else:
			R=(T-y)*Dinvs
			r=np.linalg.norm(R)
			rinv=0 if r == 0 else num_zeros/r
			y1=max(0,1-rinv)*T+min(1,rinv)*y
		if euclidean(y,y1)<eps:
			return y1
		y=y1

def quantile_min(x):
	return x.quantile(0.02)
	
def quantile_max(x):
	return x.quantile(0.98)
	
def quantile25(x):
	return x.quantile(0.25)
	
def quantile75(x):
	return x.quantile(0.75)
							
def get_stat_features(data):
	temp=data.copy()
	temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
	temp.sort_values(['ship','group','time'],inplace=True)
	temp['r']=(temp['x']**2+temp['y']**2)**0.5
	temp['the']=temp.apply(lambda x:math.atan2(x.y,x.x)*180/math.pi,axis=1)
	features=pd.DataFrame()
	for col in ['x','y','speed','direction']:		
		feature=temp.groupby('group',as_index=False)[col].agg({
			'{}_min'.format(col):quantile_min,
			'{}_max'.format(col):quantile_max,
			'{}_mean'.format(col):'mean',
			'{}_median'.format(col):'median',
			'{}_std'.format(col):'std',
			'{}_skew'.format(col):'skew',
			'{}_sum'.format(col):'sum',
			'{}_diff_abs_mean'.format(col):diff_abs_mean,
			'{}_mode'.format(col):lambda x:x.value_counts().index[0],
			'{}_quantile25'.format(col):quantile25,#stat+
			'{}_quantile75'.format(col):quantile75,#stat+
			'count':'count'})
		if features.shape[0]==0:
			features=feature
		else:
			feature=feature.drop('count',axis=1)
			features=features.merge(feature,on='group',how='left')
	for col in ['speed','direction']:
		feature=temp.groupby('group',as_index=False)[col].agg({
			'{}_cid_ce'.format(col):cid_ce,
			'{}_longest_above_median'.format(col):longest_above_median,})
		features=features.merge(feature,on='group',how='left')

	first_feature=temp.groupby('group').first()
	first_feature=first_feature[['direction','speed','x','y']]
	first_feature.columns=[c+'_first' for c in first_feature.columns]
	first_feature=first_feature.reset_index()
	features=features.merge(first_feature,on='group',how='left')
	last_feature=temp.groupby('group').last()
	last_feature=last_feature[['direction','speed','x','y']]
	last_feature.columns=[c+'_last' for c in last_feature.columns]	
	last_feature=last_feature.reset_index()
	features=features.merge(last_feature,on='group',how='left')
	
	features['x_gap_max_min']=features['x_max']-features['x_min']
	features['y_gap_max_min']=features['y_max']-features['y_min']
	features['xy_gap_max_min']=features['x_max']-features['y_min']
	features['yx_gap_max_min']=features['y_max']-features['x_min']
	features['angle_max_min']=features['y_gap_max_min']/(features['x_gap_max_min']+0.00001)
	features['area_max_min']=features['x_gap_max_min']*features['y_gap_max_min']
	features['dis_max_min']=(features['x_gap_max_min']**2+features['y_gap_max_min']**2)**0.5

	features['x_gap_last_first']=features['x_last']-features['x_first']
	features['y_gap_last_first']=features['y_last']-features['y_first']
	features['xy_gap_last_first']=features['x_last']-features['y_first']
	features['yx_gap_last_first']=features['y_last']-features['x_first']
	features['angle_last_first']=features['y_gap_last_first']/(features['x_gap_last_first']+0.00001)
	features['area_last_first']=abs(features['x_gap_last_first']*features['y_gap_last_first'])
	features['dis_last_first']=(features['x_gap_last_first']**2+features['y_gap_last_first']**2)**0.5
	
	temp['x_gap']=temp.groupby('group')['x'].diff()#stat+
	temp['x_gap_shift']=temp.groupby('group')['x_gap'].shift(-1)
	temp['y_gap']=temp.groupby('group')['y'].diff()
	temp['y_gap_shift']=temp.groupby('group')['y_gap'].shift(-1)
	temp['angle']=temp.apply(lambda x:angle(x.x_gap,x.x_gap_shift,x.y_gap,x.y_gap_shift),axis=1)
	temp['time_gap']=temp.groupby('group')['time'].diff()
	temp['time_gap']=temp.apply(lambda x:x.time_gap.total_seconds()/3600,axis=1)
	temp['dis']=(temp.x_gap**2+temp.y_gap**2)**0.5
	temp['area']=abs(temp.x_gap*temp.y_gap)		
	feature_temp=temp.groupby('group',as_index=False).agg({
		'time_gap':'sum',
		'dis':'sum',
		'area':'sum',
		'angle':'sum'})
	features=features.merge(feature_temp,on='group',how='left')
	features['dis_max_min_ration']=features['dis']/(features['dis_max_min']+0.00001)
	features['dis_last_first_ration']=features['dis']/(features['dis_last_first']+0.00001)
	features['area_max_min_ration']=features['area']/(features['area_max_min']+0.00001)
	features['area_last_first_ration']=features['area']/(features['area_last_first']+0.00001)	
	features['area_max_min_dis_ration']=features['area_max_min']/(features['dis_max_min']+0.00001)
	features['area_last_first_dis_ration']=features['area_last_first']/(features['dis_last_first']+0.00001)
	features['area_dis_ration']=features['area']/(features['dis']+0.00001)
	features['area_max_min_time_ration']=features['area_max_min']/(features['time_gap']+0.00001)
	features['area_last_first_time_ration']=features['area_last_first']/(features['time_gap']+0.00001)
	features['area_time_ration']=features['area']/(features['time_gap']+0.00001)
	features['dis_max_min_time_ration']=features['dis_max_min']/(features['time_gap']+0.00001)
	features['dis_last_first_time_ration']=features['dis_last_first']/(features['time_gap']+0.00001)
	features['dis_time_ration']=features['dis']/(features['time_gap']+0.00001)		
	features['angle_ration_dis']=features['angle']/(features['dis']+0.00001)
	features['angle_ration_time']=features['angle']/(features['time_gap']+0.00001)
	
	dis_range=data.groupby('group').apply(min_dis_range)#stat+
	dis_range=dis_range.reset_index().rename(columns={0:'dis_range'})
	dis_range['dis_q25']= dis_range.apply(lambda x:np.percentile(x.dis_range,25),axis=1)
	dis_range['dis_q50']= dis_range.apply(lambda x:np.percentile(x.dis_range,50),axis=1)
	dis_range['dis_q75']= dis_range.apply(lambda x:np.percentile(x.dis_range,75),axis=1)		
	dis_range=dis_range[['group','dis_q25','dis_q50','dis_q75']]
	features=features.merge(dis_range,on='group',how='left')
	
	points=temp.groupby('group').apply(geometric_median)#stat+
	points=points.reset_index().rename(columns={0:'points'})
	points['x_geometric_median']=points.points.apply(lambda x:x[0])
	points['y_geometric_median']=points.points.apply(lambda x:x[1])
	points=points[['group','x_geometric_median','y_geometric_median']]
	features=features.merge(points,on='group',how='left')
	temp=temp.merge(points,on='group',how='left')
	temp['dis_geometric_median']=((temp.x-temp.x_geometric_median)**2+(temp.y-temp.y_geometric_median)**2)**0.5
	feature=temp.groupby('group',as_index=False)['dis_geometric_median'].agg({
		'dis_geometric_median_min':quantile_min,
		'dis_geometric_median_max':quantile_max,
		'dis_geometric_median_mean':'mean',
		'dis_geometric_median_median':'median',
		'dis_geometric_median_std':'std',
		'dis_geometric_median_quantile25':quantile25,
		'dis_geometric_median_quantile75':quantile75,
		})
	features=features.merge(feature,on='group',how='left')
	return features	

def get_stat_features_2(data):
	temp=data.copy()
	temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
	temp.sort_values(['ship','group','time'],inplace=True)
	temp['r']=(temp['x']**2+temp['y']**2)**0.5
	temp['the']=temp.apply(lambda x:math.atan2(x.y,x.x)*180/math.pi,axis=1)
	features=pd.DataFrame()
	for col in ['x','y','speed','direction','r','the']:		
		feature=temp.groupby('group',as_index=False)[col].agg({
			'{}_min'.format(col):quantile_min,
			'{}_max'.format(col):quantile_max,
			'{}_mean'.format(col):'mean',
			'{}_median'.format(col):'median',
			'{}_std'.format(col):'std',
			'{}_skew'.format(col):'skew',
			'{}_sum'.format(col):'sum',
			'{}_diff_abs_mean'.format(col):diff_abs_mean,
			'{}_mode'.format(col):lambda x:x.value_counts().index[0],
			'{}_quantile25'.format(col):quantile25,#stat+
			'{}_quantile75'.format(col):quantile75,#stat+
			'count':'count'})
		if features.shape[0]==0:
			features=feature
		else:
			feature=feature.drop('count',axis=1)
			features=features.merge(feature,on='group',how='left')
	for col in ['speed','direction']:
		feature=temp.groupby('group',as_index=False)[col].agg({
			'{}_cid_ce'.format(col):cid_ce,
			'{}_longest_above_median'.format(col):longest_above_median,})
		features=features.merge(feature,on='group',how='left')

	first_feature=temp.groupby('group').first()
	first_feature=first_feature[['direction','speed','x','y']]
	first_feature.columns=[c+'_first' for c in first_feature.columns]
	first_feature=first_feature.reset_index()
	features=features.merge(first_feature,on='group',how='left')
	last_feature=temp.groupby('group').last()
	last_feature=last_feature[['direction','speed','x','y']]
	last_feature.columns=[c+'_last' for c in last_feature.columns]	
	last_feature=last_feature.reset_index()
	features=features.merge(last_feature,on='group',how='left')
	
	features['x_gap_max_min']=features['x_max']-features['x_min']
	features['y_gap_max_min']=features['y_max']-features['y_min']
	features['xy_gap_max_min']=features['x_max']-features['y_min']
	features['yx_gap_max_min']=features['y_max']-features['x_min']
	features['angle_max_min']=features['y_gap_max_min']/(features['x_gap_max_min']+0.00001)
	features['area_max_min']=features['x_gap_max_min']*features['y_gap_max_min']
	features['dis_max_min']=(features['x_gap_max_min']**2+features['y_gap_max_min']**2)**0.5

	features['x_gap_last_first']=features['x_last']-features['x_first']
	features['y_gap_last_first']=features['y_last']-features['y_first']
	features['xy_gap_last_first']=features['x_last']-features['y_first']
	features['yx_gap_last_first']=features['y_last']-features['x_first']
	features['angle_last_first']=features['y_gap_last_first']/(features['x_gap_last_first']+0.00001)
	features['area_last_first']=abs(features['x_gap_last_first']*features['y_gap_last_first'])
	features['dis_last_first']=(features['x_gap_last_first']**2+features['y_gap_last_first']**2)**0.5
	
	temp['x_gap']=temp.groupby('group')['x'].diff()#stat+
	temp['x_gap_shift']=temp.groupby('group')['x_gap'].shift(-1)
	temp['y_gap']=temp.groupby('group')['y'].diff()
	temp['y_gap_shift']=temp.groupby('group')['y_gap'].shift(-1)
	temp['angle']=temp.apply(lambda x:angle(x.x_gap,x.x_gap_shift,x.y_gap,x.y_gap_shift),axis=1)
	temp['time_gap']=temp.groupby('group')['time'].diff()
	temp['time_gap']=temp.apply(lambda x:x.time_gap.total_seconds()/3600,axis=1)
	temp['dis']=(temp.x_gap**2+temp.y_gap**2)**0.5
	temp['area']=abs(temp.x_gap*temp.y_gap)		
	feature_temp=temp.groupby('group',as_index=False).agg({
		'time_gap':'sum',
		'dis':'sum',
		'area':'sum',
		'angle':'sum'})
	features=features.merge(feature_temp,on='group',how='left')
	features['dis_max_min_ration']=features['dis']/(features['dis_max_min']+0.00001)
	features['dis_last_first_ration']=features['dis']/(features['dis_last_first']+0.00001)
	features['area_max_min_ration']=features['area']/(features['area_max_min']+0.00001)
	features['area_last_first_ration']=features['area']/(features['area_last_first']+0.00001)	
	features['area_max_min_dis_ration']=features['area_max_min']/(features['dis_max_min']+0.00001)
	features['area_last_first_dis_ration']=features['area_last_first']/(features['dis_last_first']+0.00001)
	features['area_dis_ration']=features['area']/(features['dis']+0.00001)
	features['area_max_min_time_ration']=features['area_max_min']/(features['time_gap']+0.00001)
	features['area_last_first_time_ration']=features['area_last_first']/(features['time_gap']+0.00001)
	features['area_time_ration']=features['area']/(features['time_gap']+0.00001)
	features['dis_max_min_time_ration']=features['dis_max_min']/(features['time_gap']+0.00001)
	features['dis_last_first_time_ration']=features['dis_last_first']/(features['time_gap']+0.00001)
	features['dis_time_ration']=features['dis']/(features['time_gap']+0.00001)		
	features['angle_ration_dis']=features['angle']/(features['dis']+0.00001)
	features['angle_ration_time']=features['angle']/(features['time_gap']+0.00001)
	
	dis_range=data.groupby('group').apply(min_dis_range)#stat+
	dis_range=dis_range.reset_index().rename(columns={0:'dis_range'})
	dis_range['dis_q25']= dis_range.apply(lambda x:np.percentile(x.dis_range,25),axis=1)
	dis_range['dis_q50']= dis_range.apply(lambda x:np.percentile(x.dis_range,50),axis=1)
	dis_range['dis_q75']= dis_range.apply(lambda x:np.percentile(x.dis_range,75),axis=1)		
	dis_range=dis_range[['group','dis_q25','dis_q50','dis_q75']]
	features=features.merge(dis_range,on='group',how='left')
	
	points=temp.groupby('group').apply(geometric_median)#stat+
	points=points.reset_index().rename(columns={0:'points'})
	points['x_geometric_median']=points.points.apply(lambda x:x[0])
	points['y_geometric_median']=points.points.apply(lambda x:x[1])
	points=points[['group','x_geometric_median','y_geometric_median']]
	features=features.merge(points,on='group',how='left')
	temp=temp.merge(points,on='group',how='left')
	temp['dis_geometric_median']=((temp.x-temp.x_geometric_median)**2+(temp.y-temp.y_geometric_median)**2)**0.5
	feature=temp.groupby('group',as_index=False)['dis_geometric_median'].agg({
		'dis_geometric_median_min':quantile_min,
		'dis_geometric_median_max':quantile_max,
		'dis_geometric_median_mean':'mean',
		'dis_geometric_median_median':'median',
		'dis_geometric_median_std':'std',
		'dis_geometric_median_quantile25':quantile25,
		'dis_geometric_median_quantile75':quantile75,
		})
	features=features.merge(feature,on='group',how='left')
	return features	

#stat_features=get_stat_features_multi(num,data,'group',get_stat_features,'stop')
def get_stat_features_multi(num,data,col,func,name):
	if os.path.exists('./data/stat_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/stat_{}.pkl'.format(name))
	else:
		features=multi_func(num,data,col,func)
		features.to_pickle('./data/stat_{}.pkl'.format(name))
	print('stat_feature done_{}'.format(name))
	return features
		
def get_corr_features(data,name):
	if os.path.exists('./data/corr_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/corr_{}.pkl'.format(name))
	else:
		temp=data.copy()
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
		temp.sort_values(['ship','group','time'],inplace=True)
		features=pd.DataFrame()
		feature=temp.groupby('group').apply(lambda x:corr(x.x,x.y)).reset_index().rename(columns={0:'x_y_corr'})
		if features.shape[0]==0:
			features=feature
		else:
			features=features.merge(feature,on='group',how='left')
		feature=temp.groupby('group').apply(lambda x:p(x.x,x.y)).reset_index().rename(columns={0:'x_y_p'})
		features=features.merge(feature,on='group',how='left')
		feature=temp.groupby('group').apply(lambda x:corr(x.speed,x.direction)).reset_index().rename(columns={0:'speed_direction_corr'})
		features=features.merge(feature,on='group',how='left')
		feature=temp.groupby('group').apply(lambda x:p(x.speed,x.direction)).reset_index().rename(columns={0:'speed_direction_p'})
		features=features.merge(feature,on='group',how='left')
				
		features.to_pickle('./data/corr_{}.pkl'.format(name))
	print('corr feature done_{}'.format(name))
	return features

def get_count_features(data,name):
	if os.path.exists('./data/count_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/count_{}.pkl'.format(name))
	else:
		temp=data.copy()
		features=pd.DataFrame()
		for col in ['speed','direction']:
			le=LabelEncoder()
			temp['{}_cut'.format(col)]=le.fit_transform(pd.cut(temp[col],10))
			feature=temp.groupby(['group','{}_cut'.format(col)])['time'].count()
			feature=feature.unstack().fillna(0)
			feature.columns=['{}_cut_{}_count'.format(col,c) for c in feature.columns]
			feature=feature.reset_index()
			if features.shape[0]==0:
				features=feature
			else:
				features=features.merge(feature,on='group',how='left')
		
		features=features.fillna(0)					
		features.to_pickle('./data/count_{}.pkl'.format(name))
	print('count feature done_{}'.format(name))
	return features
	
def get_time_features(data,name):
	if os.path.exists('./data/time_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/time_{}.pkl'.format(name))
	else:
		temp=data.copy()
		features=pd.DataFrame()
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
		temp.sort_values(['ship','group','time'],inplace=True)
		temp['time_gap']=temp.groupby('group')['time'].diff()
		temp['time_gap']=temp.apply(lambda x:x.time_gap.total_seconds()/3600,axis=1)
		temp['hour']=temp.time.dt.hour
		temp['hour_day']=temp['hour'].apply(lambda x:'day' if (x>=6)*(x<18) else 'night')
		feature=temp.groupby('group',as_index=False)['hour'].agg({
			'hour_first':'first',
			'hour_last':'last',
			'hour_min':'min',
			'hour_max':'max',
			'hour_mean':'mean',
			'hour_median':'median',
			'hour_quantile25':quantile25,
			'hour_quantile25':quantile75,
			'hour_std':'std'})			
		if features.shape[0]==0:
			features=feature
		else:
			features=features.merge(feature,on='group',how='left')
		
		features.to_pickle('./data/time_{}.pkl'.format(name))
	print('time feature done_{}'.format(name))
	return features

def get_coastline_features(data):
	temp=data.copy()
	features=pd.DataFrame()
	
	coastline=pd.read_csv('./data/coastline.csv')
	coastline['xy']=coastline.apply(lambda x:get_xy(base[0],base[1],x.lat,x.lon),axis=1)
	coastline['x']=coastline['xy'].apply(lambda x:x[0])
	coastline['y']=coastline['xy'].apply(lambda x:x[1])
	coastline_line=geom.LineString(coastline[['x','y']].values)
	coastline_Polygon=geom.Polygon(coastline[['x','y']].values)

	temp['point']=temp.apply(lambda x:geom.Point(x.x,x.y),axis=1)
	temp['coastline_dis']=temp.apply(lambda x:x.point.distance(coastline_line),axis=1)
	temp['is_in']=temp.apply(lambda x:coastline_Polygon.contains(x.point),axis=1)
	temp['is_in']=temp['is_in'].apply(lambda x:-1 if x else 1)
	temp['coastline_dis']=temp['coastline_dis']*temp['is_in']
	feature=temp.groupby('group',as_index=False)['coastline_dis'].agg({
		'coastline_dis_first':'first',
		'coastline_dis_last':'last',
		'coastline_dis_min':quantile_min,
		'coastline_dis_max':quantile_max,
		'coastline_dis_mean':'mean',
		'coastline_dis_median':'median',
		'coastline_dis_quantile25':quantile25,
		'coastline_dis_quantile25':quantile75,
		'coastline_dis_std':'std'})			
	if features.shape[0]==0:
		features=feature
	else:
		features=features.merge(feature,on='group',how='left')			
	return features	

def get_coastline_features_multi(num,data,col,func,name):
	if os.path.exists('./data/coastline_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/coastline_{}.pkl'.format(name))
	else:
		features=multi_func(num,data,col,func)
		features.to_pickle('./data/coastline_{}.pkl'.format(name))
	print('coastline feature done_{}'.format(name))
	return features	

def get_port():
	data=pd.read_pickle('./data/data_3.pkl')
	temp=data.copy()
	
	coastline=pd.read_csv('./data/coastline.csv')
	coastline['xy']=coastline.apply(lambda x:get_xy(base[0],base[1],x.lat,x.lon),axis=1)
	coastline['x']=coastline['xy'].apply(lambda x:x[0])
	coastline['y']=coastline['xy'].apply(lambda x:x[1])
	coastline_line=geom.LineString(coastline[['x','y']].values)
	coastline_Polygon=geom.Polygon(coastline[['x','y']].values)
	
	temp_train=temp[~temp.type.isnull()]
	temp_train=temp_train[temp_train.stop==1]
	temp_train=temp_train.groupby('group',as_index=False).agg({'x':'mean','y':'mean'})
	temp_train['point']=temp_train.apply(lambda x:geom.Point(x.x,x.y),axis=1)
	temp_train['coastline_dis']=temp_train.apply(lambda x:x.point.distance(coastline_line),axis=1)
	temp_train['is_in']=temp_train.apply(lambda x:coastline_Polygon.contains(x.point),axis=1)
	temp_train['is_in']=temp_train['is_in'].apply(lambda x:-1 if x else 1)
	temp_train['coastline_dis']=temp_train['coastline_dis']*temp_train['is_in']
	temp_train=temp_train[temp_train.coastline_dis<=15000]
	dbscan=DBSCAN(eps=10000,min_samples=8)
	temp_train['port']=dbscan.fit_predict(temp_train[['x','y']].values)
	port=temp_train.groupby('port',as_index=False).agg({'x':'mean','y':'mean'})
	port=port[port.port!=-1].rename(columns={'x':'x_port','y':'y_port'})
	ports=list(port.port)
	coastline_port={}
	for p in ports:
		coastline_port[p]=geom.Point(port[port['port']==p][['x_port']].values[0],port[port['port']==p][['y_port']].values[0])
	def min_port_dis(point):
		coastline_port_dis={}	
		for (k,v) in coastline_port.items():
			coastline_port_dis[k]=point.distance(v)
		return (
			min(zip(coastline_port_dis.values(),coastline_port_dis.keys()))[0],
			min(zip(coastline_port_dis.values(),coastline_port_dis.keys()))[1])	
	
	temp.sort_values(['ship','time'],inplace=True)
	temp_first=temp.groupby('ship',as_index=False).first()
	temp_first['point']=temp_first.apply(lambda x:geom.Point(x.x,x.y),axis=1)
	temp_first['prot_dis']=temp_first.apply(lambda x:min_port_dis(x.point),axis=1)
	temp_first['port']=temp_first['prot_dis'].apply(lambda x:x[1])
	temp_first=temp_first[['ship','port']]
	temp_first=temp_first.merge(port,on='port',how='left')
	return temp_first

def get_port_features(data,name):
	if os.path.exists('./data/port_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/port_{}.pkl'.format(name))
	else:
		temp=data.copy()
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')	
		temp.sort_values(['ship','group','time'],inplace=True)
		
		port=get_port()
		temp=temp.merge(port,on='ship',how='left')
		temp['port_dis']=((temp['x']-temp['x_port'])**2+(temp['y']-temp['y_port'])**2)**0.5
		features=temp.groupby(['group','port','x_port','y_port'],as_index=False)['port_dis'].agg({
			'port_dis_first':'first',
			'port_dis_last':'last',
			'port_dis_min':'min',
			'port_dis_max':'max',
			'port_dis_mean':'mean',
			'port_dis_median':'median',
			'port_dis_quantile25':quantile25,
			'port_dis_quantile25':quantile75,
			'port_dis_std':'std'})
		features=features[['group','port','x_port','y_port']]
		features.to_pickle('./data/port_{}.pkl'.format(name))
	print('port feature done_{}'.format(name))
	return features
	
def get_province(data):
	temp=data.copy()
	features=pd.DataFrame()
	base=[25.173215180565872, 119.27677013681826]
	coastline=pd.read_csv('./data/coastline.csv')
	coastline['xy']=coastline.apply(lambda x:get_xy(base[0],base[1],x.lat,x.lon),axis=1)
	coastline['x']=coastline['xy'].apply(lambda x:x[0])
	coastline['y']=coastline['xy'].apply(lambda x:x[1])
	provinces=list(set(coastline.province))
	coastline_province={}
	for province in provinces:
		if province!='other':
			coastline_province[province]=geom.LineString(coastline[coastline.province==province][['x','y']].values)		
	def min_province_dis(point):
		coastline_province_dis={}	
		for (k,v) in coastline_province.items():
			coastline_province_dis[k]=point.distance(v)
		return (
			min(zip(coastline_province_dis.values(),coastline_province_dis.keys()))[0],
			min(zip(coastline_province_dis.values(),coastline_province_dis.keys()))[1])
	temp['point']=temp.apply(lambda x:geom.Point(x.x,x.y),axis=1)
	temp['pro_dis']=temp.apply(lambda x:min_province_dis(x.point),axis=1)
	temp['province']=temp['pro_dis'].apply(lambda x:x[1])
	temp['min_dis']=temp['pro_dis'].apply(lambda x:x[0])
	temp.sort_values(['ship','min_dis'],inplace=True)
	temp=temp.drop_duplicates('ship')
	le=LabelEncoder()
	temp['province']=le.fit_transform(temp['province'])
	province=temp[['ship','province']]
	return province
	
def get_city(data):
	temp=data.copy()
	features=pd.DataFrame()

	coastcity=pd.read_csv('./data/coastcity.csv')
	coastcity['xy']=coastcity.apply(lambda x:get_xy(base[0],base[1],x.lat,x.lon),axis=1)
	coastcity['x']=coastcity['xy'].apply(lambda x:x[0])
	coastcity['y']=coastcity['xy'].apply(lambda x:x[1])
	cities=list(set(coastcity.city))
	coastcity_city={}
	for city in cities:
		coastcity_city[city]=geom.LineString(coastcity_city[coastcity_city.city==city][['x','y']].values)		
	def min_city_dis(point):
		coastcity_city_dis={}	
		for (k,v) in coastcity_city.items():
			coastcity_city_dis[k]=point.distance(v)
		return (
			min(zip(coastcity_city_dis.values(),coastcity_city_dis.keys()))[0],
			min(zip(coastcity_city_dis.values(),coastcity_city_dis.keys()))[1])
	temp['point']=temp.apply(lambda x:geom.Point(x.x,x.y),axis=1)
	temp['city_dis']=temp.apply(lambda x:min_city_dis(x.point),axis=1)
	temp['city']=temp['city_dis'].apply(lambda x:x[1])
	temp['min_city_dis']=temp['city_dis'].apply(lambda x:x[0])
	temp.sort_values(['ship','min_city_dis'],inplace=True)
	temp=temp.drop_duplicates('ship')
	le=LabelEncoder()
	temp['city']=le.fit_transform(temp['city'])
	province=temp[['ship','city','min_city_dis']]
	return province
	
def get_city_features(data,name):
	temp=data.copy()
	features=pd.DataFrame()
	base=[25.173215180565872, 119.27677013681826]
	coastcity=pd.read_csv('./data/coastcity.csv')
	coastcity['xy']=coastcity.apply(lambda x:get_xy(base[0],base[1],x.lat,x.lon),axis=1)
	coastcity['x']=coastcity['xy'].apply(lambda x:x[0])
	coastcity['y']=coastcity['xy'].apply(lambda x:x[1])
	cities=list(coastcity.city)
	for city in cities:
		city_x=coastcity[coastcity.city==city]['x'].values[0]
		city_y=coastcity[coastcity.city==city]['y'].values[0]
		temp['{}_min_ids'.format(city)]=((temp.x-city_x)**2+(temp.y-city_y)**2)**0.5
		feature=temp.groupby('group',as_index=False)['{}_min_ids'.format(city)].min()
		if features.shape[0]==0:
			features=feature
		else:
			features=features.merge(feature,on='group',how='left')
	return features	

def direction_diff_change(x):
	if x<=-180:
		x=x+360
	elif x>180:
		x=x-360
	else:
		x=x
	return x
	
def get_diff_features(data,name):
	if os.path.exists('./diff_{}.pkl'.format(name)):
		features=pd.read_pickle('./diff_{}.pkl'.format(name))
	else:
		temp=data.copy()
		features=pd.DataFrame()		
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
		temp.sort_values(['ship','group','time'],inplace=True)
		temp['time_gap']=temp.groupby('group')['time'].diff()
		temp['time_gap']=temp.apply(lambda x:x.time_gap.total_seconds()/3600,axis=1)
		temp['x_gap']=temp.groupby('group')['x'].diff()
		temp['y_gap']=temp.groupby('group')['y'].diff()
		temp['x_gap_after']=temp.groupby('group')['x_gap'].shift(-1)
		temp['y_gap_after']=temp.groupby('group')['y_gap'].shift(-1)
		temp['x_after_2']=temp.groupby('group')['x'].shift(-2)
		temp['y_after_2']=temp.groupby('group')['y'].shift(-2)
		temp['angle']=temp.apply(lambda x:angle(x.x_gap,x.x_gap_after,x.y_gap,x.y_gap_after),axis=1)
		temp['angle']=temp.apply(lambda x:x.angle*1 if -x.x_gap*x.y_gap_after+x.y_gap*x.x_gap_after>=0 else x.angle*(-1),axis=1)
		temp['dis_gap']=(temp['x_gap']**2+temp['y_gap']**2)**0.5
		temp['dis_gap_after']=(temp['x_gap_after']**2+temp['y_gap_after']**2)**0.5
		temp['dis_gap_line']=((temp['x']-temp['x_after_2'])**2+(temp['y']-temp['y_after_2'])**2)**0.5
		temp['curvity']=(temp['dis_gap']+temp['dis_gap_after'])/(temp['dis_gap_line']+0.00001)
		temp['speed_gap']=temp['dis_gap']/temp['time_gap']
		temp['speed_gap_after']=temp.groupby('group')['speed_gap'].shift(-1)
		temp['acceleration']=(temp['speed_gap_after']-temp['speed_gap'])/temp['time_gap']
		temp.to_csv('diff.csv',index=False)

		for col in ['angle','curvity','acceleration']:
			feature=temp.groupby('group',as_index=False)[col].agg({
				'{}_min'.format(col):'min',
				'{}_max'.format(col):'max',
				'{}_mean'.format(col):'mean',
				'{}_median'.format(col):'median',
				'{}_std'.format(col):'std',
				'{}_skew'.format(col):'skew',
				'{}_quantile25'.format(col):quantile25,
				'{}_quantile75'.format(col):quantile75})
			if features.shape[0]==0:
				features=feature
			else:
				features=features.merge(feature,on='group',how='left')
			
		for col in ['angle','curvity','acceleration']:
			le=LabelEncoder()
			feature_temp=temp[~temp[col].isnull()]
			if col=='curvity':
				bins=[-0.1,0.5,1,1.5,2,2.5,3,3.5,4,4.5,feature_temp[col].max()]
				feature_temp['{}_cut'.format(col)]=le.fit_transform(pd.cut(feature_temp[col],bins))
			elif col=='acceleration':
				bins=[feature_temp[col].min()-1,-4000,-3000,-2000,-1000,0,1000,2000,3000,4000,feature_temp[col].max()]
				feature_temp['{}_cut'.format(col)]=le.fit_transform(pd.cut(feature_temp[col],bins))
			else:
				feature_temp['{}_cut'.format(col)]=le.fit_transform(pd.cut(feature_temp[col],10))
			feature=feature_temp.groupby(['group','{}_cut'.format(col)])['time'].count()
			feature=feature.unstack().fillna(0)
			feature.columns=['{}_cut_{}_count'.format(col,c) for c in feature.columns]
			feature=feature.reset_index()
			if features.shape[0]==0:
				features=feature
			else:
				features=features.merge(feature,on='group',how='left')				
		features.to_pickle('./diff_{}.pkl'.format(name))
	print('diff feature done_{}'.format(name))
	return features
	
def get_diff_features_multi(num,data,col,func,name):
	if os.path.exists('./data/diff_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/diff_{}.pkl'.format(name))
	else:
		features=multi_func(num,data,col,func)		
		features.to_pickle('./data/diff_{}.pkl'.format(name))
	print('diff feature done_{}'.format(name))
	return features	

def get_w2v_features(data,name):
	if os.path.exists('./w2v_{}.pkl'.format(name)):
		features=pd.read_pickle('./w2v_{}.pkl'.format(name))
	else:
		temp=data.copy()
		features=pd.DataFrame()
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
		temp.sort_values(['ship','group','time'],inplace=True)
		temp['r']=(temp['x']**2+temp['y']**2)**0.5
		temp['the']=temp.apply(lambda x:math.atan2(x.y,x.x)*180/math.pi,axis=1)
		le=LabelEncoder()
		
		temp['x_cut']=le.fit_transform(pd.cut(temp['x'],80))
		temp['y_cut']=le.fit_transform(pd.cut(temp['y'],100))		
		temp['x_y_cut']=le.fit_transform(temp['x_cut'].astype(str)+'_'+temp['y_cut'].astype(str))
		temp['x_y_cut']=temp['x_y_cut'].astype(str)
		x_y_list=temp.groupby('group').agg({'x_y_cut':lambda x:x.tolist()}).reset_index()
		model=Word2Vec(list(x_y_list['x_y_cut']),size=50,window=5,min_count=1,seed=2020,workers=1)
		temp['x_y_cut']=temp['x_y_cut'].apply(lambda x:model[x])
		temp_=temp.groupby('group').agg({'x_y_cut':lambda x:x.tolist()}).reset_index()
		temp_['emb']=temp_['x_y_cut'].apply(lambda x:sum(x))
		feature=pd.DataFrame(np.vstack(temp_['emb']))
		feature.columns=['xy_w2v_feat{}'.format(i) for i in range(feature.shape[1])]
		feature['group']=temp_['group']
		if features.shape[0]==0:
			features=feature
		else:
			features=features.merge(feature,on='group',how='left')
			
		temp['s_cut']=le.fit_transform(pd.cut(temp['speed'],20))
		temp['d_cut']=le.fit_transform(pd.cut(temp['direction'],36))		
		temp['s_d_cut']=le.fit_transform(temp['s_cut'].astype(str)+'_'+temp['d_cut'].astype(str))
		temp['s_d_cut']=temp['s_d_cut'].astype(str)
		s_d_list=temp.groupby('group').agg({'s_d_cut':lambda x:x.tolist()}).reset_index()
		model=Word2Vec(list(s_d_list['s_d_cut']),size=50,window=5,min_count=1,seed=2020,workers=1)
		temp['s_d_cut']=temp['s_d_cut'].apply(lambda x:model[x])
		temp_=temp.groupby('group').agg({'s_d_cut':lambda x:x.tolist()}).reset_index()
		temp_['emb']=temp_['s_d_cut'].apply(lambda x:sum(x))
		feature=pd.DataFrame(np.vstack(temp_['emb']))
		feature.columns=['sd_w2v_feat{}'.format(i) for i in range(feature.shape[1])]
		feature['group']=temp_['group']
		if features.shape[0]==0:
			features=feature
		else:
			features=features.merge(feature,on='group',how='left')
					
		features.to_pickle('./w2v_{}.pkl'.format(name))
	print('w2v feature done_{}'.format(name))
	return features
																	
def stdbscan_cluster(x):
	stdbscan=STDBSCAN(
		col_lat='y',
		col_lon='x',
		col_time='time',
		spatial_threshold=10000,
		temporal_threshold=6000,
		min_neighbors=3)
	results=stdbscan.run(x)
	return results['cluster'].values
	
def stdbscan_cluster_stop(x):
	if x['x'].std()<=200 and x['y'].std()<=200:
		return np.ones(shape=(x.shape[0]))
	else:
		stdbscan=STDBSCAN(
			col_lat='y',
			col_lon='x',
			col_time='time',
			spatial_threshold=500,
			temporal_threshold=9000,
			min_neighbors=21)
		results=stdbscan.run(x)
		return results['cluster'].values

def prepare_data_1(data):
	print('prepare_data_1_before:',data.shape)
	if os.path.exists('./data/data_1.pkl'):
		data=pd.read_pickle('./data/data_1.pkl')
	else:
		train_data=data[~data.type.isnull()]
		base=[train_data.lat.mean(),train_data.lon.mean()]
		print(base)
		data['xy']=data.apply(lambda x:get_xy(base[0],base[1],x.lat,x.lon),axis=1)
		data['x']=data['xy'].apply(lambda x:x[0])
		data['y']=data['xy'].apply(lambda x:x[1])
		data=data.drop(['xy'],axis=1)
		data.to_pickle('./data/data_1.pkl')
	print('prepare_data_1_after:',data.shape)
	return data
			
def prepare_data_2(data):
	cols=list(data.columns)
	data['time']=pd.to_datetime(data['time'],format='%m%d %H:%M:%S')
	data.sort_values(['ship','time'],inplace=True)
	
	data=data[~((data.lat>54)+(data.lat<3)+(data.lon>136)+(data.lon<73))]
	
	data['count']=1
	data['index']=data.groupby('ship')['count'].cumsum()
	cluster=data.groupby('ship').apply(stdbscan_cluster)
	data['cluster']=data['ship'].map(cluster)
	data['cluster_preds']=data.apply(lambda x:x['cluster'][x['index']-1],axis=1)
	
	outer=data.groupby('ship',as_index=False)['cluster_preds'].mean()
	outer_ships=list(outer[outer.cluster_preds==-1]['ship'])
	print('outer_ships',outer_ships)
	data.loc[data.ship.isin(outer_ships),'cluster_preds']=1
	
	data=data[data.cluster_preds!=-1]
	data=data[cols]
	data['speed']=data['speed'].apply(lambda x:20 if x>20 else x)#stat+
	data['direction']=data['direction'].apply(lambda x:x-360 if x>180 else x)
	return data
	
def prepare_data_2_multi(num,data,col,func):
	print('prepare_data_2_before:',data.shape)
	if os.path.exists('./data/data_2.pkl'):
		data=pd.read_pickle('./data/data_2.pkl')
	else:
		data=multi_func(num,data,col,func)
		data.to_pickle('./data/data_2.pkl')
	print('prepare_data_2_after:',data.shape)
	return data
	
def prepare_data_3(data):
	cols=list(data.columns)
	data['time']=pd.to_datetime(data['time'],format='%m%d %H:%M:%S')
	data.sort_values(['ship','time'],inplace=True)
	data['count']=1
	data['index']=data.groupby('ship')['count'].cumsum()
	cluster=data.groupby('ship').apply(stdbscan_cluster_stop)
	data['cluster']=data['ship'].map(cluster)
	data['stop']=data.apply(lambda x:x['cluster'][x['index']-1],axis=1)
	data=data.drop('cluster',axis=1)
	data['stop']=data['stop'].apply(lambda x:1 if x>0 else 0)
	data['diff']=data.groupby('ship')['stop'].diff(1)
	data['count']=data['diff'].apply(lambda x:1 if x!=0 else 0)
	data['group']=data['count'].cumsum()
	
	temp=data.groupby(['ship','group','stop'],as_index=False)['time'].agg({'max':'max','min':'min'})
	temp['time_gap']=temp.apply(lambda x:(x['max']-x['min']).total_seconds()/3600,axis=1)
	groups=list(temp[(temp.time_gap<=1)]['group'])
	data=data[~data.group.isin(groups)]
	data['diff']=data.groupby('ship')['stop'].diff(1)
	data['count']=data['diff'].apply(lambda x:1 if x!=0 else 0)
	data['group']=data['count'].cumsum()
				
	data=data[cols+['group','stop']]
	return data
	
def prepare_data_3_multi(num,data,col,func):
	print('prepare_data_3_before:',data.shape)
	if os.path.exists('./data/data_3.pkl'):
		data=pd.read_pickle('./data/data_3.pkl')
	else:
		data=multi_func(num,data,col,func)
		data.to_pickle('./data/data_3.pkl')
	print('prepare_data_3_after:',data.shape)
	return data
	
def get_f1_score(y_true,y_pred):
	y_pred=y_pred.reshape(3,-1)
	y_pred=np.argmax(y_pred,axis=0)
	score=f1_score(y_true,y_pred,average='macro')
	f1s=f1_score(y_true,y_pred,average=None)
	return 'f1_score',score,True
	
def get_samples():

	train=prepare_data(train_path,'train')
	
	train['type']=train['type'].map(type_map)
	train_ships=list(train[['ship']].drop_duplicates().sample(frac=0.85,random_state=0)['ship'])
	test=train[~train.ship.isin(train_ships)]
	train=train[train.ship.isin(train_ships)]
	y_test=test.drop_duplicates('ship')[['ship','type']].rename(columns={'type':'type_true'})
	test['type']=np.nan
	data=pd.concat([train,test],axis=0)
	
	num=cpu_count()-4
	data=prepare_data_1(data)
	data=prepare_data_2_multi(num,data,'ship',prepare_data_2)
	data_origin=data.copy()				
	data=prepare_data_3_multi(num,data,'ship',prepare_data_3)
	# data_origin['stop']=5
	data_origin['stop']=2
	data_origin['group']=data_origin['ship'].diff()
	data_origin['group']=data_origin['group'].apply(lambda x:1 if x!=0 else 0)
	data_origin['group']=data_origin['group'].cumsum()
	data_origin['group']=data_origin['group']+data.group.max()
	
	label=get_label(data,'stop')
	stat_features=get_stat_features_multi(num,data,'group',get_stat_features,'stop')
	samples_stop=label.merge(stat_features,on='group',how='left')
	corr_features=get_corr_features(data,'stop')
	samples_stop=samples_stop.merge(corr_features,on='group',how='left')
	count_features=get_count_features(data,'stop')
	samples_stop=samples_stop.merge(count_features,on='group',how='left')
	time_features=get_time_features(data,'stop')
	samples_stop=samples_stop.merge(time_features,on='group',how='left')
	coastline_features=get_coastline_features_multi(num,data,'group',get_coastline_features,'stop')
	samples_stop=samples_stop.merge(coastline_features,on='group',how='left')
	# city_features=get_city_features(data,'stop')
	# samples_stop=samples_stop.merge(city_features,on='group',how='left')
	# diff_features=get_diff_features_multi(num,data,'group',get_diff_features,'stop')
	# samples_stop=samples_stop.merge(diff_features,on='group',how='left')
	w2v_features=get_w2v_features(data,'stop')
	samples_stop=samples_stop.merge(w2v_features,on='group',how='left')
	print('samples_stop:',samples_stop.shape)
	
	label=get_label(data_origin,'origin')
	stat_features=get_stat_features_multi(num,data_origin,'group',get_stat_features,'origin')
	samples_origin=label.merge(stat_features,on='group',how='left')
	corr_features=get_corr_features(data_origin,'origin')
	samples_origin=samples_origin.merge(corr_features,on='group',how='left')
	count_features=get_count_features(data_origin,'origin')
	samples_origin=samples_origin.merge(count_features,on='group',how='left')
	time_features=get_time_features(data_origin,'origin')
	samples_origin=samples_origin.merge(time_features,on='group',how='left')
	coastline_features=get_coastline_features_multi(num,data_origin,'group',get_coastline_features,'origin')
	samples_origin=samples_origin.merge(coastline_features,on='group',how='left')
	# city_features=get_city_features(data_origin,'origin')
	# samples_origin=samples_origin.merge(city_features,on='group',how='left')
	# diff_features=get_diff_features_multi(num,data_origin,'group',get_diff_features,'origin')
	# samples_origin=samples_origin.merge(diff_features,on='group',how='left')
	w2v_features=get_w2v_features(data_origin,'origin')
	samples_origin=samples_origin.merge(w2v_features,on='group',how='left')
	print('samples_origin:',samples_origin.shape)
								
	#2,first_stop,#3,last_stop,#4,unique_stop,#5,origin
	# group_n=samples_stop.groupby('ship',as_index=False)['group'].agg({'group_n':'nunique'})
	# samples_stop=samples_stop.merge(group_n,on='ship',how='left')
	# first=samples_stop.groupby('ship',as_index=False).first()
	# first_stop_groups=list(first[(first.stop==1)*(first.group_n>1)]['group'])
	# print('first_stop_groups',len(first_stop_groups))
	# samples_stop.loc[samples_stop.group.isin(first_stop_groups),'stop']=2
	# last=samples_stop.groupby('ship',as_index=False).last()
	# last_stop_groups=list(last[(last.stop==1)*(last.group_n>1)]['group'])
	# print('last_stop_groups',len(last_stop_groups))
	# samples_stop.loc[samples_stop.group.isin(last_stop_groups),'stop']=3
	# unique_group=list(set(samples_stop[(samples_stop.group_n==1)*(samples_stop.stop==1)]['group']))
	# print('unique_group',len(unique_group))
	# samples_stop.loc[samples_stop.group.isin(unique_group),'stop']=4
	# samples_stop=samples_stop.drop('group_n',axis=1)
	
	samples=pd.concat([samples_stop,samples_origin],axis=0)
	samples.to_pickle('./data/samples.pkl')
	
	print('samples:',samples.shape)
	
	for col in ['stop']:
		samples[col]=samples[col].astype('category')
		
	features=[c for c in samples.columns if c not in ['ship','type','time','group','lat','lon']]
	print(len(features),features)
	print(samples.head(20))
	train=samples[~samples.type.isnull()]
	test=samples[samples.type.isnull()]
	return train,test,features,y_test
	#return samples_stop,samples_origin,samples,features,y_test
	#return samples,y_test
	
def get_samples_1():

	train=prepare_data(train_path,'train')
	
	train['type']=train['type'].map(type_map)
	train_ships=list(train[['ship']].drop_duplicates().sample(frac=0.85,random_state=0)['ship'])
	test=train[~train.ship.isin(train_ships)]
	train=train[train.ship.isin(train_ships)]
	y_test=test.drop_duplicates('ship')[['ship','type']].rename(columns={'type':'type_true'})
	test['type']=np.nan
	data=pd.concat([train,test],axis=0)
	
	num=cpu_count()-4
	data=prepare_data_1(data)
	data=prepare_data_2_multi(num,data,'ship',prepare_data_2)
	data_origin=data.copy()				
	data=prepare_data_3_multi(num,data,'ship',prepare_data_3)
	# data_origin['stop']=5
	data_origin['stop']=2
	data_origin['group']=data_origin['ship'].diff()
	data_origin['group']=data_origin['group'].apply(lambda x:1 if x!=0 else 0)
	data_origin['group']=data_origin['group'].cumsum()
	data_origin['group']=data_origin['group']+data.group.max()
	
	label=get_label(data,'stop')
	stat_features=get_stat_features_multi(num,data,'group',get_stat_features,'stop')
	samples_stop=label.merge(stat_features,on='group',how='left')
	corr_features=get_corr_features(data,'stop')
	samples_stop=samples_stop.merge(corr_features,on='group',how='left')
	count_features=get_count_features(data,'stop')
	samples_stop=samples_stop.merge(count_features,on='group',how='left')
	time_features=get_time_features(data,'stop')
	samples_stop=samples_stop.merge(time_features,on='group',how='left')
	coastline_features=get_coastline_features_multi(num,data,'group',get_coastline_features,'stop')
	samples_stop=samples_stop.merge(coastline_features,on='group',how='left')
	# city_features=get_city_features(data,'stop')
	# samples_stop=samples_stop.merge(city_features,on='group',how='left')
	# diff_features=get_diff_features_multi(num,data,'group',get_diff_features,'stop')
	# samples_stop=samples_stop.merge(diff_features,on='group',how='left')
	# w2v_features=get_w2v_features(data,'stop')
	# samples_stop=samples_stop.merge(w2v_features,on='group',how='left')
	print('samples_stop:',samples_stop.shape)
	
	label=get_label(data_origin,'origin')
	stat_features=get_stat_features_multi(num,data_origin,'group',get_stat_features,'origin')
	samples_origin=label.merge(stat_features,on='group',how='left')
	corr_features=get_corr_features(data_origin,'origin')
	samples_origin=samples_origin.merge(corr_features,on='group',how='left')
	count_features=get_count_features(data_origin,'origin')
	samples_origin=samples_origin.merge(count_features,on='group',how='left')
	time_features=get_time_features(data_origin,'origin')
	samples_origin=samples_origin.merge(time_features,on='group',how='left')
	coastline_features=get_coastline_features_multi(num,data_origin,'group',get_coastline_features,'origin')
	samples_origin=samples_origin.merge(coastline_features,on='group',how='left')
	# city_features=get_city_features(data_origin,'origin')
	# samples_origin=samples_origin.merge(city_features,on='group',how='left')
	# diff_features=get_diff_features_multi(num,data_origin,'group',get_diff_features,'origin')
	# samples_origin=samples_origin.merge(diff_features,on='group',how='left')
	# w2v_features=get_w2v_features(data_origin,'origin')
	# samples_origin=samples_origin.merge(w2v_features,on='group',how='left')
	print('samples_origin:',samples_origin.shape)
								
	#2,first_stop,#3,last_stop,#4,unique_stop,#5,origin
	# group_n=samples_stop.groupby('ship',as_index=False)['group'].agg({'group_n':'nunique'})
	# samples_stop=samples_stop.merge(group_n,on='ship',how='left')
	# first=samples_stop.groupby('ship',as_index=False).first()
	# first_stop_groups=list(first[(first.stop==1)*(first.group_n>1)]['group'])
	# print('first_stop_groups',len(first_stop_groups))
	# samples_stop.loc[samples_stop.group.isin(first_stop_groups),'stop']=2
	# last=samples_stop.groupby('ship',as_index=False).last()
	# last_stop_groups=list(last[(last.stop==1)*(last.group_n>1)]['group'])
	# print('last_stop_groups',len(last_stop_groups))
	# samples_stop.loc[samples_stop.group.isin(last_stop_groups),'stop']=3
	# unique_group=list(set(samples_stop[(samples_stop.group_n==1)*(samples_stop.stop==1)]['group']))
	# print('unique_group',len(unique_group))
	# samples_stop.loc[samples_stop.group.isin(unique_group),'stop']=4
	# samples_stop=samples_stop.drop('group_n',axis=1)
	
	samples=pd.concat([samples_stop,samples_origin],axis=0)
	samples.to_pickle('./data/samples.pkl')
	
	print('samples:',samples.shape)
	
	for col in ['stop']:
		samples[col]=samples[col].astype('category')
		
	features=[c for c in samples.columns if c not in ['ship','type','time','group','lat','lon']]
	print(len(features),features)
	print(samples.head(20))
	train=samples[~samples.type.isnull()]
	test=samples[samples.type.isnull()]
	# return train,test,features,y_test
	return samples_stop,samples_origin,samples,features,y_test

def get_samples_2():

	train=prepare_data(train_path,'train')
	
	train['type']=train['type'].map(type_map)
	train_ships=list(train[['ship']].drop_duplicates().sample(frac=0.85,random_state=0)['ship'])
	test=train[~train.ship.isin(train_ships)]
	train=train[train.ship.isin(train_ships)]
	y_test=test.drop_duplicates('ship')[['ship','type']].rename(columns={'type':'type_true'})
	test['type']=np.nan
	data=pd.concat([train,test],axis=0)
	
	num=cpu_count()-4
	data=prepare_data_1(data)
	data=prepare_data_2_multi(num,data,'ship',prepare_data_2)
	data_origin=data.copy()				
	data=prepare_data_3_multi(num,data,'ship',prepare_data_3)
	# data_origin['stop']=5
	data_origin['stop']=2
	data_origin['group']=data_origin['ship'].diff()
	data_origin['group']=data_origin['group'].apply(lambda x:1 if x!=0 else 0)
	data_origin['group']=data_origin['group'].cumsum()
	data_origin['group']=data_origin['group']+data.group.max()
	
	label=get_label(data,'stop')
	stat_features=get_stat_features_multi(num,data,'group',get_stat_features,'stop')
	samples_stop=label.merge(stat_features,on='group',how='left')
	time_features=get_time_features(data,'stop')
	samples_stop=samples_stop.merge(time_features,on='group',how='left')
	diff_features=get_diff_features(data,'stop')
	samples_stop=samples_stop.merge(diff_features,on='group',how='left')
	w2v_features=get_w2v_features(data,'stop')
	samples_stop=samples_stop.merge(w2v_features,on='group',how='left')
	print('samples_stop:',samples_stop.shape)
	
	label=get_label(data_origin,'origin')
	stat_features=get_stat_features_multi(num,data_origin,'group',get_stat_features,'origin')
	samples_origin=label.merge(stat_features,on='group',how='left')
	time_features=get_time_features(data_origin,'origin')
	samples_origin=samples_origin.merge(time_features,on='group',how='left')
	diff_features=get_diff_features(data_origin,'origin')
	samples_origin=samples_origin.merge(diff_features,on='group',how='left')
	w2v_features=get_w2v_features(data_origin,'origin')
	samples_origin=samples_origin.merge(w2v_features,on='group',how='left')
	print('samples_origin:',samples_origin.shape)
								
	#2,first_stop,#3,last_stop,#4,unique_stop,#5,origin
	# group_n=samples_stop.groupby('ship',as_index=False)['group'].agg({'group_n':'nunique'})
	# samples_stop=samples_stop.merge(group_n,on='ship',how='left')
	# first=samples_stop.groupby('ship',as_index=False).first()
	# first_stop_groups=list(first[(first.stop==1)*(first.group_n>1)]['group'])
	# print('first_stop_groups',len(first_stop_groups))
	# samples_stop.loc[samples_stop.group.isin(first_stop_groups),'stop']=2
	# last=samples_stop.groupby('ship',as_index=False).last()
	# last_stop_groups=list(last[(last.stop==1)*(last.group_n>1)]['group'])
	# print('last_stop_groups',len(last_stop_groups))
	# samples_stop.loc[samples_stop.group.isin(last_stop_groups),'stop']=3
	# unique_group=list(set(samples_stop[(samples_stop.group_n==1)*(samples_stop.stop==1)]['group']))
	# print('unique_group',len(unique_group))
	# samples_stop.loc[samples_stop.group.isin(unique_group),'stop']=4
	# samples_stop=samples_stop.drop('group_n',axis=1)
	
	samples=pd.concat([samples_stop,samples_origin],axis=0)
	samples.to_pickle('./data/samples.pkl')
	
	print('samples:',samples.shape)
	
	for col in ['stop']:
		samples[col]=samples[col].astype('category')
		
	features=[c for c in samples.columns if c not in ['ship','type','time','group','lat','lon']]
	print(len(features),features)
	print(samples.head(20))
	train=samples[~samples.type.isnull()]
	test=samples[samples.type.isnull()]
	# return train,test,features,y_test
	return samples_stop,samples_origin,samples,features,y_test
		
def sub(mission='sub'):
	train,test,features,y_test=get_samples()
	
	train=train.sample(frac=1,random_state=0)
	x=train[features]
	y=train['type']
	groups=train.ship.values
	x_test=test[features]
	preds_train=np.zeros((train.shape[0],3))
	preds=np.zeros((test.shape[0],3))
	train_score=[]
	val_score=[]
	importance=pd.DataFrame()
	importance['feats']=features
	for i in range(3):
		importance['im_{}'.format(i)]=0
	importance['im']=0
	kfold=GroupKFold(n_splits=splits)
	for i,(tr_idx,val_idx) in enumerate(kfold.split(x,y,groups=groups)):
		print('model {}'.format(i+1))
		x_train=x.iloc[tr_idx]
		y_train=y.iloc[tr_idx]
		x_val=x.iloc[val_idx]
		y_val=y.iloc[val_idx]
		print(x_train.shape,x_val.shape)
		model=lgb.LGBMClassifier(
			n_estimators=3000,
			learning_rate=0.03,
			objective='logloss',
			num_class=3,		
			random_state=0,
			subsample=0.8,
			colsample_bytree=0.8)
		model.fit(
			x_train,y_train,
			eval_set=[(x_train,y_train),(x_val,y_val)],
			eval_metric='multi_logloss',
			early_stopping_rounds=100,
			verbose=1000,
			categorical_feature=['stop'])
		preds_train[val_idx]+=model.predict_proba(x_val)		
		preds+=model.predict_proba(x_test)/splits
		train_data=train.iloc[tr_idx]
		train_data=train_data.reset_index(drop=True)
		val_data=train.iloc[val_idx]
		val_data=val_data.reset_index(drop=True)
		train_proba=model.predict_proba(x_train)
		train_proba=pd.DataFrame(train_proba,columns=['0_proba','1_proba','2_proba'])
		val_proba=model.predict_proba(x_val)
		val_proba=pd.DataFrame(val_proba,columns=['0_proba','1_proba','2_proba'])
		train_data=pd.concat([train_data,train_proba],axis=1)
		val_data=pd.concat([val_data,val_proba],axis=1)
		train_data=train_data.groupby('ship').agg({
			'type':'mean',
			'0_proba':'mean',
			'1_proba':'mean',
			'2_proba':'mean'})
		val_data=val_data.groupby('ship').agg({
			'type':'mean',
			'0_proba':'mean',
			'1_proba':'mean',
			'2_proba':'mean'})		
		print(
			f1_score(train_data.type,np.argmax(train_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'),
			f1_score(val_data.type,np.argmax(val_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		train_score.append(f1_score(train_data.type,np.argmax(train_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		val_score.append(f1_score(val_data.type,np.argmax(val_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		print('='*50)
		
		explainer=shap.TreeExplainer(model)
		shap_values=explainer.shap_values(x_train)
		for i in range(3):
			importance['im_{}'.format(i)]+=np.mean(np.abs(shap_values[i]),axis=0)/splits		
	for i in range(3):
		importance['im']+=importance['im_{}'.format(i)]
	importance.sort_values('im',ascending=False,inplace=True)
	print(importance.head(500))
	
	preds_train=pd.DataFrame(preds_train,columns=['0_proba','1_proba','2_proba'])
	preds=pd.DataFrame(preds,columns=['0_proba','1_proba','2_proba'])
	train=train.reset_index(drop=True)
	test=test.reset_index(drop=True)
	train=pd.concat([train,preds_train],axis=1)
	test=pd.concat([test,preds],axis=1)
	train=train.groupby('ship').agg({
		'type':'mean',
		'0_proba':'mean',
		'1_proba':'mean',
		'2_proba':'mean'})	
	test=test.groupby('ship',as_index=False).agg({
		'0_proba':'mean',
		'1_proba':'mean',
		'2_proba':'mean'})			
	test['type']=np.argmax(test[['0_proba','1_proba','2_proba']].values,axis=1)
	test=test.merge(y_test,on='ship',how='left')
	# test['type']=test['type'].map(type_map_rev)
	# sub=test[['ship','type']]
	# print(sub.type.value_counts())
	# sub.to_csv(
		# './result/result_b_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
		# index=False,
		# header=False,
		# encoding='utf-8')
	print(np.mean(train_score),np.std(train_score),np.mean(val_score),np.std(val_score))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average=None))
	print(f1_score(test.type_true,test.type,average='macro'))
	print('sub done')
	
def sub_xgb(mission='sub'):
	samples,y_test=get_samples()
	samples_stop=pd.get_dummies(samples[['stop']],columns=['stop'],prefix_sep='_')
	samples_num=samples.drop('stop',axis=1)	
	samples=pd.concat([samples_num,samples_stop],axis=1)
	train=samples[~samples.type.isnull()]
	test=samples[samples.type.isnull()]
	features=[c for c in samples.columns if c not in ['ship','type','time','group','lat','lon']]
	print(len(features),features)
	print(samples.head(20))
	
	train=train.sample(frac=1,random_state=2020)
	x=train[features]
	y=train['type']
	groups=train.ship.values
	x_test=test[features]
	preds_train=np.zeros((train.shape[0],3))
	preds=np.zeros((test.shape[0],3))
	train_score=[]
	val_score=[]
	importance=pd.DataFrame()
	importance['feats']=features
	for i in range(3):
		importance['im_{}'.format(i)]=0
	importance['im']=0
	kfold=GroupKFold(n_splits=splits)
	for i,(tr_idx,val_idx) in enumerate(kfold.split(x,y,groups=groups)):
		print('model {}'.format(i+1))
		x_train=x.iloc[tr_idx]
		y_train=y.iloc[tr_idx]
		x_val=x.iloc[val_idx]
		y_val=y.iloc[val_idx]
		print(x_train.shape,x_val.shape)
		# model=lgb.LGBMClassifier(
			# n_estimators=3000,
			# learning_rate=0.03,
			# objective='multiclass',
			# num_class=3,		
			# random_state=0,
			# subsample=0.8,
			# colsample_bytree=0.8)
		# model.fit(
			# x_train,y_train,
			# eval_set=[(x_train,y_train),(x_val,y_val)],
			# eval_metric='logloss',
			# early_stopping_rounds=100,
			# verbose=1000,
			# categorical_feature=['stop'])
		model=xgb.XGBClassifier(
			n_estimators=5000,
			learning_rate=0.05,
			objective='multi:softmax',
			num_class=3,
			random_state=2020,
			subsample=0.8,
			colsample_bytree=0.8)
		model.fit(
			x_train,y_train,
			eval_set=[(x_train,y_train),(x_val,y_val)],
			eval_metric='mlogloss',
			early_stopping_rounds=100,
			verbose=1000)
			
		preds_train[val_idx]+=model.predict_proba(x_val)		
		preds+=model.predict_proba(x_test)/splits
		train_data=train.iloc[tr_idx]
		train_data=train_data.reset_index(drop=True)
		val_data=train.iloc[val_idx]
		val_data=val_data.reset_index(drop=True)
		train_proba=model.predict_proba(x_train)
		train_proba=pd.DataFrame(train_proba,columns=['0_proba','1_proba','2_proba'])
		val_proba=model.predict_proba(x_val)
		val_proba=pd.DataFrame(val_proba,columns=['0_proba','1_proba','2_proba'])
		train_data=pd.concat([train_data,train_proba],axis=1)
		val_data=pd.concat([val_data,val_proba],axis=1)
		train_data=train_data.groupby('ship').agg({
			'type':'mean',
			'0_proba':'mean',
			'1_proba':'mean',
			'2_proba':'mean'})
		val_data=val_data.groupby('ship').agg({
			'type':'mean',
			'0_proba':'mean',
			'1_proba':'mean',
			'2_proba':'mean'})		
		print(
			f1_score(train_data.type,np.argmax(train_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'),
			f1_score(val_data.type,np.argmax(val_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		train_score.append(f1_score(train_data.type,np.argmax(train_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		val_score.append(f1_score(val_data.type,np.argmax(val_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		print('='*50)
		
		# explainer=shap.TreeExplainer(model)
		# shap_values=explainer.shap_values(x_train)
		# for i in range(3):
			# importance['im_{}'.format(i)]+=np.mean(np.abs(shap_values[i]),axis=0)/splits		
	# for i in range(3):
		# importance['im']+=importance['im_{}'.format(i)]
	# importance.sort_values('im',ascending=False,inplace=True)
	# print(importance)
	
	preds_train=pd.DataFrame(preds_train,columns=['0_proba','1_proba','2_proba'])
	preds=pd.DataFrame(preds,columns=['0_proba','1_proba','2_proba'])
	train=train.reset_index(drop=True)
	test=test.reset_index(drop=True)
	train=pd.concat([train,preds_train],axis=1)
	test=pd.concat([test,preds],axis=1)
	train=train.groupby('ship').agg({
		'type':'mean',
		'0_proba':'mean',
		'1_proba':'mean',
		'2_proba':'mean'})	
	test=test.groupby('ship',as_index=False).agg({
		'0_proba':'mean',
		'1_proba':'mean',
		'2_proba':'mean'})			
	test['type']=np.argmax(test[['0_proba','1_proba','2_proba']].values,axis=1)
	test=test.merge(y_test,on='ship',how='left')
	# test['type']=test['type'].map(type_map_rev)
	# sub=test[['ship','type']]
	# print(sub.type.value_counts())
	# sub.to_csv(
		# './result/result_b_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
		# index=False,
		# header=False,
		# encoding='utf-8')
	print(np.mean(train_score),np.std(train_score),np.mean(val_score),np.std(val_score))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average=None))
	print(f1_score(test.type_true,test.type,average='macro'))
	print('sub done')

def gmean_average(data,threshold=0.97):
	corr=data.corr()
	rank=np.tril(corr.values,-1)
	print(rank)
	m=(rank>0).sum()-(rank>threshold).sum()
	print(m)
	m_gmean,s=0,0
	for i in range(m):
		mx=np.unravel_index(rank.argmin(),rank.shape)
		w=(m-i)/m
		m_gmean+=w*(np.log(data.iloc[:,mx[0]])+np.log(data.iloc[:,mx[1]]))/2
		s+=w
		rank[mx]=1
	m_gmean=np.exp(m_gmean/s)
	return m_gmean
	
def sub_gmean(mission='sub'):
	
	train,test,features,y_test=get_samples()
	
	data_gmean=pd.DataFrame()#
	data_gmean['ship']=test['ship'].values#
	data_gmean['group']=test['group'].values#
	
	train=train.sample(frac=1,random_state=0)
	x=train[features]
	y=train['type']
	groups=train.ship.values
	x_test=test[features]
	preds_train=np.zeros((train.shape[0],3))
	preds=np.zeros((test.shape[0],3))
	train_score=[]
	val_score=[]
	importance=pd.DataFrame()
	importance['feats']=features
	for i in range(3):
		importance['im_{}'.format(i)]=0
	importance['im']=0
	kfold=GroupKFold(n_splits=splits)
	for i,(tr_idx,val_idx) in enumerate(kfold.split(x,y,groups=groups)):
		print('model {}'.format(i+1))
		x_train=x.iloc[tr_idx]
		y_train=y.iloc[tr_idx]
		x_val=x.iloc[val_idx]
		y_val=y.iloc[val_idx]
		print(x_train.shape,x_val.shape)
		model=lgb.LGBMClassifier(
			n_estimators=3000,
			learning_rate=0.03,
			objective='multiclass',
			num_class=3,		
			random_state=0,
			subsample=0.8,
			colsample_bytree=0.8)
		model.fit(
			x_train,y_train,
			eval_set=[(x_train,y_train),(x_val,y_val)],
			eval_metric='logloss',
			early_stopping_rounds=100,
			verbose=1000,
			categorical_feature=['stop'])
		preds_train[val_idx]+=model.predict_proba(x_val)		
		preds+=model.predict_proba(x_test)/splits
		train_data=train.iloc[tr_idx]
		train_data=train_data.reset_index(drop=True)
		val_data=train.iloc[val_idx]
		val_data=val_data.reset_index(drop=True)
		train_proba=model.predict_proba(x_train)
		train_proba=pd.DataFrame(train_proba,columns=['0_proba','1_proba','2_proba'])
		val_proba=model.predict_proba(x_val)
		val_proba=pd.DataFrame(val_proba,columns=['0_proba','1_proba','2_proba'])
		train_data=pd.concat([train_data,train_proba],axis=1)
		val_data=pd.concat([val_data,val_proba],axis=1)
		train_data=train_data.groupby('ship').agg({
			'type':'mean',
			'0_proba':'mean',
			'1_proba':'mean',
			'2_proba':'mean'})
		val_data=val_data.groupby('ship').agg({
			'type':'mean',
			'0_proba':'mean',
			'1_proba':'mean',
			'2_proba':'mean'})		
		print(
			f1_score(train_data.type,np.argmax(train_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'),
			f1_score(val_data.type,np.argmax(val_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		train_score.append(f1_score(train_data.type,np.argmax(train_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		val_score.append(f1_score(val_data.type,np.argmax(val_data[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
		print('='*50)
		
		data_gmean_part=pd.DataFrame(model.predict_proba(x_test),columns=['model_{}_proba_{}'.format(i+1,r) for r in [0,1,2]])#
		data_gmean=pd.concat([data_gmean,data_gmean_part],axis=1)#
				
		# explainer=shap.TreeExplainer(model)
		# shap_values=explainer.shap_values(x_train)
		# for i in range(3):
			# importance['im_{}'.format(i)]+=np.mean(np.abs(shap_values[i]),axis=0)/splits		
	# for i in range(3):
		# importance['im']+=importance['im_{}'.format(i)]
	# importance.sort_values('im',ascending=False,inplace=True)
	# print(importance)
	
	preds_train=pd.DataFrame(preds_train,columns=['0_proba','1_proba','2_proba'])
	preds=pd.DataFrame(preds,columns=['0_proba','1_proba','2_proba'])
	train=train.reset_index(drop=True)
	test=test.reset_index(drop=True)
	train=pd.concat([train,preds_train],axis=1)
	test=pd.concat([test,preds],axis=1)
	train=train.groupby('ship').agg({
		'type':'mean',
		'0_proba':'mean',
		'1_proba':'mean',
		'2_proba':'mean'})	
	test=test.groupby('ship',as_index=False).agg({
		'0_proba':'mean',
		'1_proba':'mean',
		'2_proba':'mean'})			
	test['type']=np.argmax(test[['0_proba','1_proba','2_proba']].values,axis=1)
	test=test.merge(y_test,on='ship',how='left')
	# test['type']=test['type'].map(type_map_rev)
	# sub=test[['ship','type']]
	# print(sub.type.value_counts())
	# sub.to_csv(
		# './result/result_b_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
		# index=False,
		# header=False,
		# encoding='utf-8')
	print(np.mean(train_score),np.std(train_score),np.mean(val_score),np.std(val_score))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average=None))
	print(f1_score(test.type_true,test.type,average='macro'))
	
	def gmean_average(data,threshold=0.97):
		corr=data.corr()
		rank=np.tril(corr.values,-1)
		m=(rank>0).sum()-(rank>threshold).sum()
		print(m)
		m_gmean,s=0,0
		for i in range(m):
			mx=np.unravel_index(rank.argmin(),rank.shape)
			print(mx)
			w=(m-i)/m
			m_gmean+=w*(np.log(data.iloc[:,mx[0]])+np.log(data.iloc[:,mx[1]]))/2
			s+=w
			rank[mx]=1
		m_gmean=np.exp(m_gmean/s)
		return m_gmean
	data_gmean.to_csv('data_gmean.csv',index=False)	
	
	data_gmean=pd.read_csv('data_gmean.csv')
	data_gmean=data_gmean.drop('group',axis=1)
	data_gmean=data_gmean.groupby('ship',as_index=False).mean()
	print(data_gmean)
	for i in range(3):
		data_part=data_gmean[[c for c in data_gmean.columns if 'proba_{}'.format(i) in c]]
		data_gmean['proba_{}_gmean'.format(i)]=gmean_average(data_part)
	data_gmean=data_gmean[['ship','proba_0_gmean','proba_1_gmean','proba_2_gmean']]
	y_test=y_test.merge(data_gmean,on='ship',how='left')
	y_test['type_gmean']=np.argmax(y_test[['ship','proba_0_gmean','proba_1_gmean','proba_2_gmean']].values,axis=1)
	print(f1_score(test.type_true,test.type_gmean,average='macro'))
	print('sub done')

def get_gmean_data(samples,features,name):	
	train=samples[~samples.type.isnull()]
	test=samples[samples.type.isnull()]
	
	data_gmean=pd.DataFrame()#
	data_gmean['ship']=test['ship'].values#
	data_gmean['group']=test['group'].values#
	
	train=train.sample(frac=1,random_state=0)
	x=train[features]
	y=train['type']
	groups=train.ship.values
	x_test=test[features]		
	kfold=GroupKFold(n_splits=splits)
	for i,(tr_idx,val_idx) in enumerate(kfold.split(x,y,groups=groups)):
		print('model {}'.format(i+1))
		x_train=x.iloc[tr_idx]
		y_train=y.iloc[tr_idx]
		x_val=x.iloc[val_idx]
		y_val=y.iloc[val_idx]
		print(x_train.shape,x_val.shape)
		model=lgb.LGBMClassifier(
			n_estimators=3000,
			learning_rate=0.03,
			objective='multiclass',
			num_class=3,		
			random_state=0,
			subsample=0.8,
			colsample_bytree=0.8)
		model.fit(
			x_train,y_train,
			eval_set=[(x_train,y_train),(x_val,y_val)],
			eval_metric='logloss',
			early_stopping_rounds=100,
			verbose=1000,
			categorical_feature=['stop'])
			
		data_gmean_part=pd.DataFrame(model.predict_proba(x_test),columns=['{}_model_{}_proba_{}'.format(name,i+1,r) for r in [0,1,2]])#
		data_gmean=pd.concat([data_gmean,data_gmean_part],axis=1)#
	data_gmean=data_gmean.drop('group',axis=1)
	data_gmean=data_gmean.groupby('ship',as_index=False).mean()
	data_gmean.to_csv('./data/data_gmean_{}.csv'.format(name),index=False)
	print('data_gmean_{} done'.format(name))
	return data_gmean
		
def sub_gmean_all(mission='sub'):
	
	samples_stop_1,samples_origin_1,samples_1,features_1,y_test_1=get_samples_1()
	# data_gmean_stop_1=get_gmean_data(samples_stop_1,features_1,'stop_1')
	# data_gmean_origin_1=get_gmean_data(samples_origin_1,features_1,'origin_1')
	# data_gmean_all_1=get_gmean_data(samples_1,features_1,'all_1')
	# data_gmean_1=data_gmean_stop_1.merge(data_gmean_origin_1,on='ship',how='left')
	# data_gmean_1=data_gmean_1.merge(data_gmean_all_1,on='ship',how='left')
	data_gmean_all_1=pd.read_csv('./data/data_gmean_all_1.csv')
	data_gmean_stop_1=pd.read_csv('./data/data_gmean_stop_1.csv')
	data_gmean_1=data_gmean_all_1.merge(data_gmean_stop_1,on='ship',how='left')
	
	samples_stop_2,samples_origin_2,samples_2,features_2,y_test_2=get_samples_2()
	# data_gmean_stop_2=get_gmean_data(samples_stop_2,features_2,'stop_2')
	# data_gmean_origin_2=get_gmean_data(samples_origin_2,features_2,'origin_2')
	# data_gmean_all_2=get_gmean_data(samples_2,features_2,'all_2')
	# data_gmean_2=data_gmean_stop_2.merge(data_gmean_origin_2,on='ship',how='left')
	# data_gmean_2=data_gmean_2.merge(data_gmean_all_2,on='ship',how='left')
	data_gmean_all_2=pd.read_csv('./data/data_gmean_all_2.csv')
	data_gmean_stop_2=pd.read_csv('./data/data_gmean_stop_2.csv')
	data_gmean_2=data_gmean_all_2.merge(data_gmean_stop_2,on='ship',how='left')
	
	data_gmean=data_gmean_1.merge(data_gmean_2,on='ship',how='left')
	
	print(data_gmean.head())
	for i in range(3):
		data_part=data_gmean[[c for c in data_gmean.columns if 'proba_{}'.format(i) in c]]
		data_gmean['proba_{}'.format(i)]=gmean_average(data_part)
	data_gmean=data_gmean[['ship','proba_0','proba_1','proba_2']]
	y_test_1=y_test_1.merge(data_gmean,on='ship',how='left')
	y_test_1['type_gmean']=np.argmax(y_test_1[['proba_0','proba_1','proba_2']].values,axis=1)
	print(f1_score(y_test_1.type_true,y_test_1.type_gmean,average='macro'))
	print('sub done')
	
if __name__=='__main__':	
	sub()
	#sub_gmean()
	# sub_gmean_all()
	#sub_xgb(mission='sub')
	
# model_1
#_______________________________________________________________________
# baseline,project_2020022603,0.88740
# 0.9999616458219478 0.00011506253415674018 0.9168685846304354 0.010879068145252533
# 0.9169779908694736
# [0.94602402 0.85335476 0.95155519]
# 围网 921
# 拖网 825
# 刺网 254
# 1.0 0.0 0.9169639797400221 0.012379320537955263
# 0.917103296961368
# [0.94636074 0.85311871 0.95183044]
#_______________________________________________________________________
# lat_lon_0,project_2020022701,0.88759
# 0.9998715527874232 0.0003853416377303298 0.9173160921379274 0.011615923640845875
# 0.9177362613821076
# [0.94584728 0.85542651 0.951935  ]
# 围网 920
# 拖网 824
# 刺网 256
# 0.9999014978171463 0.00026125746196111295 0.9151073409955475 0.011530484612483582
# 0.9155668086331622
# [0.94474045 0.85048232 0.95147766]
#_______________________________________________________________________
# coastline_dis,project_2020022803,0.88792
# 0.9999809176746635 5.724697600978379e-05 0.9188609413311593 0.012210116373334533
# 0.9193698248630847
# [0.94477408 0.86045582 0.95287958]
# 围网 924
# 拖网 821
# 刺网 255
# 0.999920726343586 0.0001639919149407428 0.9203948783392857 0.010726063858131872
# 0.9208029834257054
# [0.94602402 0.8621654 0.95421953]
#_______________________________________________________________________
# direction360-x,project_2020022902,0.88991
# 围网 924
# 拖网 820
# 刺网 256
# 1.0 0.0 0.917088446636345 0.011028779729815065
# 0.917502312848634
# [0.94427527 0.85657211 0.95165955]
#_______________________________________________________________________
# stop-0——1——2,project_2020030102,0.89213
# 围网 923
# 拖网 818
# 刺网 259
# 0.9999618552588819 0.00011443422335415397 0.9161970969836004 0.010765629323082632
# 0.9166093358723568
# [0.94490107 0.85358005 0.95134689]
#_______________________________________________________________________
# chusai,project_2020030602,0.89152
# 围网 916
# 拖网 827
# 刺网 257
# 0.9999836526474264 3.4095100074056014e-05 0.9230223610829198 0.006681498137406888
# 0.9230278138164084
# [0.95377958 0.88223373 0.93307013]
#_______________________________________________________________________
# w2v,the,r,project_2020030801,0.8937
# 围网 912
# 拖网 831
# 刺网 257
# 0.9999896637777091 3.1008666872456296e-05 0.9257143981795981 0.0068203946335783925
# 0.9257260212305981
# [0.95473305 0.88824318 0.93420183]
#_______________________________________________________________________
# sd_w2v,project_2020030805,0.89486
# 围网 909
# 拖网 830
# 刺网 261
# 0.9999793309672217 6.200709833509288e-05 0.9268771542000428 0.006256937337734765
# 0.9268205572436123
# [0.9552747 0.88943489 0.93575208]
#_______________________________________________________________________
# shap_0.1,project_2020031102,0.89505
# 围网 904
# 拖网 832
# 刺网 264
# 0.9999631074830468 5.848715494334531e-05 0.9271127312250289 0.007562683604197792
# 0.9270589500186368
# [0.95516135 0.88968449 0.936331 ]




# model_2
# base_line,0.88124
# 围网 928
# 拖网 819
# 刺网 253
# 1.0 0.0 0.9184292153562141 0.011163958922507945
# 0.9189175268286552
# [0.94305239 0.86147705 0.95222314]
#_______________________________________________________________________
# +chusai,0.88717
# 围网 917
# 拖网 827
# 刺网 256
# 0.9999484141268489 5.158605981809788e-05 0.9218013623238921 0.007687280355540695
# 0.9217805335289838
# [0.95207542 0.88424365 0.92902253]

# 0.9999773024407477 6.809267775698125e-05 0.9102297471350136 0.007304910942480085
# 0.9101213102913283
# [0.94171997 0.84170253 0.94694142]
# 0.908881475498648



























































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































