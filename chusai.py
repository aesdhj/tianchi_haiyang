import pandas as pd
pd.options.display.max_columns=None
pd.options.display.max_rows=1000
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler,LabelEncoder
from sklearn.neighbors import NearestNeighbors
import lightgbm as lgb
import shap
import math
from stdbscan import STDBSCAN
import itertools
from scipy import stats
from scipy.spatial.distance import cdist, euclidean
from datetime import datetime
from gensim.models import Word2Vec,Doc2Vec
from gensim.models.doc2vec import TaggedDocument

test_path='./hy_round1_testA_20200102/'
train_path='./hy_round1_train_20200102/'
type_map={'拖网':0,'刺网':1,'围网':2}
type_map_rev={0:'拖网',1:'刺网',2:'围网'}
splits=10

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
		
#def geometric_median(x,eps=1e-5):
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

def quantile25(x):
	return x.quantile(0.25)
	
def quantile75(x):
	return x.quantile(0.75)
							
def get_stat_features(data,name):
	if os.path.exists('./data/stat_{}.pkl'.format(name)):
		features=pd.read_pickle('./data/stat_{}.pkl'.format(name))
	else:
		temp=data.copy()
		temp['time']=pd.to_datetime(temp['time'],format='%m%d %H:%M:%S')
		temp.sort_values(['ship','group','time'],inplace=True)
		features=pd.DataFrame()
		for col in ['x','y','speed','direction']:		
			feature=temp.groupby('group',as_index=False)[col].agg({
				'{}_min'.format(col):'min',
				'{}_max'.format(col):'max',
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
		
		points=temp.groupby('group').apply(geometric_median)#stat+N
		points=points.reset_index().rename(columns={0:'points'})
		points['x_geometric_median']=points.points.apply(lambda x:x[0])
		points['y_geometric_median']=points.points.apply(lambda x:x[1])
		points=points[['group','x_geometric_median','y_geometric_median']]
		features=features.merge(points,on='group',how='left')
		temp=temp.merge(points,on='group',how='left')
		temp['dis_geometric_median']=((temp.x-temp.x_geometric_median)**2+(temp.y-temp.y_geometric_median)**2)**0.5
		feature=temp.groupby('group',as_index=False)['dis_geometric_median'].agg({
			'dis_geometric_median_min':'min',
			'dis_geometric_median_max':'max',
			'dis_geometric_median_mean':'mean',
			'dis_geometric_median_median':'median',
			'dis_geometric_median_std':'std',
			'dis_geometric_median_quantile25':quantile25,
			'dis_geometric_median_quantile75':quantile75,
			})
		features=features.merge(feature,on='group',how='left')
					
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
	if os.path.exists('./data/time.pkl_{}'.format(name)):
		features=pd.read_pickle('./data/time.pkl_{}'.format(name))
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
		
		features.to_pickle('./data/time.pkl_{}'.format(name))
	print('time feature done_{}'.format(name))
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
	
def prepare_data_2(data):
	print('prepare_data_2_before:',data.shape)
	if os.path.exists('./data/data_2.pkl'):
		data=pd.read_pickle('./data/data_2.pkl')
	else:
		cols=list(data.columns)
		data['time']=pd.to_datetime(data['time'],format='%m%d %H:%M:%S')
		data.sort_values(['ship','time'],inplace=True)
		data['count']=1
		data['index']=data.groupby('ship')['count'].cumsum()
		cluster=data.groupby('ship').apply(stdbscan_cluster)
		data['cluster']=data['ship'].map(cluster)
		data['cluster_preds']=data.apply(lambda x:x['cluster'][x['index']-1],axis=1)
		data=data[data.cluster_preds!=-1]
		data=data[cols]
		data['speed']=data['speed'].apply(lambda x:20 if x>20 else x)#stat+
		data.to_pickle('./data/data_2.pkl')
	print('prepare_data_2_after:',data.shape)
	return data
	
def prepare_data_3(data):
	print('prepare_data_3_before:',data.shape)
	if os.path.exists('./data/data_3.pkl'):
		data=pd.read_pickle('./data/data_3.pkl')
	else:
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
		
		data.to_pickle('./data/data_3.pkl')
	print('prepare_data_3_after:',data.shape)
	return data

def prepare_data_4(data):
	print('prepare_data_4_before:',data.shape)
	if os.path.exists('./data/data_4.pkl'):
		data=pd.read_pickle('./data/data_4.pkl')
	else:
		cols=list(data.columns)
		data['time']=pd.to_datetime(data['time'],format='%m%d %H:%M:%S')
		data.sort_values(['ship','time'],inplace=True)
		data['x_gap']=data.groupby('ship')['x'].diff()
		data['x_gap_shift']=data.groupby('ship')['x_gap'].shift(-1)
		data['y_gap']=data.groupby('ship')['y'].diff()
		data['y_gap_shift']=data.groupby('ship')['y_gap'].shift(-1)
		data['angle_1']=data.apply(lambda x:math.atan2(x.y_gap,x.x_gap)*180/math.pi,axis=1)
		data['angle_2']=data.apply(lambda x:math.atan2(x.y_gap_shift,x.x_gap_shift)*180/math.pi,axis=1)
		data['angle']=data.apply(lambda x:angle(x.x_gap,x.x_gap_shift,x.y_gap,x.y_gap_shift),axis=1)
		data['angle_line']=(data['angle']<=12)*(data['angle']>=0)
		data['angle_line']=data['angle_line'].astype(int)
		data['angle_line']=data.apply(lambda x:1 if ((x.angle_1+x.angle_2)!=0)*((x.angle_1*x.angle_2)==0) else x.angle_line,axis=1)
		data['angle_line_diff']=data.groupby('ship')['angle_line'].diff()
		#angel turn
		data['angle_line_diff']=data['angle_line_diff'].apply(lambda x:1 if x!=0 else 0)
		data['group']=data['angle_line_diff'].cumsum()
		n=data.groupby('group',as_index=False)['ship'].agg({'n':'count'})
		data=data.merge(n,on='group',how='left')
		data['dis_line']=(data.x_gap**2+data.y_gap**2)**0.5
		data['time_line']=data.groupby('ship')['time'].diff()
		data['time_line']=data.apply(lambda x:x.time_line.total_seconds()/3600,axis=1)
		temp=data[data.angle_line==1].groupby(['ship','group'],as_index=False).agg({
			'time_line':'sum',
			'dis_line':'sum',
			'n':'mean'})	
		groups=list(temp[(temp.time_line>4)*(temp.dis_line>24000)*(temp.n>6)]['group'])
		data=data[data.group.isin(groups)]		
		data=data[cols+['group']]
		
		data.to_pickle('./data/data_4.pkl')
	print('prepare_data_4_after:',data.shape)
	return data
		
def get_f1_score(y_true,y_pred):
	y_pred=y_pred.reshape(3,-1)
	y_pred=np.argmax(y_pred,axis=0)
	score=f1_score(y_true,y_pred,average='macro')
	f1s=f1_score(y_true,y_pred,average=None)
	return 'f1_score',score,True
	
def get_samples():
	if os.path.exists('./data/samples.pkl'):
		samples=pd.read_pickle('./data/samples.pkl')
	else:
		test=prepare_data(test_path,'test')
		train=prepare_data(train_path,'train')
		train['type']=train['type'].map(type_map)
		data=pd.concat([train,test],axis=0)
		
		data=prepare_data_2(data)
		data_origin=data.copy()
		data_line=prepare_data_4(data.copy())					
		data=prepare_data_3(data)
		data_origin['stop']=5
		data_origin['group']=data_origin['ship'].diff()
		data_origin['group']=data_origin['group'].apply(lambda x:1 if x!=0 else 0)
		data_origin['group']=data_origin['group'].cumsum()
		data_origin['group']=data_origin['group']+data.group.max()
		data_line['stop']=6
		data_line['group']=data_line['group'].diff()
		data_line['group']=data_line['group'].apply(lambda x:1 if x!=0 else 0)
		data_line['group']=data_line['group'].cumsum()
		data_line['group']=data_line['group']+data_origin.group.max()
		data_origin=pd.concat([data_origin,data_line],axis=0)
		data_origin=data_origin.reset_index(drop=True)
						
		label=get_label(data,'stop')
		stat_features=get_stat_features(data,'stop')
		samples_stop=label.merge(stat_features,on='group',how='left')
		corr_features=get_corr_features(data,'stop')
		samples_stop=samples_stop.merge(corr_features,on='group',how='left')
		count_features=get_count_features(data,'stop')
		samples_stop=samples_stop.merge(count_features,on='group',how='left')
		time_features=get_time_features(data,'stop')
		samples_stop=samples_stop.merge(time_features,on='group',how='left')
		print('samples_stop:',samples_stop.shape)
		
		label=get_label(data_origin,'origin')
		stat_features=get_stat_features(data_origin,'origin')
		samples_origin=label.merge(stat_features,on='group',how='left')
		corr_features=get_corr_features(data_origin,'origin')
		samples_origin=samples_origin.merge(corr_features,on='group',how='left')
		count_features=get_count_features(data_origin,'origin')
		samples_origin=samples_origin.merge(count_features,on='group',how='left')
		time_features=get_time_features(data_origin,'origin')
		samples_origin=samples_origin.merge(time_features,on='group',how='left')
		print('samples_origin:',samples_origin.shape)
									
		#2,first_stop,#3,last_stop,#4,unique_stop,#5,origin,#6,line
		group_n=samples_stop.groupby('ship',as_index=False)['group'].agg({'group_n':'nunique'})
		samples_stop=samples_stop.merge(group_n,on='ship',how='left')
		first=samples_stop.groupby('ship',as_index=False).first()
		first_stop_groups=list(first[(first.stop==1)*(first.group_n>1)]['group'])
		print('first_stop_groups',len(first_stop_groups))
		samples_stop.loc[samples_stop.group.isin(first_stop_groups),'stop']=2
		last=samples_stop.groupby('ship',as_index=False).last()
		last_stop_groups=list(last[(last.stop==1)*(last.group_n>1)]['group'])
		print('last_stop_groups',len(last_stop_groups))
		samples_stop.loc[samples_stop.group.isin(last_stop_groups),'stop']=3
		unique_group=list(set(samples_stop[(samples_stop.group_n==1)*(samples_stop.stop==1)]['group']))
		print('unique_group',len(unique_group))
		samples_stop.loc[samples_stop.group.isin(unique_group),'stop']=4
		samples_stop=samples_stop.drop('group_n',axis=1)
		
		samples=pd.concat([samples_stop,samples_origin],axis=0)
		print('samples:',samples.shape)
		
		for col in ['stop']:
			samples[col]=samples[col].astype('category')
		
		samples.to_pickle('./data/samples.pkl')
		
	features=[c for c in samples.columns if c not in ['ship','type','time','group']]
	print(len(features),features)
	print(samples.head(20))
	train=samples[~samples.type.isnull()]
	test=samples[samples.type.isnull()]
	return train,test,features
	
def sub(mission='sub'):
	train,test,features=get_samples()
	
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
	test['type']=test['type'].map(type_map_rev)
	sub=test[['ship','type']]
	print(sub.type.value_counts())
	sub.to_csv(
		'./result/result_{}.csv'.format(datetime.now().strftime('%Y%m%d_%H%M%S')),
		index=False,
		header=False,
		encoding='utf-8')
	print(np.mean(train_score),np.std(train_score),np.mean(val_score),np.std(val_score))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average='macro'))
	print(f1_score(train.type,np.argmax(train[['0_proba','1_proba','2_proba']].values,axis=1),average=None))
	print('sub done')
	
sub()

'''
# base
# 0.9876520491765707 0.013548966035398522 0.8834898764726586 0.009184166365643395
# 0.883668280891257
# [0.9439413  0.85830371 0.84875983]
#_______________________________________________________________________
# stat+
# 0.9921913264755153 0.02253640352775487 0.8931125639560173 0.0171338914606894
# 0.893434199790717
# [0.94706691 0.87341129 0.8598244 ]
#_______________________________________________________________________
# eval_metric logloss
# 1.0 				 0.0 				 0.8936093075293778 0.014601584004587836
# 0.8940812292947342
# [0.94711483 0.87772704 0.85740181]
#_______________________________________________________________________
# count
# 0.9999775637169913 6.730884902635958e-05 0.8942567765489438 0.013741061212222253
# 0.8946386406396455
# [0.94862385 0.87329631 0.86199575]
#_______________________________________________________________________
# stop
# 1.0 0.0 0.9110911006210838 0.013051784420234178
# 0.9112388129001046
# [0.95581502 0.89923274 0.87866868]
#_______________________________________________________________________
# time_feature 0.8805
# 1.0 0.0 0.9141214676401017 0.014534759266675556
# 0.9142689885972303
# [0.95741758 0.90594814 0.87944124]
#_______________________________________________________________________
# stop_1_2_3_4 0.87903
# 1.0 0.0 0.9156185597171984 0.012882911767418697
# 0.9159570809908719
# [0.95694985 0.90742625 0.88349515]
'''





































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































