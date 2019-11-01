from surprise import SVD,SVDpp
from surprise.model_selection import KFold
from surprise import accuracy
import pandas as pd
import os
from surprise import Reader,Dataset
import time

def get_data(path):
    if not os.path.exists(path):
        print('开始转换用户评分数据')
        fp=pd.read_table('ml-10M100k/ratings.dat',sep='::',engine='python',names=['userid','movieid','rating','timestamp'])
        fp.to_csv(path,index=False)
        print('over!')
    data=pd.read_csv(path)
    data['rating']=data['rating'].astype(int)
    print(data.shape)
    print(data.head())
    reader = Reader(line_format='user item rating', sep=',', skip_lines=1)
    dataset = Dataset.load_from_df(data.iloc[:, :3], reader=reader)
    return dataset

def funksvd(dataset):
    start=time.time()
    algo=SVD(biased=False)
    kf=KFold(n_splits=5)
    for trainset,testset in kf.split(dataset):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('funksvd花分钟数为：',(end-start)/60)
    return acc

def biassvd(dataset):
    start=time.time()
    algo=SVD(biased=True)
    kf=KFold(n_splits=5)
    for trainset,testset in kf.split(dataset):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('biassvd花分钟数为：',(end-start)/60)
    return acc

def svdpp(dataset):
    start=time.time()
    algo=SVDpp()
    kf=KFold(n_splits=5)
    for trainset,testset in kf.split(dataset):
        algo.fit(trainset)
        predictions=algo.test(testset)
        acc=accuracy.rmse(predictions,verbose=True)
    end=time.time()
    print('svdpp花分钟数为：',(end-start)/60)
    return acc


if __name__=='__main__':
    dataset=get_data('ratings.csv')
    acc1 = funksvd(dataset)
    print('funksvd的rsme为', acc1)
    acc2 = biassvd(dataset)
    print('biassvd的rmse为', acc2)
    acc3 = svdpp(dataset)
    print('svdpp的rmse为', acc3)

'''
(10000054, 4)
   userid  ...  timestamp
0       1  ...  838985046
1       1  ...  838983525
2       1  ...  838983392
3       1  ...  838983421
4       1  ...  838983392

[5 rows x 4 columns]
RMSE: 0.8307
RMSE: 0.8306
RMSE: 0.8313
RMSE: 0.8316
RMSE: 0.8310
funksvd花分钟数为： 46.26817183097204
funksvd的rsme为 0.8309751120423668
RMSE: 0.8250
RMSE: 0.8254
RMSE: 0.8243
RMSE: 0.8245
RMSE: 0.8247
biassvd花分钟数为： 46.35041272640228
biassvd的rmse为 0.8246529152447132

#不知道是不是1000万的数据太大 SVDPP跑不出结果

'''