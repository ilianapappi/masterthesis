import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.preprocessing import normalize
from sklearn import model_selection
from sklearn.decomposition import PCA
import scipy
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
#from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import math
from sklearn.model_selection import train_test_split

#------------------------------------------------------------>

df1 = pd.read_csv('/home/11394153/featuresvsent/wendys_visualfeats.csv')
df1 = df1.drop(df1.columns[0],axis=1)
print df1.shape

X_train = df1.loc[0:5965,:].values
print X_train.shape
print(type(X_train))

X_test = df1.loc[5966:,:].values
print X_test.shape
print(type(X_test))

df2 = pd.read_excel('/home/11394153/data/datatxt/wendys2_clean.xlsx')
df3 = df2['imageLikeCount']
print df3.shape

Y_train = df3.loc[0:5965].values
Y_train = Y_train.reshape((Y_train.shape[0],1))
print(type(Y_train))
print Y_train.shape

Y_test = df3.loc[5966:].values
Y_test = Y_test.reshape((Y_test.shape[0],1))
print(type(Y_test))
print Y_test.shape

#------------------------------------------------------------>

X_train = normalize(X_train,norm='l1')
print X_train.shape
X_test = normalize(X_test,norm='l1')
print X_test.shape

Y_train = np.log(Y_train+np.ones((Y_train.shape[0],1)))
print Y_train.shape
Y_test = np.log(Y_test+np.ones((Y_test.shape[0],1)))
print Y_test.shape
#np.save('YtestWENDYS.npy',Y_test)

#--------------------------------------------------------------->

#lsC = [0.01,0.1,1,10,100,1000]

#for C in lsC:
print "SVR"
#print C
modelHLvisual= SVR(C=1000)
modelHLvisual.fit(X_train,Y_train)
predictionsHL = modelHLvisual.predict(X_test)
np.save('YpredWENDYSvs.npy',predictionsHL)
print predictionsHL.shape
        
spearmanscoeff = scipy.stats.spearmanr(Y_test,predictionsHL)
print spearmanscoeff


print "RandomForrest100"

model4 = RandomForestRegressor(n_estimators=100)

model4.fit(X_train,Y_train)

preds4 = model4.predict(X_test)
print preds4.shape

spearmanscoeff4 = scipy.stats.spearmanr(Y_test,preds4)
print spearmanscoeff4

print "MLP"

model5 = MLPRegressor()
model5.fit(X_train,Y_train)

preds5 = model5.predict(X_test)
print preds5.shape

spearmanscoeff5 = scipy.stats.spearmanr(Y_test,preds5)
print spearmanscoeff5
