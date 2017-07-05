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


df19 = pd.read_csv('/home/11394153/featureskeras/pets_hl.csv',header=None)
df22 = pd.read_csv('/home/11394153/featureskeras/selfie_hl.csv',header=None)

#sum of samples: 1467 files 

dfY19 = pd.read_excel('/home/11394153/data/datatxt/pets2.xlsx')['like_count']
dfY22 = pd.read_excel('/home/11394153/data/datatxt/selfie2.xlsx')['like_count']

#------------------------------------------------------>

X19_train = df19.loc[0:527,:].values
X19_test = df19.loc[528:,:].values

Y19_train = dfY19.loc[0:527].values
Y19_train = Y19_train.reshape((Y19_train.shape[0],1))
Y19_test = dfY19.loc[528:].values
Y19_test = Y19_test.reshape((Y19_test.shape[0],1))

X22_train = df22.loc[0:499,:].values
X22_test = df22.loc[500:,:].values

Y22_train = dfY22.loc[0:499].values
Y22_train = Y22_train.reshape((Y22_train.shape[0],1))
Y22_test = dfY22.loc[500:].values
Y22_test = Y22_test.reshape((Y22_test.shape[0],1))

X_train = np.concatenate((X19_train,X22_train),axis=0)

print X_train.shape

X_test = np.concatenate((X19_test,X22_test),axis=0)

print X_test.shape


Y_train = np.concatenate((Y19_train,Y22_train),axis=0)

print Y_train.shape

Y_test = np.concatenate((Y19_test,Y22_test),axis=0)

print Y_test.shape

#------------------------------------------------------->


X_train = normalize(X_train,norm='l1')
X_test = normalize(X_test,norm='l1')

Y_train = np.log(Y_train+np.ones((Y_train.shape[0],1)))
Y_test = np.log(Y_test+np.ones((Y_test.shape[0],1)))
np.save('YtestPEOPETShl,npy',Y_test)

#lsC = [0.01,0.1,1,10,100,1000]

#for C in lsC:
    
print "SVR"
#print C
modelHLvisual= SVR(C=0.1)
modelHLvisual.fit(X_train,Y_train)
predictionsHL = modelHLvisual.predict(X_test)
np.save('YpredPEOPETShl.npy',predictionsHL)
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

model5 = MLPRegressor(hidden_layer_sizes=(1000, ))
model5.fit(X_train,Y_train)

preds5 = model5.predict(X_test)
print preds5.shape

spearmanscoeff5 = scipy.stats.spearmanr(Y_test,preds5)
print spearmanscoeff5





