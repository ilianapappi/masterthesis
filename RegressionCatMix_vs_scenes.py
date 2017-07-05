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



df1 = pd.read_csv('/home/11394153/featuresvsent/art_gallery_vsent.csv').drop('id_new',1)
df2 = pd.read_csv('/home/11394153/featuresvsent/bar_vsent.csv').drop('id_new',1)
df4 = pd.read_csv('/home/11394153/featuresvsent/beach_vsent.csv').drop('id_new',1)
df5 = pd.read_csv('/home/11394153/featuresvsent/bedroom_vsent.csv').drop('id_new',1)
df6 = pd.read_csv('/home/11394153/featuresvsent/cafe_vsent.csv').drop('id_new',1)
df7 = pd.read_csv('/home/11394153/featuresvsent/canals_vsent.csv').drop('id_new',1)
df11 = pd.read_csv('/home/11394153/featuresvsent/fields_vsent.csv').drop('id_new',1)
df13 = pd.read_csv('/home/11394153/featuresvsent/forest_vsent.csv').drop('id_new',1)
df14 = pd.read_csv('/home/11394153/featuresvsent/home_vsent.csv').drop('id_new',1)
df18 = pd.read_csv('/home/11394153/featuresvsent/kitchen_vsent.csv').drop('id_new',1)
df24 = pd.read_csv('/home/11394153/featuresvsent/street_vsent.csv').drop('id_new',1)
df26 = pd.read_csv('/home/11394153/featuresvsent/swimming_pool_vsent.csv').drop('id_new',1)
df27 = pd.read_csv('/home/11394153/featuresvsent/Urban_vsent.csv').drop('id_new',1)

#sum of scene samples: 15794?

dfY1 = pd.read_excel('/home/11394153/data/datatxt/art_gallery2.xlsx')['like_count']
dfY2 = pd.read_excel('/home/11394153/data/datatxt/bar2.xlsx')['like_count']
dfY4 = pd.read_excel('/home/11394153/data/datatxt/beach2.xlsx')['like_count']
dfY5 = pd.read_excel('/home/11394153/data/datatxt/bedroom2.xlsx')['like_count']
dfY6 = pd.read_excel('/home/11394153/data/datatxt/cafe2.xlsx')['like_count']
dfY7 = pd.read_excel('/home/11394153/data/datatxt/canals2.xlsx')['like_count']
dfY11 = pd.read_excel('/home/11394153/data/datatxt/fields2.xlsx')['like_count']
dfY13 = pd.read_excel('/home/11394153/data/datatxt/forest2.xlsx')['like_count']
dfY14 = pd.read_excel('/home/11394153/data/datatxt/home2.xlsx')['like_count']
dfY18 = pd.read_excel('/home/11394153/data/datatxt/kitchen2.xlsx')['like_count']
dfY24 = pd.read_excel('/home/11394153/data/datatxt/street2.xlsx')['like_count']
dfY26 = pd.read_excel('/home/11394153/data/datatxt/swimming_pool2.xlsx')['like_count']
dfY27 = pd.read_excel('/home/11394153/data/datatxt/urban2.xlsx')['like_count']



X1_train = df1.loc[0:349,:].values
X1_test = df1.loc[350:,:].values

Y1_train = dfY1.loc[0:349].values
Y1_train = Y1_train.reshape((Y1_train.shape[0],1))
Y1_test = dfY1.loc[350:].values
Y1_test = Y1_test.reshape((Y1_test.shape[0],1))

X2_train = df2.loc[0:729,:].values
X2_test = df2.loc[730:,:].values

Y2_train = dfY2.loc[0:729].values
Y2_train = Y2_train.reshape((Y2_train.shape[0],1))
Y2_test = dfY2.loc[730:].values
Y2_test = Y2_test.reshape((Y2_test.shape[0],1))

X4_train = df4.loc[0:855,:].values
X4_test = df4.loc[856:,:].values

Y4_train = dfY4.loc[0:855].values
Y4_train = Y4_train.reshape((Y4_train.shape[0],1))
Y4_test = dfY4.loc[856:].values
Y4_test = Y4_test.reshape((Y4_test.shape[0],1))


X5_train = df5.loc[0:867,:].values
X5_test = df5.loc[868:,:].values

Y5_train = dfY5.loc[0:867].values
Y5_train = Y5_train.reshape((Y5_train.shape[0],1))
Y5_test = dfY5.loc[868:].values
Y5_test = Y5_test.reshape((Y5_test.shape[0],1))

X6_train = df6.loc[0:1097,:].values
X6_test = df6.loc[1098:,:].values

Y6_train = dfY6.loc[0:1097].values
Y6_train = Y6_train.reshape((Y6_train.shape[0],1))
Y6_test = dfY6.loc[1098:].values
Y6_test = Y6_test.reshape((Y6_test.shape[0],1))

X7_train = df7.loc[0:619,:].values
X7_test = df7.loc[620:,:].values

Y7_train = dfY7.loc[0:619].values
Y7_train = Y7_train.reshape((Y7_train.shape[0],1))
Y7_test = dfY7.loc[620:].values
Y7_test = Y7_test.reshape((Y7_test.shape[0],1))

X11_train = df11.loc[0:884,:].values
X11_test = df11.loc[885:,:].values

Y11_train = dfY11.loc[0:884].values
Y11_train = Y11_train.reshape((Y11_train.shape[0],1))
Y11_test = dfY11.loc[885:].values
Y11_test = Y11_test.reshape((Y11_test.shape[0],1))

X13_train = df13.loc[0:1252,:].values
X13_test = df13.loc[1253:,:].values

Y13_train = dfY13.loc[0:1252].values
Y13_train = Y13_train.reshape((Y13_train.shape[0],1))
Y13_test = dfY13.loc[1253:].values
Y13_test = Y13_test.reshape((Y13_test.shape[0],1))

X14_train = df14.loc[0:915,:].values
X14_test = df14.loc[916:,:].values

Y14_train = dfY14.loc[0:915].values
Y14_train = Y14_train.reshape((Y14_train.shape[0],1))
Y14_test = dfY14.loc[916:].values
Y14_test = Y14_test.reshape((Y14_test.shape[0],1))

X18_train = df18.loc[0:1133,:].values
X18_test = df18.loc[1134:,:].values

Y18_train = dfY18.loc[0:1133].values
Y18_train = Y18_train.reshape((Y18_train.shape[0],1))
Y18_test = dfY18.loc[1134:].values
Y18_test = Y18_test.reshape((Y18_test.shape[0],1))

X24_train = df24.loc[0:985,:].values
X24_test = df24.loc[986:,:].values

Y24_train = dfY24.loc[0:985].values
Y24_train = Y24_train.reshape((Y24_train.shape[0],1))
Y24_test = dfY24.loc[986:].values
Y24_test = Y24_test.reshape((Y24_test.shape[0],1))

X26_train = df26.loc[0:290,:].values
X26_test = df26.loc[291:,:].values

Y26_train = dfY26.loc[0:290].values
Y26_train = Y26_train.reshape((Y26_train.shape[0],1))
Y26_test = dfY26.loc[291:].values
Y26_test = Y26_test.reshape((Y26_test.shape[0],1))

X27_train = df27.loc[0:1079,:].values
X27_test = df27.loc[1080:,:].values

Y27_train = dfY27.loc[0:1079].values
Y27_train = Y27_train.reshape((Y27_train.shape[0],1))
Y27_test = dfY27.loc[1080:].values
Y27_test = Y27_test.reshape((Y27_test.shape[0],1))

print X1_train.shape, Y1_train.shape, X1_test.shape, Y1_test.shape
print X2_train.shape, Y2_train.shape, X2_test.shape, Y2_test.shape
print X4_train.shape, Y4_train.shape, X4_test.shape, Y4_test.shape
print X5_train.shape, Y5_train.shape, X5_test.shape, Y5_test.shape
print X6_train.shape, Y6_train.shape, X6_test.shape, Y6_test.shape
print X7_train.shape, Y7_train.shape, X7_test.shape, Y7_test.shape
print X11_train.shape, Y11_train.shape, X11_test.shape, Y11_test.shape
print X13_train.shape, Y13_train.shape, X13_test.shape, Y13_test.shape
print X14_train.shape, Y14_train.shape, X14_test.shape, Y14_test.shape
print X18_train.shape, Y18_train.shape, X18_test.shape, Y18_test.shape
print X24_train.shape, Y24_train.shape, X24_test.shape, Y24_test.shape
print X26_train.shape, Y26_train.shape, X26_test.shape, Y26_test.shape
print X27_train.shape, Y27_train.shape, X27_test.shape, Y27_test.shape

#------------------------------------------------------------------------------>


X_train = np.concatenate((X1_train,X2_train,X4_train,X5_train,X6_train,X7_train,X11_train,X13_train,X14_train,X18_train,X24_train,X26_train,X27_train), axis=0)

print X_train.shape



X_test = np.concatenate((X1_test,X2_test,X4_test,X5_test,X6_test,X7_test,X11_test,X13_test,X14_test,X18_test,X24_test,X26_test,X27_test), axis=0)

print X_test.shape


Y_train = np.concatenate((Y1_train,Y2_train,Y4_train,Y5_train,Y6_train,Y7_train,Y11_train,Y13_train,Y14_train,Y18_train,Y24_train,Y26_train,Y27_train), axis=0)

print Y_train.shape

Y_test = np.concatenate((Y1_test,Y2_test,Y4_test,Y5_test,Y6_test,Y7_test,Y11_test,Y13_test,Y14_test,Y18_test,Y24_test,Y26_test,Y27_test), axis=0)

print Y_test.shape

#print X_train[2,:]
#print np.argwhere(np.isnan(X_train))


X_train = normalize(X_train,norm='l1')
X_test = normalize(X_test,norm='l1')

Y_train = np.log(Y_train+np.ones((Y_train.shape[0],1)))
Y_test = np.log(Y_test+np.ones((Y_test.shape[0],1)))
#np.save('YtestSCENE.npy',Y_test)


#lsC = [0.01,0.1,1,10,100,1000]

#for C in lsC:
    
print "SVR"
#print C
modelHLvisual= SVR(C=1000)
modelHLvisual.fit(X_train,Y_train)
predictionsHL = modelHLvisual.predict(X_test)
np.save('YpredSCENEvs.npy',predictionsHL)
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
