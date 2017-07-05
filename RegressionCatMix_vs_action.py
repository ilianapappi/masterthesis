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


#---------------------------------------



df3 = pd.read_csv('/home/11394153/featuresvsent/basketball_vsent.csv').drop('id_new',1)
df8 = pd.read_csv('/home/11394153/featuresvsent/climbing_vsent.csv').drop('id_new',1)
df9 = pd.read_csv('/home/11394153/featuresvsent/cycling_vsent.csv').drop('id_new',1)
df10 = pd.read_csv('/home/11394153/featuresvsent/dance_vsent.csv').drop('id_new',1)
df12 = pd.read_csv('/home/11394153/featuresvsent/football_vsent.csv').drop('id_new',1)
df15 = pd.read_csv('/home/11394153/featuresvsent/horseriding_vsent.csv').drop('id_new',1)
df16 = pd.read_csv('/home/11394153/featuresvsent/hug_vsent.csv').drop('id_new',1)
df17 = pd.read_csv('/home/11394153/featuresvsent/kiss_vsent.csv').drop('id_new',1)
df20 = pd.read_csv('/home/11394153/featuresvsent/playing_music_vsent.csv').drop('id_new',1)
df21 = pd.read_csv('/home/11394153/featuresvsent/running_vsent.csv').drop('id_new',1)
df23 = pd.read_csv('/home/11394153/featuresvsent/ski_vsent.csv').drop('id_new',1)
df25 = pd.read_csv('/home/11394153/featuresvsent/surfing_vsent.csv').drop('id_new',1)

#sum of action samples: 13205?

df3tr = df3.loc[0:738,:]
df3ts = df3.loc[739:,:]
df8tr = df8.loc[0:705,:]
df8ts = df8.loc[706:,:]
df9tr = df9.loc[0:1250,:]
df9ts = df9.loc[1251:,:]
df10tr = df10.loc[0:331,:]
df10ts = df10.loc[332:,:]
df12tr = df12.loc[0:1084,:]
df12ts = df12.loc[1085:,:]
df15tr = df15.loc[0:880,:]
df15ts = df15.loc[881:,:]
df16tr = df16.loc[0:903,:]
df16ts = df16.loc[904:,:]
df17tr = df17.loc[0:637,:]
df17ts = df17.loc[638:,:]
df20tr = df20.loc[0:129,:]
df20ts = df20.loc[130:,:]
df21tr = df21.loc[0:1367,:]
df21ts = df21.loc[1368:,:]
df23tr = df23.loc[0:886,:]
df23ts = df23.loc[887:,:]
df25tr = df25.loc[0:330,:]
df25ts = df25.loc[331:,:]

dfY3 = pd.read_excel('/home/11394153/data/datatxt/basketball2.xlsx')['like_count']
dfY8 = pd.read_excel('/home/11394153/data/datatxt/climbing2.xlsx')['like_count']
dfY9 = pd.read_excel('/home/11394153/data/datatxt/cycling2.xlsx')['like_count']
dfY10 = pd.read_excel('/home/11394153/data/datatxt/dance2.xlsx')['like_count']
dfY12 = pd.read_excel('/home/11394153/data/datatxt/football2.xlsx')['like_count']
dfY15 = pd.read_excel('/home/11394153/data/datatxt/horseriding2.xlsx')['like_count']
dfY16 = pd.read_excel('/home/11394153/data/datatxt/hug2.xlsx')['like_count']
dfY17 = pd.read_excel('/home/11394153/data/datatxt/kiss2.xlsx')['like_count']
dfY20 = pd.read_excel('/home/11394153/data/datatxt/playing_music2.xlsx')['like_count']
dfY21 = pd.read_excel('/home/11394153/data/datatxt/running2.xlsx')['like_count']
dfY23 = pd.read_excel('/home/11394153/data/datatxt/ski2.xlsx')['like_count']
dfY25 = pd.read_excel('/home/11394153/data/datatxt/surfing2.xlsx')['like_count']


lb3tr = dfY3.loc[0:738]
lb3ts = dfY3.loc[739:]
lb8tr = dfY8.loc[0:705]
lb8ts = dfY8.loc[706:]
lb9tr = dfY9.loc[0:1250]
lb9ts = dfY9.loc[1251:]
lb10tr = dfY10.loc[0:331]
lb10ts = dfY10.loc[332:]
lb12tr = dfY12.loc[0:1084]
lb12ts = dfY12.loc[1085:]
lb15tr = dfY15.loc[0:880]
lb15ts = dfY15.loc[881:]
lb16tr = dfY16.loc[0:903]
lb16ts = dfY16.loc[904:]
lb17tr = dfY17.loc[0:637]
lb17ts = dfY17.loc[638:]
lb20tr = dfY20.loc[0:129]
lb20ts = dfY20.loc[130:]
lb21tr = dfY21.loc[0:1367]
lb21ts = dfY21.loc[1368:]
lb23tr = dfY23.loc[0:886]
lb23ts = dfY23.loc[887:]
lb25tr = dfY25.loc[0:330]
lb25ts = dfY25.loc[331:]

dfXtrain1 = [df3tr,df8tr,df9tr,df10tr,df12tr,df15tr,df16tr,df17tr,df20tr,df21tr,df23tr,df25tr]
dfXtrain = pd.concat(dfXtrain1)
print dfXtrain.shape

dfXtest1 = [df3ts,df8ts,df9ts,df10ts,df12ts,df15ts,df16ts,df17ts,df20ts,df21ts,df23ts,df25ts]
dfXtest = pd.concat(dfXtest1)
print dfXtest.shape

dfYtrain1 = [lb3tr,lb8tr,lb9tr,lb10tr,lb12tr,lb15tr,lb16tr,lb17tr,lb20tr,lb21tr,lb23tr,lb25tr]
dfYtrain = pd.concat(dfYtrain1)
print dfYtrain.shape

dfYtest1 = [lb3ts,lb8ts,lb9ts,lb10ts,lb12ts,lb15ts,lb16ts,lb17ts,lb20ts,lb21ts,lb23ts,lb25ts]
dfYtest = pd.concat(dfYtest1)
print dfYtest.shape


X_train = dfXtrain.as_matrix()
print X_train.shape
print(type(X_train))

X_test = dfXtest.as_matrix()
print X_test.shape
print(type(X_test))

Y_train = dfYtrain.as_matrix()
Y_train = Y_train.reshape((Y_train.shape[0],1))
print Y_train.shape
print(type(Y_train))

Y_test = dfYtest.as_matrix()
Y_test = Y_test.reshape((Y_test.shape[0],1))
print Y_test.shape
print(type(Y_test))

X_train = normalize(X_train,norm='l1')
X_test = normalize(X_test,norm='l1')

Y_train = np.log(Y_train+np.ones((Y_train.shape[0],1)))
Y_test = np.log(Y_test+np.ones((Y_test.shape[0],1)))
#np.save('YtestACTION.npy',Y_test)



#lsC = [0.01,0.1,1,10,100,1000]

#for C in lsC:
print 'SVR'
#print C


modelHLvisual= SVR(C=1000)
modelHLvisual.fit(X_train,Y_train)
predictionsHL = modelHLvisual.predict(X_test)
np.save('YpredACTIONvs.npy',predictionsHL)
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

model5 = MLPRegressor(hidden_layer_sizes=(1000,))
model5.fit(X_train,Y_train)

preds5 = model5.predict(X_test)
print preds5.shape

spearmanscoeff5 = scipy.stats.spearmanr(Y_test,preds5)
print spearmanscoeff5
