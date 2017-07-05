import numpy as np

import scipy.stats



print 'pool action'

predictionsHLa = np.load('/home/11394153/Regression/YpredACTIONhl_MLP.npy')

predictionsLLa = np.load('/home/11394153/Regression/YpredACTIONll_MLP.npy')

predictionsVSa = np.load('/home/11394153/Regression_vsent/YpredACTIONvs_MLP.npy')

act = np.vstack((predictionsHLa,predictionsLLa,predictionsVSa))

print act.shape

meanact = np.amax(act, axis=0)

print meanact.shape

Ytestact = np.load('/home/11394153/Regression/YtestACTION.npy')

print Ytestact.shape


spearmanscoeff1 = scipy.stats.spearmanr(Ytestact,meanact)
print spearmanscoeff1

print 'pool scene'

predictionsHLs = np.load('/home/11394153/Regression/YpredSCENEhl_MLP.npy')

predictionsLLs = np.load('/home/11394153/Regression/YpredSCENESll_MLP.npy')

predictionsVSs = np.load('/home/11394153/Regression_vsent/YpredSCENEvs_MLP.npy')

sc = np.vstack((predictionsHLs,predictionsLLs,predictionsVSs))

print sc.shape

meansc = np.amax(sc, axis=0)

print meansc.shape

Ytestsc = np.load('/home/11394153/Regression/YtestSCENE.npy')

print Ytestsc.shape



spearmanscoeff2 = scipy.stats.spearmanr(Ytestsc,meansc)
print spearmanscoeff2

print "pp"

predictionsHLpp = np.load('/home/11394153/Regression/YpredPEOPETShl_MLP.npy')

predictionsLLpp = np.load('/home/11394153/Regression/YpredPEOPETSll_MLP.npy')

predictionsVSpp = np.load('/home/11394153/Regression_vsent/YpredPEOPETSvs_MLP.npy')

pp = np.vstack((predictionsHLpp,predictionsLLpp,predictionsVSpp))

print pp.shape

meanpp = np.amax(pp, axis=0)

print meanpp.shape

Ytestpp = np.load('/home/11394153/Regression/YtestPEOPETS.npy')

print Ytestpp.shape


spearmanscoeff3 = scipy.stats.spearmanr(Ytestpp,meanpp)
print spearmanscoeff3

print 'wendys'


predictionsHLw = np.load('/home/11394153/Regression/YpredWENDYShl_MLP.npy')

predictionsLLw = np.load('/home/11394153/Regression/YpredWENDYSll_MLP.npy')

predictionsVSw = np.load('/home/11394153/Regression_vsent/YpredWENDYSvs_MLP.npy')

print predictionsHLw.shape, predictionsLLw.shape, predictionsVSw.shape

w = np.vstack((predictionsHLw,predictionsLLw,predictionsVSw))

print w.shape

meanw = np.amax(w, axis=0)

print meanw.shape

Ytestw = np.load('/home/11394153/Regression/YtestWENDYS.npy')

print Ytestw.shape


spearmanscoeff4 = scipy.stats.spearmanr(Ytestw,meanw)
print spearmanscoeff4