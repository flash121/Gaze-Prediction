'''
Created on 2013-12-15

@author: yfeng
'''
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
import numpy as np
import sklearn.cluster as cl

class GazeFit(object):
    '''
    @param corrlection center:
    @param data index: 
    @param ARMA parameter:  
    @param SVR parameter:   
    @param mode: mode switch param, provide [LR][SVR]
    @version: 0.01
    @author: yfeng
    '''

    def __init__(self,data=None,order=10,C=1.0,eps=0.2,nclus=10,mode='SVR'):
        '''
        @note: Store cluster information
        '''
        self.order=order
        self.gaze=np.array(data)
        self.nclus=nclus
        self.svrparam=[]
        self.C=C
        self.eps=eps
        self.mode=mode
        self.armaparam=[]
        self.kmode=self.kmeans(data, nclus)
    def fit(self):
        '''
        @note: Overall fit process
        '''
        label=np.array(self.kmode.labels_)
        i=0
        for l in range(0,self.nclus):
            index=label==l
            labgaze=self.gaze[index]
            ga=labgaze[0]           
            for gaze in labgaze:
                if(i==0):
                    i+=1
                    continue
                ga.stack(gaze)
                print i
                i+=1
            self.SVR(ga)                
            self.AR(ga)
    def kmeans(self,gazeset,n):
        '''
        @note: implement Kmeans for clustering
        '''
        data=np.array([gadata.corr for gadata in gazeset],dtype='float64')
        model=cl.KMeans(n_clusters=n, init='k-means++', n_init=15, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1, k=None)
        model.fit(data)
        return model
    def SVR(self,ga):
        '''
        @note: SVR train processing
        '''
        if(self.mode=='SVR'):
            svrx=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
            svry=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
        else:
            svrx=LinearRegression()
            svry=LinearRegression()
        svrx.fit(ga.SVRX[0], ga.SVRX[1])
        svry.fit(ga.SVRY[0], ga.SVRY[1])
        self.svrparam.append((svrx,svry))
    def AR(self,ga):
        '''
        @note: AR train processing
        '''
        if(self.mode=='SVR'):
            svrx=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
            svry=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
        else:
            svrx=LinearRegression()
            svry=LinearRegression()        
        svrx.fit(ga.ARX[0], ga.ARX[1])
        svry.fit(ga.ARY[0], ga.ARY[1])
        self.armaparam.append((svrx,svry))
    def predict(self,chunk):
        '''
        @note: predict function
        @note: basic idea;
               1. classify cluster by AR-paramters, choose best number 
               2. SVR model regression -> gaze position (for x,y)
        @param x: 1xorder+1 time series
        '''
        mo=[self.measure(model[0].predict(chunk.ARX[0]),chunk.ARX[1])+self.measure(model[0].predict(chunk.ARY[0]),chunk.ARY[1]) for model in self.armaparam]
        hand=np.argmax(np.array(mo))
        svr=self.svrparam[hand]
        x=np.array(svr[0].predict(chunk.SVRX[0]))
        y=np.array(svr[1].predict(chunk.SVRY[0]))
        return np.hstack((x,y))
    def scoring(self,chunk,target):
        '''
        @note similar to predict 
        '''
        mo=[self.measure(model[0].predict(chunk.ARX[0]),chunk.ARX[1])+self.measure(model[0].predict(chunk.ARY[0]),chunk.ARY[1]) for model in self.armaparam]
        hand=np.argmax(np.array(mo))
        svr=self.svrparam[hand]
        x=np.array(svr[0].predict(chunk.SVRX[0]))
        y=np.array(svr[1].predict(chunk.SVRY[0]))
        return self.measure(x,target.SVRX[1])+self.measure(y,target.SVRY[1]) 
    def measure(self,x,y):
        '''
        @note: distance measure, current only norm-2 measure
        @note: use for measuring performance
        '''    
        return sum(abs(x-y))
        
    '''
    @note: older part/plan to delete
    def ARMA(self,data):
        arm=model.ARMA(data)
        result=arm.fit(order=self.order)
        return (result.arparams,result.maparams)
    '''
    