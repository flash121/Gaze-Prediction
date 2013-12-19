'''
Created on 2013-12-15

@author: yfeng
'''
from sklearn.svm import SVR
import numpy as np

class GazeFit(object):
    '''
    @param corrlection center:
    @param data index: 
    @param ARMA parameter:  
    @param SVR parameter:   
    '''

    def __init__(self,model=None,data=None,order=10,C=1.0,eps=0.2,nclus=10):
        '''
        @note: Store cluster information
        '''
        self.kmode=model
        self.order=order
        self.gaze=np.array(data)
        self.nclus=nclus
        self.svrparam=[]
        self.C=C
        self.eps=eps
        self.armaparam=[]
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
   
    def SVR(self,ga):
        '''
        @note: SVR train processing
        '''
        svrx=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
        svry=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
        svrx.fit(ga.SVRX[0], ga.SVRX[1])
        svry.fit(ga.SVRY[0], ga.SVRY[1])
        self.svrparam.append((svrx,svry))
    def AR(self,ga):
        '''
        @note: AR train processing
        '''
        svrx=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
        svry=SVR(C=self.C,epsilon=self.eps,max_iter=1000)
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
    