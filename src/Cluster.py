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
        self.svrx=SVR(C=C,epsilon=eps,max_iter=1000)
        self.svry=SVR(C=C,epsilon=eps,max_iter=1000)
        self.gaze=np.array(data)
        self.nclus=nclus
        self.svrparam=[]
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
        self.svrx.fit(ga.SVRX[0], ga.SVRX[1])
        self.svry.fit(ga.SVRY[0], ga.SVRY[1])
        self.svrparam.append((self.svrx,self.svry))
    def AR(self,ga):
        '''
        @note: AR train processing
        '''
        self.svrx.fit(ga.ARX[0], ga.ARX[1])
        self.svry.fit(ga.ARY[0], ga.ARY[1])
        self.armaparam.append((self.svrx,self.svry))
    def predict(self,x):
        '''
        @note: predict function
        @note: basic idea;
               1. classify cluster by AR-paramters, choose best number 
               2. SVR model regression -> gaze position (for x,y)
        @param x: 1xorder+1 time series
        '''
        pass
        
    '''
    @note: older part/plan to delete
    def ARMA(self,data):
        arm=model.ARMA(data)
        result=arm.fit(order=self.order)
        return (result.arparams,result.maparams)
    '''
    