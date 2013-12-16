'''
Created on 2013-12-15

@author: yfeng
'''
#from sklearn.svm import SVR
from sklearn.linear_model  import LogisticRegression
import  statsmodels.tsa.arima_model as model
import numpy as np

class Cluster(object):
    '''
    @param corrlection center:
    @param data index: 
    @param ARMA parameter:  
    @param SVR parameter:   
    '''


    def __init__(self,model=None,data=None,order=(1,0),C=1.0,eps=0.2,nclus=10):
        '''
        Store cluster information
        '''
        self.kmode=model
        self.order=order
        self.svrx=LogisticRegression(C=C,max_iter=1000)
        self.svry=LogisticRegression(C=C,max_iter=1000)
        self.gaze=np.array(data)
        self.nclus=nclus
        self.svrparam=[]
    def fit(self):
        label=np.array(self.kmode.labels_)
        i=0
        for l in range(0,self.nclus):
            feat=(np.array([0,0,0,0,0,0,0,0,0,0]),np.array([0]),np.array([0,0,0,0,0,0,0,0,0,0]),np.array([0]))
            index=label==l
            labgaze=self.gaze[index]
            for gaze in labgaze:
                temp=self.organziation(gaze)
                feat=(np.vstack((feat[0],temp[0])),np.vstack((feat[1],temp[1])),np.vstack((feat[2],temp[2])),np.vstack((feat[3],temp[3])))
                print i
                i+=1
            self.SVR(feat)                
            
    def organziation(self,gaze):
        gaX=gaze.gaze[:,0]
        gaY=gaze.gaze[:,1]
        moX=gaze.mouse[:,0]
        moY=gaze.mouse[:,1]
        l=moX.shape[0]
        startX=moX[0:10]
        startY=gaX[9]
        startX2=moY[0:10]
        startY2=gaY[9]
        for i in range(1,l-10):
            startX=np.vstack((startX,moX[i:i+10]))
            startY=np.vstack((startY,gaX[i+9]))
            startX2=np.vstack((startX2,moY[i:i+10]))
            startY2=np.vstack((startY2,gaY[i+9]))
        return (startX,startY,startX2,startY2)
    
    def SVR(self,temp):
        self.svrx.fit(temp[0], temp[1].T[0])
        self.svry.fit(temp[2], temp[3].T[0])
        self.svrparam.append((self.svrx,self.svry))
    def ARMA(self,data):
        arm=model.ARMA(data)
        result=arm.fit(order=self.order)
        return (result.arparams,result.maparams)
