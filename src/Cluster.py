'''
Created on 2013-12-15

@author: yfeng
'''
from sklearn.svm import SVR
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
        self.svrx=SVR(C=C,epsilon=eps,max_iter=1000)
        self.svry=SVR(C=C,epsilon=eps,max_iter=1000)
        self.gaze=np.array(data)
        self.nclus=nclus
        self.svrparam=[]
        self.armaparam=[]
    def fit(self):
        label=np.array(self.kmode.labels_)
        i=0
        for l in range(0,self.nclus):
            index=label==l
            labgaze=self.gaze[index]
            par1=(np.zeros(self.order[0]),np.zeros(self.order[1]))
            par2=(np.zeros(self.order[0]),np.zeros(self.order[1]))
            feat=(np.array([0,0,0,0,0,0,0,0,0,0]),np.array([0]),np.array([0,0,0,0,0,0,0,0,0,0]),np.array([0]))
            for gaze in labgaze:
                par=self.ARMA(gaze.mouse[:,0])
                par1=(par1[0]+par[0],par1[1]+par1[1])
                par=self.ARMA(gaze.mouse[:,1])
                par2=(par2[0]+par[0],par2[1]+par2[1])
                temp=self.organziation(gaze)
                feat=(np.vstack((feat[0],temp[0])),np.vstack((feat[1],temp[1])),np.vstack((feat[2],temp[2])),np.vstack((feat[3],temp[3])))
                print i
                i+=1
            self.SVR(feat)                
            par1=(par1[0]/gaze.shape[0],par1[1]/gaze.shape[0])
            par2=(par2[0]/gaze.shape[0],par2[1]/gaze.shape[0])
            self.armaparam.append((par1,par2))

            
            
    def organziation(self,gaze,order):
        gaX=gaze.gaze[:,0]/1920.0
        gaY=gaze.gaze[:,1]/1280.0
        moX=gaze.mouse[:,0]/1920.0
        moY=gaze.mouse[:,1]/1280.0
        l=moX.shape[0]
        startX=moX[0:order]
        startY=gaX[order-1]
        startX2=moY[0:order]
        startY2=gaY[order-1]
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
