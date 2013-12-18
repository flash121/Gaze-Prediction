'''
Created on 2013-12-15

@author: yfeng
'''
from sklearn.svm import SVR
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
        self.svrx=SVR(C=C,epsilon=eps,max_iter=3000)
        self.svry=SVR(C=C,epsilon=eps,max_iter=3000)
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
            feat=(np.array([0,0,0,0,0,0,0,0,0,0]),np.array([0]),np.array([0,0,0,0,0,0,0,0,0,0]),np.array([0]))
            featARMA=(np.array([0,0]),np.array([0]),np.array([0,0]),np.array([0]))
            for gaze in labgaze:
                temp=self.organziation(gaze,10)
                feat=(np.vstack((feat[0],temp[0])),np.vstack((feat[1],temp[1])),np.vstack((feat[2],temp[2])),np.vstack((feat[3],temp[3])))
                temp=self.organziation2(gaze,2)
                featARMA=(np.vstack((featARMA[0],temp[0])),np.vstack((featARMA[1],temp[1])),np.vstack((featARMA[2],temp[2])),np.vstack((featARMA[3],temp[3])))   
                print i
                i+=1
            self.SVR(feat)                
            self.AR(featARMA)

            
            
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
        for i in range(1,l-order):
            startX=np.vstack((startX,moX[i:i+order]))
            startY=np.vstack((startY,gaX[i+order-1]))
            startX2=np.vstack((startX2,moY[i:i+order]))
            startY2=np.vstack((startY2,gaY[i+order-1]))
        return (startX,startY,startX2,startY2)
    
    def organziation2(self,gaze,order):
        gaX=gaze.gaze[:,0]/1920.0
        gaY=gaze.gaze[:,1]/1280.0
        moX=gaze.mouse[:,0]/1920.0
        moY=gaze.mouse[:,1]/1280.0
        l=moX.shape[0]
        startX=moX[0:order]
        startY=moX[order]
        startX2=moY[0:order]
        startY2=moY[order]
        for i in range(1,l-order-1):
            startX=np.vstack((startX,moX[i:i+order]))
            startY=np.vstack((startY,gaX[i+order]))
            startX2=np.vstack((startX2,moY[i:i+order]))
            startY2=np.vstack((startY2,gaY[i+order]))
        return (startX,startY,startX2,startY2)
    
    def SVR(self,temp):
        self.svrx.fit(temp[0], temp[1].T[0])
        self.svry.fit(temp[2], temp[3].T[0])
        self.svrparam.append((self.svrx,self.svry))
    def AR(self,temp):
        self.svrx.fit(temp[0], temp[1].T[0])
        self.svry.fit(temp[2], temp[3].T[0])
        self.armaparam.append((self.svrx,self.svry))
    '''
    def ARMA(self,data):
        arm=model.ARMA(data)
        result=arm.fit(order=self.order)
        return (result.arparams,result.maparams)
    '''