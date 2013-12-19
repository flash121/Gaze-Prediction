'''
Created on 2013-12-15

@author: yfeng
'''
import numpy as np
from Gaze import Gaze
from GazeFit import GazeFit
from CVSet import CVTest 
from CVSet import Options

class GPDriver(object):
    '''
    Reading raw data.txt content, store into Gaze Class List
    '''

    def __init__(self,name='data1.txt'):
        '''
        reading process
        '''
        f=open(name,'r')
        data=[line.split() for line in f]
        #dive into each sample
        l=len(data)
        termin=[x-1 for x in range(1,l) if data[x][7]!=data[x-1][7]]
        ndata=np.array(data) #using np array to doing vector operation
        del data #release data
        termin.insert(0, -1)
        termin.append(l-1)
        self.gaze=[]
        i=0
        for ind in range(0,len(termin)-1):
            if(termin[ind+1]-termin[ind]-1>=20):
                if(i==100):
                    break
                print str(termin[ind]+1)+' '+str(termin[ind+1])
                self.gaze.append(Gaze(ndata[termin[ind]+1:termin[ind+1],:]))
                print (i,self.gaze[i].corr)
                i+=1


def Selectloop(opt,data):
    '''
    @note: feature selection for training, model params:
    @param order: order of AR & SVR Prediction
    @param C: SVR model parameter
    @param eps: SVR model paramter
    @param nclus: number of clustering for separable training
    @param data: main training data
    @attention: input is Options class
    @return: best feature set
    @note: most simple version: muti-layer feature selection
    
    @author: yfeng
    @version: 0.01
    '''
    #first - layer: order
    cv=CVTest(data=data, options=opt)
    mval=np.mean(np.array([score for score in cv]))
    bestorder=1
    for i in range(2,11):
        opt.rw('o', i)
        cv=CVTest(data=data, options=opt)
        val=np.mean(np.array([score for score in cv]))
        if mval>val:
            mval=val
            bestorder=i
    opt.rw('o', bestorder)
    #second - layer: C
    bestC=opt.C
    for i in range(0.1,0.5,10):
        opt.rw('C', i)
        cv=CVTest(data=data, options=opt)
        val=np.mean(np.array([score for score in cv]))
        if mval>val:
            mval=val
            bestC=i
    opt.rw('C', bestC)    
    #third layer
    beste=opt.eps
    for i in range(0.1,0.1,1):
        opt.rw('e', i)
        cv=CVTest(data=data, options=opt)
        val=np.mean(np.array([score for score in cv]))
        if mval>val:
            mval=val
            bestC=i
    opt.rw('e', beste)    
    #forth layer
    bestn=opt.nclus
    for i in range(2,1,10):
        opt.rw('n', i)
        cv=CVTest(data=data, options=opt)
        val=np.mean(np.array([score for score in cv]))
        if mval>val:
            mval=val
            bestC=i
    opt.rw('n', bestn)       
    
    return opt

def FS(opt,data):
    opt=Selectloop(opt,data)
    while 1:
        optN=Selectloop(opt,data)
        if(optN == opt):
            break
        else:
            opt=optN
    return opt
        
        


opt=Options(order=10,C=1.0,eps=0.2,nclus=3)
u=GPDriver()
model=GazeFit(order=10,C=1.0,eps=0.2,data=u.gaze,nclus=3)
model.fit()
model.predict(u.gaze[30])