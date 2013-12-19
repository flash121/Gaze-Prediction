'''
Created on 2013-12-15

@author: yfeng
'''
import numpy as np
import scipy.stats as stat
class Gaze(object):
    '''
    Gaze Class:
    @note: this class [Gaze] for basic data item operations
    @note: now provide: [stack]
    @note: store-type:  [trainAR], [trainSVR], [raw data] 
    @note: other annotation only available before stack
    GazeDataStore Class
    @version: 0.01
    ------------------------------------------------
    @author:  yfeng
    '''

    def __init__(self,data,order=10):
        '''
        store data into class
        '''
        data=data.T
        self.time=np.array(data[:][0],dtype='float32').T
        self.gaze=np.array(data[:][[1,2]],dtype='float64').T
        self.standard(self.gaze)
        self.mouse=np.array(data[:][[3,4]],dtype='float64').T
        self.mouse=self.standard(self.mouse)
        self.isclick=np.array(data[:][5],dtype='float16').T
        self.tag=data[7,0]
        self.order=order
        del data
        self.corr=self.correction()
        self.corr[np.isnan(self.corr)]=0
        self.ConvertARtrain(self.mouse, order)
        self.ConvertSVRtrain(self.mouse, self.gaze, order)
        return
    
        
    def correction(self):
        temp=np.hstack((self.gaze,self.mouse)) 
        correctV=np.array([stat.pearsonr(temp[:,i], temp[:,j])[0] for i in range(0,4) for j in range(0,4) if(i>j)],dtype='float64')
        return correctV
    def ConvertARtrain(self,label,order):
        '''
        AR train version
        @param  label: label <-> numpy
        @param order: order for learning
        '''
        labelX=label[:,0]
        labelY=label[:,1]
        l=label.shape[0]
        XX=labelX[0:order]
        XY=labelX[order]
        YX=labelY[0:order]
        YY=labelY[order]
        for i in range(1,l-order-1):
            XX=np.vstack((XX,labelX[i:i+order]))
            XY=np.vstack((XY,labelX[i+order]))
            YX=np.vstack((YX,labelY[i:i+order]))
            YY=np.vstack((YY,labelY[i+order]))
        self.ARX=(XX,XY.T[0])
        self.ARY=(YX,YY.T[0])
    def ConvertSVRtrain(self,label,target,order):
        '''
        SVR train version
        @param label: label <> numpy, for mouse
        @param order: order for learning
        @param target: gaze position  
        '''
        
        labelX=label[:,0]
        labelY=label[:,1]
        targetX=target[:,0]
        targetY=target[:,1]
        l=label.shape[0]
        XX=labelX[0:order]
        XY=targetX[order-1]
        YX=labelY[0:order]
        YY=targetY[order-1]
        for i in range(1,l-order-1):
            XX=np.vstack((XX,labelX[i:i+order]))
            XY=np.vstack((XY,targetX[i+order-1]))
            YX=np.vstack((YX,labelY[i:i+order]))
            YY=np.vstack((YY,targetY[i+order-1]))
        self.SVRX=(XX,XY.T[0])
        self.SVRY=(YX,YY.T[0])            
    def stack(self,stack):
        '''
        gaze.stack(gaze2)
        stack two object
        Only combined SVRX/Y & ARX/Y will combined, other remains, 
        for training process
        '''
        self.ARX=(np.vstack((self.ARX[0],stack.ARX[0])),np.hstack((self.ARX[1],stack.ARX[1])))
        self.ARY=(np.vstack((self.ARY[0],stack.ARY[0])),np.hstack((self.ARY[1],stack.ARY[1])))
        self.SVRX=(np.vstack((self.SVRX[0],stack.SVRX[0])),np.hstack((self.SVRX[1],stack.SVRX[1])))
        self.SVRY=(np.vstack((self.SVRY[0],stack.SVRY[0])),np.hstack((self.SVRY[1],stack.SVRY[1])))
    def standard(self,item,s=(1920,1280)):
        '''
        standard measure
        '''
        item[:,0]=item[:,0]/s[0]
        item[:,1]=item[:,1]/s[1]
        return item
    def __iter__(self):
        '''
        @note: iteration schema: 
        @note: each iteration return first n-vector AR,SVR
        '''
        for n in range(0,self.ARX[0].shape[0]):
            yield GazeIter(self,n)
    def __getitem__(self,k):
        '''
        @attention: same as __iter__
        '''
        return GazeIter(self,k)
    def __str__(self):
        '''
        @note: print information about: Len, Order
        '''
        return "AR shape: (%d,%d), order: %d" % (self.ARX[0].shape[0],self.ARX[0].shape[1],self.order)
class GazeIter(object):
    '''
    @note: a simple object for iter Gaze - AR, SVR value
    @note: for prediction simulate
    @note: only init func
    '''
    def __init__(self,gaze,n=1):
        self.n=n
        self.ARX=(gaze.ARX[0][0:n,:],gaze.ARX[1][0:n])
        self.ARY=(gaze.ARY[0][0:n,:],gaze.ARY[1][0:n])
        self.SVRX=(gaze.SVRX[0][0:n,:],gaze.SVRX[1][0:n])
        self.SVRY=(gaze.SVRY[0][0:n,:],gaze.SVRY[1][0:n])
    def __str__(self):
        return "Gaze Iteration, n= %d" % (self.n)
    