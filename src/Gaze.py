'''
Created on 2013-12-15

@author: yfeng
'''
import numpy as np
import scipy.stats as stat
class Gaze(object):
    '''
    GazeDataStore Class
    gaze x-y
    mouse x-y
    user
    click status 
    notes
    '''

    def __init__(self,data):
        '''
        store data into class
        '''
        data=data.T
        self.time=np.array(data[:][0],dtype='float32').T
        self.gaze=np.array(data[:][[1,2]],dtype='float64').T
        self.mouse=np.array(data[:][[3,4]],dtype='float64').T
        self.isclick=np.array(data[:][5],dtype='float16').T
        self.tag=data[7,0]
        del data
        self.corr=self.correction()
        self.corr[np.isnan(self.corr)]=0
        return
    
        
    def correction(self):
        temp=np.hstack((self.gaze,self.mouse)) 
        correctV=np.array([stat.pearsonr(temp[:,i], temp[:,j])[0] for i in range(0,4) for j in range(0,4) if(i>j)],dtype='float64')
        return correctV
        