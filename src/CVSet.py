'''
Created on 2013-12-18

@author: yfeng
'''
import Gaze
import GazeFit
from sklearn import cross_validation

class CVTest(object):
    '''
    @note: the class for Cross Validation
    @note: extend from scikit-learn CV class 
    @note: the main process for cv training
    '''


    def __init__(self,data=None,mode='SVR',per=0.8,N=10):
        '''
        Constructor
        @param data: 1xn gaze objects
        @param mode: mode for [LR][SVR]
        @param per: percent for splitting 
        '''
        self.data=data
        self.mode=mode
        self.per=per
        self.N=N
        self.flag=True
        self.scoring=[]
    def __iter__(self):
        '''
        @note: each iter -> one time cross_validation
        @note: usage:
                        [result]=[score for [object] in [CVTest]]                
        '''
        if self.flag:
            pass
        else:
            pass
    def __str__(self):
        '''
        @note: show status
        '''
        pass
    def __getitem__(self,k):
        try:
            return self.scoring[k]
        except:
            return None    
    def CVscoring(self):
        '''
        @note: main CV processing
        '''
        pass
    def clean(self):
        self.flag=True
    def status(self):
        '''
        @note: return summarizing variable
        '''
        pass