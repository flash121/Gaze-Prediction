'''
Created on 2013-12-18

@author: yfeng
'''

from GazeFit import GazeFit
from sklearn import cross_validation
import numpy as np

class CVTest(object):
    '''
    @note: the class for Cross Validation
    @note: extend from scikit-learn CV class 
    @note: the main process for cv training
    '''

    def __init__(self,data=None,mode='SVR',per=0.8,N=10,rand_stage=0,options):
        '''
        Constructor
        @param data: 1xn gaze objects
        @param mode: mode for [LR][SVR]
        @param per: percent for splitting 
        '''
        self.options=options
        self.rand_stage=rand_stage
        self.data=data
        self.mode=mode
        self.size=per
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
            if len(self.scoring)<self.N:
                val=self.CVscoring()
                self.scoring.append(val)
                yield val
            else:
                self.flag=False
        else:
            for x in self.scoring:
                yield x
    def __str__(self):
        '''
        @note: show status
        '''
        return "CV Configuration: Number of Train: %d, isComplete %d, CVsize %d \n Scores: "+self.scoring.__str__() % (self.N,self.flag,self.size)
    def __getitem__(self,k):
        try:
            return self.scoring[k]
        except:
            return None    
    def CVscoring(self):
        '''
        @note: main CV processing
        '''
        X_train,X_test,Y_train,Y_test=cross_validation.train_test_split(self.data,self.data, test_size=self.size, random_state=self.rand_stage)
        del Y_train
        del Y_test
        model=GazeFit(self.option.order,self.options.C,self.options.eps,X_train,self.options.nclus)
        model.fit()
        X_Gaze=combine(X_test)
        scores=np.array([model.scoring(data) for data in X_Gaze])
        return np.average(scores)
    
    def clean(self):
        self.flag=True
    def status(self):
        '''
        @note: return summarizing variable
        '''
        pass
    
class Options(object):
    def __init__(self,order=None,C=None,eps=None,nclus=None):
        self.order=order
        self.C=C
        self.eps=eps
        self.nclus=nclus
    def rw(self,key,val):
        '''
        @note: rewrite func, for rewirite special val
        '''
        if key == 'o':
            self.order=val
        else:
            if key == 'C':
                self.C=val
            else:
                if key == 'e':
                    self.eps=val
                else:
                    self.nclus=val
                    
def combine(gazes):
    gaze=gazes[0]
    i=0
    for ga in gazes:
        if i==0:
            i+=1
            continue
        gaze.stack(ga)
    return gaze