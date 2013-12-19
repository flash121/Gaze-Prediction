'''
Created on 2013-12-18

@author: yfeng
'''

class CVTest(object):
    '''
    @note: the class for Cross Validation
    @note: extend from scikit-learn CV class 
    @note: the main process for cv training
    '''


    def __init__(self,data=None,mode='SVR',per=0.8):
        '''
        Constructor
        @param data: 1xn gaze objects
        @param mode: mode for [LR][SVR]
        @param per: percent for splitting 
        '''
        self.data=data
        self.mode=mode
        self.per=per