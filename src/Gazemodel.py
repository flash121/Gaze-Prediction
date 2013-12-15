'''
Created on 2013-12-15

@author: yfeng
'''
from sklearn import svm

class GazeModel(object):
    '''
    Main Gaze Prediction model
    '''

    def __init__(self,C=1.0,eps=0.2,func='rbf',data=None):
        '''
        SVR model for prediction Gaze from Mouse
        '''
        