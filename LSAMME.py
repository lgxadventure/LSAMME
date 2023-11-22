import os
import copy
import math
import time
from turtle import shape
import utils
from re import L
import numpy as np
from scipy.special import xlogy

from sklearn.tree import DecisionTreeClassifier
from numpy.lib.nanfunctions import _nanmedian_small
from sklearn.ensemble import AdaBoostClassifier
import pickle
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
class Algorithmerror(RuntimeError):
    """ Runtime error throwed out when the specified Adaboost algorithm does not fit the data format"""
    def __init__(self, n_classes, algorithm):
        self.n_classes = n_classes
        self.algorithm = algorithm
    def __str__(self):
        print("algorithm %s does not match the dataset with %d classes"%(self.algorithm, self.n_classes))


class AdaboostClassifier:
    """ An Adaboost classifier supporting basic Adaboost, Adaboost.M2, Adaboost.MH, SAMME, SAMME.R algorithms.
        Also, this classifier can be optimized by alpha-improving method
    Parameters : 
    ------------
    base_estimator : classifier object ; default=None
        Base classifier from which the boosted ensemble is built.
        the default base_estimator is sklearn.tree.DecisionTreeClassifier(max_depth=1)

    n_estimators : int ; default=50
        The number of estimators at which boosting is terminated.

    learning_rate : float ; default=1.0
        Weight applied to each classifier at each boosting iteration.

    algorithm : string ; default=None
        Specified Adaboost algorithm.
        the default algorithm is SAMME algorithm
    
    random_state : int/RandomState/None ; default=None
        Controls the random seed given at each `base_estimator` at each boosting iteration.

    Attributes :
    ------------
    base_estimator : estimator object
        Instance of base_estimator
    estimator_list_ : list
        List of base classifiers
    estimator_weights_ : ndarray of float
        Weights for each estimator in the boosted ensemble.
    estimator_errors_ : ndarray of float
        error rate for each estimator in the boosted ensemble.
    
    classes_ : ndarray of shape (n_classes,)
        The classes labels set
    n_estimators : int
        Number of estimators
    n_classes_ : int
        Number of classes in Labels
    attr_names_ : ndarray of shape(n_attrs_,)
        Names of features seen during fitting
    n_attrs_ : int
        Number of attributes in input vectors.
    """

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0,algorithm='SAMME', random_state=None ,ckl=None):
        """ Initialize the Adaboost Framework
        """

        self.base_estimator = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.random_state = random_state

        self.ckl= ckl

    def fit(self, X, Y, sample_weight=None):
        """ fit the Adaboost model with training set (X, Y).
        Parameters
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input traing samples.
        
        Y : array-like of shape (n_samples,)
            The target values (class labels).
        
        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. Initialized to 1/n_samples when sample_weight None.

        Returens
        --------
        self : object
            Fitted Model
        """

        # Check the specified algorithm
        if self.algorithm not in ('Basic', 'SAMME', 'SAMME.R', 'M2', 'MH'):
            raise ValueError("Algorithm must in ['Basic', 'SAMME', 'SAMME.R', 'M2', 'MH']."
            f" Got {self.algorithm!r} instead.")
        
        # initialize the weight matrix of samples
        n_samples = X.shape[0]
        n_classes = len(set(Y))
        sample_weight = self._init_sample_weight(n_samples, n_classes, Y)
        self.init_sampleweight = sample_weight

        # initialize the estimator
        self._init_estimator(default=DecisionTreeClassifier(max_depth=2))

        self.estimator_list_ = []
        self.estimator_weights_ = np.zeros(self.n_estimators_, dtype=np.float64)
        self.estimator_errors_ = np.ones(self.n_estimators_, dtype=np.float64)

        # Initialize random numbers
        random_state = utils._check_random_state(self.random_state)

        for iboost in range(self.n_estimators_):
            sample_weight, estimator_weight, estimator_error = self._boost(
                iboost, X, Y, sample_weight, random_state
            )
            # Early termination
            if sample_weight is None:
                break

            self.estimator_weights_[iboost] = estimator_weight
            self.estimator_errors_[iboost] = estimator_error

            if estimator_error == 0:
                break
            
            sample_weight_sum = np.sum(sample_weight)

            if sample_weight_sum <= 0:
                break

            if iboost < self.n_estimators_-1:
                sample_weight /= sample_weight_sum
        #self.estimator_weights_ = self.estimator_weights_/np.sum(self.estimator_weights_)
        self.scale = sum(self.estimator_weights_)
        return self


    def _boost(self, iboost, X, Y, sample_weight, random_state):
        """ The iboost step of fitting. Choose proper method by specified algorithm name.
        Parameters
        -----------
        iboost : int
            current step(train the iboost'th estimator in this function)

        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Input traing samples.
        
        Y : array-like of shape (n_samples,)
            The target values (class labels).

        sample_weight : list or array of shape (n_samples,)
            The weight of samples trained in current estimator.
            Different training algorithms has different types of weights, generated by function _distrib_cal_().
        
        random_state : int/RandomState/None ; default=None
            Controls the random seed given at each `base_estimator` at each boosting iteration.

        Returens
        --------
        sample_weight : list or array  of shape (n_samples,) or (n_samples, n_classes)
            The weight of samples used in base estimator learning.
            Its shape depends on the training algorithm.
        
        estimator_weight : list or array  of shape (n_estimators,)
            The weight of samples to generate base estimators to complete additive model.
            It depens on the accuracy of generated base estimators till the iboost'th step.
        
        estimator_error : list or array  of shape (n_estimators,)
            Error rate of base estimator trained in the iboost'th step.
        """
        
        D, Q = self._distrib_cal_(sample_weight)
        
        if self.algorithm == "Basic":
            return self._boost_Basic(iboost, X, Y, D, random_state)
        elif self.algorithm == "SAMME":
            return self._boost_SAMME(iboost, X, Y, D, random_state)
        elif self.algorithm == "SAMME.R":
            return self._boost_SAMME_R(iboost, X, Y, D, random_state)
        elif self.algorithm == "M2":
            return self._boost_M2(iboost, X, Y, D, Q, random_state)
        elif self.algorithm == "MH":
            return self._boost_MH(iboost, X, Y, D, Q, random_state)

    def _boost_Basic(self, iboost, X, Y, sample_weight, random_state):
        """ Implement a single boost using original Adaboost algorithm, only for two-class classification problems.
        Target label must in {+1, -1} 
        """
        estimator = self._make_estimator(random_state=random_state)

        # results of prediction (labels) come from {+1, -1}
        estimator.fit(X, Y, sample_weight)
        y_predict = estimator.predict(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)
        
        if (not self.n_classes_ == 2) or (1 not in self.classes_) or (-1 not in self.classes_):
            raise Algorithmerror(self.n_classes_, self.algorithm)
        
        estimator_error = np.average(y_predict!=Y, weights=sample_weight)

        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        
        if estimator_error >= 0.5:
            self.estimator_list_.pop(-1)
            if len(self.estimator_list_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit."
                )
            return None, None, None
        
        beta = estimator_error / (1-estimator_error)
        estimator_weight = 0.5*np.log(1/beta)

        _update = np.exp(-y_predict * Y * estimator_weight)

        sample_weight = sample_weight*_update
        sample_weight /= np.sum(sample_weight)

        return sample_weight, estimator_weight, estimator_error
    
    def _boost_SAMME_R(self, iboost, X, Y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm.
        """

        # initialize current estimator and fit it with input samples
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, Y, sample_weight=sample_weight)
        
        # predict predicted probability to all classes with estimator
        y_pred_prob = estimator.predict_proba(X)
        
        # convert the output to label and calculate the error rate
        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_pred_prob, axis=1), axis=0)
        incorrect = y_predict != Y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # stop when the ewstimator is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        # Construct y coding gt by predicted result
        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == Y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        proba = y_pred_prob
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME.R algorithm
        estimator_weight = (-1 * self.learning_rate * ((n_classes - 1.0)/n_classes) 
                            * xlogy(y_coding, y_pred_prob).sum(axis=1))

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators_ - 1:
            # Only boost positive weights
            sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)))

        return sample_weight, 1.0, estimator_error

    def _boost_SAMME(self, iboost, X, Y, sample_weight, random_state):
        """Implement a single boost using the SAMME discrete algorithm.
        """

        
        ckl=self.ckl
   
       

        # A=np.zeros(l)
        # B=np.zeros(l)
        # for i in range(0,n):
        #     for j in range(0,l):
        #     A[i]=
        #     B=

        # for i in range(0,n):
        #     sample_weight[i]=sample_weight[i]* (np.exp(-1/n_classes*A[i])-np.exp(-1/n_classes*B[i]))



        # initialize current estimator and fit it with input samples
        estimator = self._make_estimator(random_state=random_state)
          
        #这里训练


        estimator.fit(X, Y, sample_weight=sample_weight)

        # convert the output to label and calculate the error rate
        if iboost == 0:#计算类别
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        l=self.n_classes_
        n=len(X)
        
        #生成0，1的n*l的矩阵
        y_l= np.zeros((n, l), dtype=np.float32)
        for i in range(0,n):
           for j in range(0,l): 
               if Y[i]==j:
                    y_l[i,j]=1


        #真实标签生成0，1的n*l的矩阵，1，-/1-l
        y_k= np.zeros((n, l), dtype=np.float32)
        for i in range(0,n):
           for j in range(0,l): 
               if Y[i]==j:
                    y_k[i,j]=1
               else:
                    y_k[i,j]=-1/(l-1)


        y_pred = estimator.predict(X)

        #生成预测标签的0，1的n*l的矩阵，1，-/1-l
        y_k_p= np.zeros((n, l), dtype=np.float32)
        for i in range(0,n):
           for j in range(0,l): 
               if y_pred[i]==j:
                    y_k_p[i,j]=1
               else:
                    y_k_p[i,j]=-1/(l-1)


        incorrect = (y_pred != Y)
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # stop when the ewstimator is perfect，当错误率小于等于0时，返还
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_

        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimator_list_.pop(-1)
            if len(self.estimator_list_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit."
                )
            return None, None, None
        
        # Boost weight using multi-class AdaBoost SAMME alg,分类器权重
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )
        
        #estimator_weight = max(estimator_weight, np.finfo(np.float).eps)
        
        Beta=estimator_weight * np.sqrt(n_classes-1)/n_classes

        Beta=max(Beta, np.finfo(np.float).eps)

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators_ - 1 and  Beta> 0:
            # Only boost positive weights
            sample_weight1 = sample_weight * np.exp(Beta * incorrect * (sample_weight > 0))
            #sample_weight = sample_weight * np.exp(Beta * incorrect * (sample_weight > 0))
            for i in range(0,n):
                #sample_weight[i]=sample_weight[i]* np.exp(-1/n_classes*estimator_weight*np.sum(ckl[i]*y_k_p[i]*y_k[i]))

                sample_weight[i]=sample_weight[i]* np.exp((-1/n_classes)*Beta*np.sum(ckl[i]*y_k_p[i]*y_k[i]))
                
            #     #*np.exp((-1/n_classes))
            #sample_weight= sample_weight * np.exp(Beta * np.sum(np.dot(y_k_p,y_k)*ckl) * (sample_weight > 0))
            #for i in range(0,n):
            # print(ckl[i])
            # print(y_k_p[i])
            # print(y_k[i])
            # print(type(ckl[i]))
            # print(type(y_k_p[i]))
            # print(type(y_k[i]))
            # print(np.sum(ckl[i]*y_k_p[i]*y_k[i]))
        #return sample_weight, estimator_weight, estimator_error
        return sample_weight,Beta, estimator_error





    def _boost_M2(self, iboost, X, Y, sample_weight, incrroct_weight, random_state):
        """Implement a single boost using the Adaboost.M2 algorithm.
        """

        # initialize current estimator and fit it with input samples
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, Y, sample_weight=sample_weight)

        pred = estimator.predict(X)

        # convert the output to label and calculate the error rate
        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)
        n_samples = Y.shape[0]
        n_classes = self.n_classes_
        name2id = self.name2id

        h = np.zeros_like(incrroct_weight)
        y = np.zeros_like(Y)
        h_y = np.zeros_like(Y)

        for i in range(n_samples):
            h[i, name2id[pred[i]]] = 1
            y[i] = name2id[Y[i]]
            h_y[i] = h[i,y[i]]
        h_y = np.tile(h_y[:,np.newaxis], (1, n_classes)).astype(np.float)

        err = 0.5 * incrroct_weight * (np.ones_like(h) + h - h_y)
        estimator_error = np.sum(err)

        for i in range(n_samples):
            estimator_error -= err[i,y[i]]

        # stop when the ewstimator is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0
        
        if estimator_error >= 1.0 - (1.0 / n_classes):
            self.estimator_list_.pop(-1)
            if len(self.estimator_list_) == 0:
                raise ValueError(
                    "BaseClassifier in AdaBoostClassifier ensemble is worse than random, ensemble can not be fit."
                )
            return None, None, None

        # Boost weight using multi-class AdaBoost.M2 alg
        beta = estimator_error / (1-estimator_error)
        estimator_weight = np.log(1/beta)
        estimator_weight = max(estimator_weight, np.finfo(np.float).eps)
        estimator_weight *= self.learning_rate

        # Only boost the weights if I will fit again
        if not iboost == self.n_estimators_ - 1:
            rec_bool = incrroct_weight<=0
            rec_val = incrroct_weight[rec_bool]

            incrroct_weight = incrroct_weight * np.exp(0.5 * np.log(beta) * (np.zeros_like(h) + h_y - h))
            incrroct_weight[rec_bool] = rec_val
            incrroct_weight /= np.sum(incrroct_weight)

        self.sample_weight = sample_weight
        self.D_t = incrroct_weight

        return incrroct_weight, estimator_weight, estimator_error


    def _boost_MH(self, iboost, X, Y, sample_weight, D_t, random_state):
        """Implement a single boost using the Adaboost.MH algorithm.
        """

        Dt = D_t.copy()
        # initialize current estimator and fit it with input samples
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, Y, sample_weight=sample_weight)
        
        # convert the output to label and calculate the error rate
        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        n_samples = Y.shape[0]
        n_classes = self.n_classes_
        name2id = self.name2id

        pred = estimator.predict(X)

        h = np.zeros((n_samples, n_classes))
        y = np.zeros((n_samples, n_classes))
        y.fill(-1)
        h.fill(-1)

        for i in range(n_samples):
            h[i, name2id[pred[i]]] = 1
            y[i, name2id[Y[i]]] = 1
        
        y_h = y*h
        r = np.sum(Dt * y_h)
        estimator_error = 0.5*(1-r)

        alpha_t = 0.5 * math.log((1+r)/(1-r))
        alpha_t = max(alpha_t, np.finfo(np.float).eps)
        
        
        if not iboost == self.n_estimators_ - 1:
            Dt = Dt * np.exp(-alpha_t * y_h)
            Dt /= sum(Dt)
        #estimator_weight *= self.learning_rate

        self.sample_weight = sample_weight
        self.D_t = Dt

        return Dt, alpha_t, estimator_error

    def predict(self, X):
        """Predict classes on data matrix X
        """
        if self.algorithm == "Basic":
            pred = self.decision_function(X)
            pred[pred>0] = 1
            pred[pred<=0] = -1
            return pred
        else:
            pred = self.decision_function(X)
            return self.classes_.take(np.argmax(pred, axis=1))
    
    def predict_proba(self, X):
        """Predict probability of each classes on data matrix X
        """

        proba = self.decision_function(X, flg="proba")
        return proba

    def predict_stage(self, x):
        """ Output the predict result on each base estimator
        """
        if self.algorithm =="Basic":
            n_samples = x.shape[0]
            h = np.zeros((n_samples, self.n_estimators_))
            for t in range(self.n_estimators_):
                h[:, t] = self.estimator_list_[t].predict(x)
        elif self.algorithm == "SAMME.R":
            n_samples = x.shape[0]
            h = np.zeros((n_samples, self.n_classes_, self.n_estimators_))
            for t in range(self.n_estimators_):
                h[:, :, t] = self.estimator_list_[t].predict_proba(x)
        else:
            n_samples = x.shape[0]
            h = np.zeros((n_samples, self.n_classes_, self.n_estimators_))
            for t in range(self.n_estimators_):
                h[:, :, t] = self.estimator_list_[t].predict(x)
        
        return h

    def decision_function(self, X, flg=None):
        """ Add the predicted resulsts of each base estimator with estimator weights on dataset X.
        Parameters:
        -----------
        flg : string ; default=None
            can be None or 'proba', depending the output type training algorithm need.
        """

        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]

        if self.algorithm == "SAMME.R":
            pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimator_list_)
        elif self.algorithm == "Basic":
            thresh = int(np.log(X.shape[0]))
            pred = sum((estimator.predict(X)).T*alpha
                        for estimator, alpha in zip(self.estimator_list_, self.estimator_weights_)
                    )
            pred[pred>thresh]=thresh
            pred[pred<-thresh]=-thresh
        else:
            if flg == 'proba':
                pred_proba = sum((estimator.predict_proba(X))*alpha
                    for estimator, alpha in zip(self.estimator_list_, self.estimator_weights_)
                )
                pred_proba /= self.estimator_weights_.sum()
                return pred_proba
            else:
                pred = sum((estimator.predict(X) == classes).T*alpha
                    for estimator, alpha in zip(self.estimator_list_, self.estimator_weights_)
                )
                if self.estimator_weights_.sum()==0:
                    print(self.estimator_weights_)
                pred /= self.estimator_weights_.sum()
        return pred

    def score(self, X, Y, mod='acc'):
        """ Calculate the score of current model.
        Parameters:
        -----------
        mod : string in {'acc', 'loss'} ; default='acc'
            choose accuracy or loss as score of model.
            loss depends on specified training algorithm.
        """

        n_samples = X.shape[0]
        n_classes = self.n_classes_
        n_setimator = self.n_estimators_
        
        if mod == 'acc':
            pred = self.predict(X)
            #return np.sum(pred==Y)/pred.shape[0]

            return accuracy_score(Y, pred)
        elif mod == 'pre':
            pred = self.predict(X)
            return precision_score(Y, pred, average='macro')
        elif mod == 'recall':
            pred = self.predict(X)
            return recall_score(Y, pred, average='macro')       
        elif mod == 'F1':
            pred = self.predict(X)
            return f1_score(Y, pred, average='macro')         
              
        elif mod == 'loss':
            loss = 0
            g, Y_ = self.XY_generate(X, Y)
            
            g[g==0] = -1/(n_classes-1)
            Y_[Y_==0] = -1/(n_classes-1)
            
            scale = self.scale/sum(self.estimator_weights_)
            #norm_estimator_weights_ = scale*self.estimator_weights_

            weighted_g = np.squeeze(np.matmul(g, self.estimator_weights_[:, np.newaxis]))
            for id in range(n_samples):
                m = -np.matmul(Y_[id][:, np.newaxis].T, weighted_g[id])/n_classes
                m *= scale
                loss += sum(np.exp(m))/n_samples
            return loss
        elif mod == 'acc_loss':
            loss = 0
            g = (self.predict(X)== self.classes_[:, np.newaxis]).astype(int).T
            Y_ = np.zeros((X.shape[0], self.n_classes_), dtype=np.float)
            for i in range(X.shape[0]):
                Y_[i, self.name2id[Y[i]]] = 1

            g[g==0] = -1/(n_classes-1)
            Y_[Y_==0] = -1/(n_classes-1)

            for id in range(n_samples):
                m = -np.matmul(Y_[id][:, np.newaxis].T, g[id])/n_classes
                loss += sum(np.exp(m))/n_samples
            loss = (loss-np.exp(-1/(self.n_classes_-1)))/(np.exp(1/(self.n_classes_-1)/(self.n_classes_-1))-np.exp(-1/(self.n_classes_-1)))

            return loss
        elif mod =='Margin':
            estimator_weights_ = self.estimator_weights_/sum(self.estimator_weights_)
            rec_sum = np.zeros((X.shape[0], self.n_classes_))
            Margin = np.zeros((X.shape[0]))
            
            for estimator_id,estimator in enumerate(self.estimator_list_):
                pred = estimator.predict(X)
                for i in range(X.shape[0]):
                    rec_sum[i, self.name2id[pred[i]]] += estimator_weights_[estimator_id]
            
            for i in range(X.shape[0]):
                correct = rec_sum[i, self.name2id[Y[i]]]
                rec_sum[i, self.name2id[Y[i]]] = -1
                Margin[i] = correct-max(rec_sum[i,:])
            return Margin
                

    def _samme_proba(self, estimator, n_classes, X):
        """ Predicted probability result of specified estimator.
        Parameters:
        -----------
        estimator : classifier
            specified base estimater.
        n_classes : int
            number of Target classes

        Returns:
        -------
        Calculated matrix to be summarized.
        """

        proba = estimator.predict_proba(X)
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        log_proba = np.log(proba)

        return (n_classes - 1) * (log_proba - (1.0 / n_classes) * log_proba.sum(axis=1)[:, np.newaxis])

    def _make_estimator(self, append=True, random_state=None):
        """get the instance of templete base estimator
        Parameters:
        -----------
        append : boolean ; default=True
            whether to append the copied base estimators to estimator list.
        random_state : RandomState
            the random state of estimator
        """

        estimator = utils.clone(self.base_estimator_)
        if random_state is not None:
            utils._set_random_states(estimator, random_state)

        if append:
            self.estimator_list_.append(estimator)
        return estimator

    def _distrib_cal_(self, sample_weight):
        """ Calculate the weights of samples and mis-classification
        Parameters:
        -----------
        sample_weight : array with shape (n_samples,) or (n_samples, n_classes)
            When algorithm in {Basic, SAMME, SAMME.R}, it's weights of samples.
            When algorithm in {MH, M2}, it's the reward of classify xi to class j
        Return:
        ------
        D : array with shape (n_samples,)
            weights of samples
        Q : array with shape (n_samples, n_classes)
            reward of classify xi to class j
        """
        if self.algorithm == "Basic":
            return sample_weight, None
        if self.algorithm == "SAMME" or self.algorithm == "SAMME.R":
            return sample_weight, None
        elif self.algorithm == "M2" or self.algorithm == "MH":
            w = np.sum(sample_weight, axis=1)
            D = w/np.sum(w)
            Q = sample_weight
            return D, Q

    def _init_estimator(self, default=None):
        """ Initialize the base estimator.
        """
        if self.base_estimator is not None:
            self.base_estimator_ = self.base_estimator
        else:
            self.base_estimator_ = default
        
        if self.base_estimator_ is None:
            raise ValueError("base_estimator cannot be None")

    def _init_sample_weight(self, n_samples, n_classes, Y=None):
        """ Initialize the sample weights according to algorithm.
        """
        classes_ = np.unique(Y)
        # convert Label to label_id
        name2id = {}
        id2name = {}
        for i, class_name in enumerate(classes_):
            name2id[class_name] = i
            id2name[i] = class_name

        self.name2id = name2id
        self.id2name = id2name

        if self.algorithm == "Basic" or self.algorithm == "SAMME" or self.algorithm == "SAMME.R":
            sample_weight = np.ones((n_samples))
            sample_weight /= n_samples
        elif self.algorithm == "MH":
            sample_weight = np.ones((n_samples, n_classes))
            sample_weight.fill(1/(2*n_samples*(n_classes-1)))
            for i in range(n_samples):
                sample_weight[i, self.name2id[Y[i]]] = 1/(2*n_samples)
        elif self.algorithm == "M2":
            sample_weight = np.ones((n_samples, n_classes))
            for i in range(n_samples):
                sample_weight[i, self.name2id[Y[i]]] = 0
            sample_weight /= np.sum(sample_weight)
        return sample_weight

    def XY_generate(self, X, Y):
        """ Change the prediction result to vector type.
        """
        classes = self.classes_
        T = self.n_estimators_
        K = self.n_classes_
        N = X.shape[0]

        # 构建 N*K*T的矩阵X，N*K矩阵Y
        pred_X = np.zeros((N, K, T))
        for t, estimator in enumerate(self.estimator_list_):
            predicted = (estimator.predict(X) == classes[:, np.newaxis]).T
            pred_X[:,:,t] = predicted

        lab_Y = np.zeros((N, K), dtype=np.float)
        for i in range(N):
            lab_Y[i, self.name2id[Y[i]]] = 1

        return pred_X, lab_Y

    def copy(self):
        """ Interface to deepcopy the model.
        """
        model = copy.deepcopy(self)
        return model

    def optimize(self, X_in, Y_in, X_test, Y_test, X_val, Y_val, n_batch=None, n_epoch=1000, 
                    learning_rate=0.01, steps=1, mod='normal', print_flg=False, flg=-1, thresh=0.003, opt_type='SGD', proj=False):
        """ Optimize the model by changing estimator weight.
        Parameters:
        -----------
            X_in : Training data for optimization
            Y_in : Training labels for optimization
            X_test : Test data for optimization
            Y_test : Test labels for optimization
            n_batch : number mini-batch used in each epoch
            n_epoch : optimization epochs during optimization
            learning_rate : original learning rate
            steps : epochs between accuracy check
            
            mod : string in {'normal', 'best'} ; 
                whether return back to best weight when accuracy's decrease greater than thresh
            print_flg : boolean ; default=False
                whether calculate train loss and train accuracy and print them
            flg : int in {+1, -1}
                indicate the direction of gradient optimization
            thresh : float
                acceptable range of training accuracy
        Returns:
        --------
            alpha_opt : array with shape (n_estimators)
                The best estimator weight during optimizing.
        """
        if not n_batch:
            n_batch = X_in.shape[0]
        if opt_type=='SGD':
            self.opt = SGD_Optimizer(clf=self, learning_rate=learning_rate, proj=proj)
            alpha_opt = self.opt.optimize(X_in, Y_in, X_test, Y_test, X_val, Y_val ,n_epoch=n_epoch, n_batch=n_batch, step=steps, mod=mod, print_flg=print_flg, flg=flg, thresh=thresh)
        return alpha_opt
    
    def save(self, save_path="./model/", model_name=""):
        
        dataset_name = model_name.split('_')[0]
        if not os.path.exists(os.path.join(save_path, dataset_name)):
            os.mkdir(os.path.join(save_path, dataset_name))
        save_path = os.path.join(save_path, dataset_name)
        
        for iternum, estimator in enumerate(self.estimator_list_):
            model_path = os.path.join(save_path, model_name)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            base_classifier_name = str(iternum)+'.pkl'
            
            f = open(os.path.join(model_path, base_classifier_name), 'wb')
            pickle.dump(estimator, f)
            f.close()

        np.save(os.path.join(model_path, 'weights'), self.estimator_weights_)
        np.save(os.path.join(model_path, 'id2num'), self.id2name)
        np.save(os.path.join(model_path, 'num2id'), self.name2id)
        np.save(os.path.join(model_path, 'classes_'), self.classes_)
        return

    def load(self, load_path="./model/"):
        if os.path.exists(load_path):
            base_classifier_list = os.listdir(load_path)
        else:
            print("Can not find the model")
            return
        name_remove = ['weights.npy', 'id2num.npy', 'num2id.npy', 'classes_.npy', 'opt_weights.npy']
        
        for name in name_remove:
            if name in base_classifier_list:
                base_classifier_list.remove(name)
            else:
                print("the model is incomplete")
                return
        
        if len(base_classifier_list) < 2:
            print("Can not find the model")
            return
        
        self.estimator_list_ = []
        self.n_estimators_= len(base_classifier_list)
        
        name_dic = {}
        for i in range(len(base_classifier_list)):
            if base_classifier_list[i].split('.')[1] == 'pkl':
                name_dic[int(base_classifier_list[i].split('.')[0])] = base_classifier_list[i]
        classifier_list = sorted(name_dic.items(), key = lambda item:item[0])
        
        for (i, file_name) in classifier_list:
            file_path = os.path.join(load_path, file_name)
            f = open(file_path,'rb')
            base_clf = pickle.load(f)
            f.close()
            self.estimator_list_.append(base_clf)
        
        self.estimator_weights_ = np.load(os.path.join(load_path, 'weights.npy'))
        self.opt_estimator_weights_ = np.load(os.path.join(load_path, 'opt_weights.npy'))
        #self.estimator_weights_ = self.estimator_weights_/np.sum(self.estimator_weights_)
        self.id2name = np.load(os.path.join(load_path, 'id2num.npy'), allow_pickle=True).item()
        self.name2id = np.load(os.path.join(load_path, 'num2id.npy'), allow_pickle=True).item()
        self.classes_ = np.load(os.path.join(load_path, 'classes_.npy'), allow_pickle=True)
        self.n_classes_ = self.classes_.shape[0]
        return
