#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulated Annealing

@author: chenyingxiang
"""

import random
import numpy as np
from sklearn.model_selection import KFold

random.seed()
np.random.seed()

class Simulated_Annealing(object):
    """
    Simulated Annealing Algorithm for feature selection

    Parameters
    ----------
    init_temp: float, default: 100.0
        The initial temperature
        
    min_temp: float, default: 1.0
        The minimum temperature to stop
        
    max_perturb: float, default: 0.2
        The maximum percentage of perturbance genarated
         
    alpha: float, default: 0.98
        The decay coefficient of temperature
    
    k: float, default: 1.0
        The constant for computing probability
    
    loss_func: object
        The loss function of the ML task. 
        loss_func(y_true, y_pred) should return the loss.
    
    iteration: int, default: 50
        Number of iteration when temperature level is above min_temp each time.
        
    estimator: object
        A supervised learning estimator 
        It has to have the `fit` and `predict` method (or `predict_proba` method for classification)
        
    predict_type: string, default="predict"
        Final prediction type.
        - For some classification loss functions, probability output is required.
          Should set predict_type to "predict_proba"
        
    Attributes
    ----------
    best_sol: np.array of int
        The index of the best subset of features.
    
    best_loss: float
        The loss associated with the best_sol
    
    References
    ----------
    1. https://blog.csdn.net/Joseph__Lagrange/article/details/94410317
    2. https://github.com/JeromeBau/SimulatedAnnealing/blob/master/gibbs_annealing.py
    
    """
    
    
    def __init__(self, loss_func, estimator, init_temp = 100.0, min_temp = 0.01, k = 1.0, 
                 max_perturb = 0.2, alpha = 0.98, iteration = 50, predict_type = 'predict'):
        
        #### check type
        if not hasattr(estimator, 'fit'):
            raise ValueError('Estimator doesn\' have fit method')
        if not hasattr(estimator, 'predict') and not hasattr(estimator, 'predict_proba'):
            raise ValueError('Estimator doesn\' have predict or predict_proba method')
            
        for instant in [init_temp, min_temp, k, max_perturb, alpha]:
            if type(instant) != float:
                raise TypeError(f'{instant} should be float type')
        
        if type(iteration) != int:
            raise TypeError(f'{iteration} should be int type')
            
        if predict_type not in ['predict', 'predict_proba']:
            raise ValueError('predict_type should be "predict" or "predict_proba"')
      
        self.loss_func = loss_func
        self.estimator = estimator
        self.init_temp = init_temp
        self.min_temp = min_temp
        self.k = k
        self.max_perturb = max_perturb
        self.alpha = alpha
        self.iteration = iteration
        self.predict_type = predict_type
        self.loss_dict = dict()
                        
    def _judge(self, new_cost, old_cost, temp):
        
        delta_cost = new_cost - old_cost
        
        if delta_cost < 0: # new solution is better
            proceed = 1
        else:
            probability = np.exp(-1 * delta_cost / (self.k * temp))
            if probability > np.random.random():
                proceed = 1
                
            else:
                proceed = 0
        
        return proceed
    
    def _get_neighbor(self, num_feature, current_sol, max_perturb):
        
        all_feature = np.ones(shape=(num_feature,)).astype(bool)
        outside_feature = np.where(all_feature != current_sol)[0]
        inside_feature = np.where(all_feature == current_sol)[0]
        num_perturb_in = int(max(np.ceil(len(inside_feature) * max_perturb),1))
        num_perturb_out = int(max(np.ceil(len(outside_feature) * max_perturb),1))
        if len(outside_feature) == 0:
            feature_in = np.array([])
        else:
            feature_in = np.random.choice(outside_feature, 
                                          size = min(len(outside_feature), 
                                                 np.random.randint(0, num_perturb_in + 1)), 
                                          replace = False) # uniform distribution
        if len(inside_feature) == 0:
            feature_out = np.array([])
        else:
            feature_out = np.random.choice(inside_feature , 
                                           size = min(len(inside_feature), 
                                                 np.random.randint(0, num_perturb_out + 1)),
                                           replace = False) # uniform distribution
        feature_change = np.append(feature_in, feature_out).astype(int)
        all_feature[feature_change] = 1 - all_feature[feature_change]
        
        return all_feature    
    
    def _get_cost(self, X, y, estimator, loss_func, X_test = None, y_test = None):
        
        estimator.fit(X, y.ravel())
        if type(X_test) is np.ndarray:
            if self.predict_type == "predict_proba": # if loss function requires probability
                y_test_pred = estimator.predict_proba(X_test)
                return loss_func(y_test, y_test_pred)
            else:
                y_test_pred = estimator.predict(X_test)
                return loss_func(y_test, y_test_pred)
        
        y_pred = estimator.predict(X)
        
        return loss_func(y, y_pred)
    
    
    def _cross_val(self, X, y, estimator, loss_func, cv):     
        
        loss_record = []
        
        for train_index, test_index in KFold(n_splits = cv).split(X):  # k-fold
            
            try: 
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                estimator.fit(X_train, y_train.ravel())
                
                if self.predict_type == "predict_proba":
                    y_test_pred = estimator.predict_proba(X_test)
                    loss = loss_func(y_test, y_test_pred)         
                    loss_record.append(loss)
                else:
                    y_test_pred = estimator.predict(X_test)
                    loss = loss_func(y_test, y_test_pred)
                    loss_record.append(loss)
            except:
                continue
       
        return np.array(loss_record).mean()
    
    def fit(self, X_train, y_train, cv = None, X_val = None, y_val = None, 
            init_sol = None, stop_point = 5):
     
        
        """
        Fit method.
        
        Parameters
        ----------
        X_train: numpy array shape = (n_samples, n_features).
            The training input samples.
        
        y_train: numpy array, shape = (n_samples,).
            The target values (class labels in classification, real numbers in regression).
            
        cv: int or None, default = None
            Specify the number of folds in KFold. None means SA will not use 
            k-fold cross-validation results to select features.
            [1] If cv = None and X_val = None, the SA will evaluate each subset on trainset.
            [2] If cv != None and X_val = None, the SA will evaluate each subset on generated validation set using k-fold.
            [3] If cv = None and X_val != None, the SA will evaluate each subset on the user-provided validation set. 
       
        X_val: numpy array, shape = (n_samples, n_features) or None. default = None.
            The validation input samples. None means no validation set is provoded.
            [1] If cv = None and X_val = None, the SA will evaluate each subset on trainset.
            [2] If cv != None and X_val = None, the SA will evaluate each subset on generated validation set using k-fold.
            [3] If cv = None and X_val != None, the SA will evaluate each subset on the user-provided validation set.
        
        y_val: numpy array, shape = (n_samples, ) or None. default = None.
            The validation target values (class labels in classification, real numbers in regression).
        
        init_sol: numpy array, shape = (num_feautre, ) or None. default = None.
            The initial solution provided by the user. It should contain bools.
            A good inital solution will save SA algorithm a lot of searching time.
            None means the SA will randomly generated a inital solution.
        
        stop_point: int, default = 5.
            The stopping conditions. If the optimal loss keeps the same for a few iterantions, then it will stop.
      
        Returns
        -------
        self : object
        
        """
      
        # make sure input has two dimensions
        assert len(X_train.shape) == 2
        num_feature = X_train.shape[1]
        
        # get initial solution
        if init_sol == None:
            init_sol = np.random.randint(2, size=num_feature)
            while sum(init_sol)==0:
                init_sol = np.random.randint(2, size=num_feature)
        
        current_sol = init_sol
        if cv:
            current_loss = self._cross_val(X_train[:,current_sol], y_train, 
                                     self.estimator, self.loss_func, cv)
            current_loss = np.round(current_loss, 4)
            
        elif type(X_val) is np.ndarray:
            current_loss = self._get_cost(X_train[:,current_sol], y_train, self.estimator, 
                                    self.loss_func, X_val[:,current_sol], y_val) 
            current_loss = np.round(current_loss, 4)
            
        else:    
            current_loss = self._get_cost(X_train[:,current_sol], y_train, self.estimator, 
                                    self.loss_func, None, None)
            current_loss = np.round(current_loss, 4)
        
        encoded_str = ''.join(['1' if x else '0' for x in current_sol])
        self.loss_dict[encoded_str] = current_loss 
        temp_history = [self.init_temp]
        loss_history = [current_loss]
        sol_history = [current_sol]
        
        current_temp = self.init_temp
        current_temp = np.round(current_temp, 4)
        
        best_loss = current_loss
        best_sol = current_sol
        
        # start looping
        while current_temp > self.min_temp:
            for step in range(self.iteration):
                current_sol = self._get_neighbor(num_feature, current_sol, self.max_perturb)
                if len(current_sol) == 0:
                    current_loss = np.Inf
                else:
                    encoded_str = ''.join(['1' if x else '0' for x in current_sol])
                    if self.loss_dict.get(encoded_str):
                        current_loss = self.loss_dict.get(encoded_str)
                    else:
                        if cv:
                            current_loss = self._cross_val(X_train[:,current_sol], y_train, 
                                                     self.estimator, self.loss_func, cv)
                            current_loss = np.round(current_loss, 4)
                            
                        elif type(X_val) is np.ndarray:
                            current_loss = self._get_cost(X_train[:,current_sol], y_train, self.estimator, 
                                                    self.loss_func, X_val[:,current_sol], y_val)
                            current_loss = np.round(current_loss, 4)
                            
                        else:    
                            current_loss = self._get_cost(X_train[:,current_sol], y_train, self.estimator, 
                                                    self.loss_func, None, None)   
                            current_loss = np.round(current_loss, 4)
                        self.loss_dict[encoded_str] = current_loss                    
 
                if (current_loss - best_loss) <= 0: # update temperature
                    current_temp = current_temp * self.alpha
                    current_temp = np.round(current_temp, 4)
               
                # judge
                if self._judge(current_loss, best_loss, current_temp): # take new solution
                    best_sol = current_sol 
                    best_loss = current_loss                                                           
                
            # keep record
            temp_history.append(current_temp)
            loss_history.append(best_loss)
            sol_history.append(best_sol)
            
            # debugging Pipeline
            # print(f"Current temperature is {current_temp}")
            # print(f"Current best loss is {best_loss}")
            # print(f"Current best solution is {best_sol}")
            
            # check stopping condition
            if len(loss_history) > stop_point:
                if len(np.unique(loss_history[-1 * stop_point : ])) == 1:
                    print(f"Stopping condition reached!")
                    break
        
        best_idx = np.argmin(loss_history)
        self.best_sol = sol_history[best_idx]
        self.best_loss = loss_history[best_idx]
  
    def transform(self, X):
        """
        Transform method.
        
        Parameters
        ----------
        X: numpy array shape = (n_samples, n_features).
            The data set needs feature reduction.
      
        Returns
        -------
        transform_X: numpy array shape = (n_samples, n_best_features).
            The data set after feature reduction.
        
        """
        transform_X = X[:, self.best_sol]
        return transform_X