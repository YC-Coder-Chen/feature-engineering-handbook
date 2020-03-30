#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:18:30 2020

@author: chenyingxiang
"""

import random
import numpy as np
from tqdm import tqdm
import random
from sklearn.model_selection import KFold
from deap import base, creator, tools, algorithms

random.seed()
np.random.seed()

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) 

class Genetic_Algorithm(object):
    """
    Genetic Algorithm Algorithm for feature selection

    Parameters
    ----------
    n_pop: int, default =20
        The number of population
    
    n_gen: int, default = 20
        The number of generation
 
    both: boolean, default = True
        Whether offsprings can result from both crossover and mutation
        If False, offsprings can result from one of them.
    
    n_children: int, default = None
        The number of children to produce when offsprings can only result from one of the operations
            including crossover, mutation and reproduction
        Default None will set n_children = n_pop
        n_children corresponds with the lambda_ parameter in deap.algorithms.varOr        
    
    cxpb: float, default = 0.5
        The probability of mating two individuals
        The sum of cxpb and mutpb shall be in [0,1]
    
    mutpb: float, default = 0.3
        The probability of mutating an individual
        The sum of cxpb and mutpb shall be in [0,1]
    
    cx_indpb: float, default = 0.25
        The independent probabily for each attribute to be exchanged under uniform crossover.
    
    mu_indpb: floatt, default = 0.25
        The independent probability for each attribute to be flipped under mutFlipBit.
    
    algorithm: string, default="one-max"
        The offspring selection algorithm
        "NSGA2" is also available
    
    loss_func: object
        The loss function of the ML task. 
        loss_func(y_true, y_pred) should return the loss.
        
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
    
    loss_dict: dictionary
        Store the evaluation results to speed up fitting process
    
    References
    ----------
    1. https://deap.readthedocs.io/en/master/index.html
    2. https://github.com/kaushalshetty/FeatureSelectionGA
    3. Haupt, R. L. (1995). An introduction to genetic algorithms for electromagnetics. 
        IEEE Antennas and Propagation Magazine, 37(2), 7-15.
    4. Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. A. M. T. (2002). A fast and elitist multiobjective genetic algorithm: NSGA-II. 
        IEEE transactions on evolutionary computation, 6(2), 182-197.
    5. Mkaouer, W., Kessentini, M., Shaout, A., Koligheu, P., Bechikh, S., Deb, K., & Ouni, A. (2015). Many-objective software remodularization using NSGA-III. 
        ACM Transactions on Software Engineering and Methodology (TOSEM), 24(3), 1-45.
    6. Fortin, F. A., Rainville, F. M. D., Gardner, M. A., Parizeau, M., & Gagné, C. (2012). DEAP: Evolutionary algorithms made easy. 
    	Journal of Machine Learning Research, 13(Jul), 2171-2175.
    
    """
    
    def __init__(self, loss_func, estimator, n_pop = 20, n_gen = 20, both = True, n_children = None, 
                 cxpb = 0.5, mutpb = 0.2, cx_indpb = 0.25, mu_indpb = 0.25,
                 algorithm = "one-max", predict_type = 'predict'):
        
        #### check type
        if not hasattr(estimator, 'fit'):
            raise ValueError('Estimator doesn\' have fit method')
        if not hasattr(estimator, 'predict') and not hasattr(estimator, 'predict_proba'):
            raise ValueError('Estimator doesn\' have predict or predict_proba method')
            
        for instant in [cxpb, mutpb, cx_indpb, mu_indpb]:
            if type(instant) != float:
                raise TypeError(f'{instant} should be float type')
            if (instant > 1) or (instant) < 0:
                raise ValueError(f'{instant} should be within range [0,1]')
        
        for instant in [n_pop, n_gen]:
            if type(instant) != int:
                raise TypeError(f'{instant} should be int type')      
        
        if type(both) != bool:
            raise TypeError(f'{both} should be boolean type')
            
        if predict_type not in ['predict', 'predict_proba']:
            raise ValueError('predict_type should be "predict" or "predict_proba"')

        if algorithm not in ['one-max', 'NSGA2']:
            raise ValueError('algorithm should be "one-max" or "NSGA2"')
      
        if not n_children:
            n_children = n_pop

        if type(n_children) != int:
            raise TypeError(f'{n_children} should be int type')
            
        if (cxpb + mutpb) > 1.0:
            raise ValueError(f'The sum of cxpb and mutpb shall be in [0,1]')
        
        self.n_pop = n_pop
        self.n_gen = n_gen
        self.both = both
        self.n_children = n_children
        self.cxpb = cxpb
        self.mutpb = mutpb 
        self.cx_indpb = cx_indpb
        self.mu_indpb = mu_indpb
        self.algorithm = algorithm
        self.loss_func = loss_func
        self.estimator = estimator
        self.predict_type = predict_type
        self.loss_dict = dict()

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
    
    def _eval_fitness(self, individual):
        
        individual = [True if x else False for x in individual]
        
        if sum(individual) == 0:
            current_loss = np.Inf
        else:       
            encoded_str = ''.join(['1' if x else '0' for x in individual])
            if self.loss_dict.get(encoded_str):
                current_loss = self.loss_dict.get(encoded_str)
            else:
                if self.cv:
                    current_loss = self._cross_val(self.X_train[:,individual], self.y_train, 
                                                   self.estimator, self.loss_func, self.cv)
                    current_loss = np.round(current_loss, 4)
                            
                elif type(self.X_val) is np.ndarray:
                    current_loss = self._get_cost(self.X_train[:,individual], self.y_train, 
                                                     self.estimator, self.loss_func, 
                                                     self.X_val[:,individual], self.y_val)
                    current_loss = np.round(current_loss, 4)
                            
                else:    
                    current_loss = self._get_cost(self.X_train[:,individual], self.y_train, 
                                                  self.estimator, self.loss_func, None, None)   
                    current_loss = np.round(current_loss, 4)
                self.loss_dict[encoded_str] = current_loss
                    
        if self.algorithm == "one-max":
            return current_loss,
        else:
            return current_loss, sum(individual)

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
            [1] If cv = None and X_val = None, the GA will evaluate each subset on trainset.
            [2] If cv != None and X_val = None, the GA will evaluate each subset on generated validation set using k-fold.
            [3] If cv = None and X_val != None, the GA will evaluate each subset on the user-provided validation set. 
       
        X_val: numpy array, shape = (n_samples, n_features) or None. default = None.
            The validation input samples. None means no validation set is provoded.
            [1] If cv = None and X_val = None, the GA will evaluate each subset on trainset.
            [2] If cv != None and X_val = None, the GA will evaluate each subset on generated validation set using k-fold.
            [3] If cv = None and X_val != None, the GA will evaluate each subset on the user-provided validation set.
        
        y_val: numpy array, shape = (n_samples, ) or None. default = None.
            The validation target values (class labels in classification, real numbers in regression).
        
        Returns
        -------
        self : object
        
        """
        
        # make sure input has two dimensions
        assert len(X_train.shape) == 2
        num_feature = X_train.shape[1]
        
        # save them for _eval_fitness function
        self.X_train = X_train
        self.y_train = y_train
        self.cv = cv
        self.X_val = X_val
        self.y_val = y_val
        
        # creator
        if self.algorithm == "one-max":
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,)) # minimize the loss
            creator.create("Individual", list, fitness=creator.FitnessMin)
        else:
            creator.create("FitnessMulti", base.Fitness, weights=(-1.0, -0.1))
            creator.create("Individual", list, fitness=creator.FitnessMulti)            
        
        # register
        toolbox = base.Toolbox()
        toolbox.register("gene", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat, creator.Individual, 
                         toolbox.gene, n = num_feature)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual, 
                         n = self.n_pop)
        toolbox.register("evaluate", self._eval_fitness)
        toolbox.register("mate", tools.cxUniform, indpb = self.cx_indpb)
        toolbox.register("mutate", tools.mutFlipBit, indpb = self.mu_indpb)
        
        if self.algorithm == "one-max":
            toolbox.register("select", tools.selTournament, tournsize=5)
        else:
            toolbox.register("select", tools.selNSGA2)

        # start evolution
        # evaluate inital population
        population = toolbox.population()
        fits = toolbox.map(toolbox.evaluate, population)
        for ind, fit in zip(population, fits):
            ind.fitness.values = fit
        
        # evolving
        for gen in tqdm(range(self.n_gen)):
            if self.both:
                offspring = algorithms.varOr(population, toolbox, 
                                              lambda_ = self.n_children, cxpb = self.cxpb,
                                              mutpb = self.mutpb)
            else:
                offspring = algorithms.varAnd(population, toolbox, cxpb = self.cxpb,
                                              mutpb = self.mutpb)  
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            
            if self.algorithm == 'one-max':
                population = toolbox.select(offspring, k = self.n_pop)
            else:
                population = toolbox.select(offspring + population, k = self.n_pop)
        
        fits = list(toolbox.map(toolbox.evaluate, population))
        if self.algorithm != "one-max":
            fits = [x[0] for x in fits]
        
        try:
            best_idx = np.argmin(np.array(fits))
            self.best_sol = [True if x else False for x in population[best_idx]]
            self.best_loss = fits[best_idx]  
            
            if np.isinf(self.best_loss): # if best loss is inf
                best_key = min([(value, key) for key, value in self.loss_dict.items()])[1]
                self.best_sol = [True if x == '1' else False for x in best_key]
                self.best_loss = min([(value, key) for key, value in self.loss_dict.items()])[0]   
        except:
            best_key = min([(value, key) for key, value in self.loss_dict.items()])[1]
            self.best_sol = [True if x == '1' else False for x in best_key]
            self.best_loss = min([(value, key) for key, value in self.loss_dict.items()])[0]
 
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