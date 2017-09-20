#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:09:59 2017

@author: Joe Bockhorst, jbockhor@amfam.com
"""
import itertools
import numpy as np
import sklearn
from os.path import exists
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import PredefinedSplit
from sklearn.isotonic import IsotonicRegression

__all__ = ['RankScoreIsoRegression', 'PairwiseRankClf']

class RankScoreIsoRegression():
    """
    References
    ----------
    `Predicting Self-reported Customer Satisfaction of Interactions with a 
     Corporate Call Center <http://ecmlpkdd2017.ijs.si/papers/paperID598.pdf>`,

    """
    
    def __init__(self, mask_size=100, pr_args={}, ir_args={"out_of_bounds":"clip"}):
        """
        Parameters
        ----------
        mask_size : int, (default=100)
            Length of the mask for smoothing rank scores
        pr_args : dict, (default={})
            Keyword arguments to PairwiseRankClf constructor
        ir_args : dict, (default={"out_of_bounds":"clip"})
            Keyword arguments to IsotonicRegression constructor

        """
        self.ir_args = ir_args
        self.pr_args = pr_args
        self.pr_clf = PairwiseRankClf(**pr_args)
        self.ir_model = IsotonicRegression(**ir_args)
        self.mask_size = mask_size

    def fit(self, X, y):
        self.pr_clf.fit(X, y)
        rank_scores = self.pr_clf.decision_function(X)
        
        if self.mask_size is None:
            self.ir_model.fit(rank_scores, y)
        else:
            mask = 1 + np.zeros((self.mask_size,))
            idx = np.argsort(rank_scores)
            rank_scores_ordered = rank_scores[idx]
            y_ordered = y[idx]
            rank_scores_smoothed = np.convolve(rank_scores_ordered, mask, mode="valid") / float(mask.size)
            y_smoothed = np.convolve(y_ordered, mask, mode="valid") / float(mask.size)
            self.ir_model.fit(rank_scores_smoothed, y_smoothed)
        return self
    
    def rank_scores(self, X):
        return self.pr_clf.decision_function(X)
    
    def predict(self, X):
        return self.ir_model.predict(self.rank_scores(X))
    
class PairwiseRankClf():
    """
    References
    ----------
    `Support Vector Learning for Ordinal Regression <http://www.herbrich.me/papers/icann99_ordinal.pdf>`,
    `Learning to Rank with the Pairwise Transform <http://>http://fa.bianp.net/blog/2012/learning-to-rank-with-scikit-learn-the-pairwise-transform/`__
    """    

    CLF_DEFAULT_ARGS = {
            "C" : [10**x for x in range(-18, -4)],
            "penalty" : ["l2"],
            "dual" : [False],
            "class_weight" : ["balanced" if sklearn.__version__ >= "0.17.1" else "auto"]}
    
    def __init__(self,
                 param_grid=None,
                 tuning_fraction=0.75,
                 pairs_filename=None, 
                 n_tuning_training_samples=10000,
                 n_tuning_eval_samples=10000,
                 n_training_samples=10000,
                 scoring="roc_auc",
                 seed=None,
                 verbose=False):
        """
        Parameters
        ----------
        param_grid : dict, optional
            parameter grid for tuning hyperparameters. Default is 
            PairwiseRankClf.CLF_DEFAULT_ARGS
        tuning_fraction : float, (default=0.75)
            Fraction of training examples passed to fit to use as training set 
            while tuning hyper-parameters. The remainder of the examples are used 
            for estimating performance.
        pairs_filename : string, optional
            If set, pairs are read from the specified file if it exists or 
            written to that file otherwise. Caching pairs in a file can help
            speed up training when training sets are large. '.npz' is appended
            (by numpy) to the filename if it does not end with '.npz'
        n_tuning_training_samples : int, (default=10000)
            The number of pairwise samples to use for training models
            during hyperpararmeter tuning. That is, the size of the train_prime set.
        n_tuning_eval_samples : int, (default=10000)
            The number of pairwise samples used to evaluate trained models
            during hyperparameter tuning. That is, the size of the tuning set.
        n_training_samples : int, (default=10000)
            The number of pairwise samples to use for training the underlying 
            classifier after hyperparametrs have been tuned. 
        scoring : string or callable, (default='roc_auc')
            the scoring parameter for GridSearchCV()
        seed : int, optional
            if set np.rand(seed) is called at the start of fit()
        """
        self.param_grid = param_grid if not param_grid is None else PairwiseRankClf.CLF_DEFAULT_ARGS
        self._pairwise_clf = None
        self.tuning_fraction = tuning_fraction
        self.seed = seed
        self.pairs_filename = pairs_filename
        self.n_tuning_training_samples = n_tuning_training_samples
        self.n_tuning_eval_samples = n_tuning_eval_samples
        self.n_training_samples = n_training_samples
        self.scoring = scoring
        self.verbose = verbose
        
        if not pairs_filename is None and not pairs_filename.endswith(".npz"):
            self.pairs_filename = pairs_filename + ".npz"
    
    def fit(self, X, y):
        """Train the model
        
        Parameters
        ----------
        X : array, shape=[n_examples, n_features]
        y : array, shape=[n_examples]
        """

        #
        # split natural training examples for tuning hyper-parameters 
        # prior to creatign the pairwise examples to prevent bleed-through
        # during tuning
        #
        if not self.seed is None:
            np.random.seed(self.seed)
            
        n_tr_prime = int(X.shape[0] * self.tuning_fraction)
        self.rand_idx = np.random.permutation(X.shape[0])
        self.tr_prime_idx = self.rand_idx[:n_tr_prime]  # indicies of examples in the train' set
        self.tu_idx = self.rand_idx[n_tr_prime:] # indicies of examples in the tuning set

        #
        # generate pairs or load from file
        #
        pairs = self._get_pairs(y)

        #
        # hyperparameter tuning
        #            
        X_pairwise_tr_prime, y_pairwise_tr_prime, sp_tr_prime = sample_pairwise_examples(n=self.n_tuning_training_samples, 
                                pairs=pairs, 
                                whitelist=self.tr_prime_idx, 
                                X=X, 
                                seed=self.seed)
                                
        X_pairwise_tune, y_pairwise_tune, sp_tune = sample_pairwise_examples(n=self.n_tuning_eval_samples,
                                pairs=pairs, 
                                whitelist=self.tu_idx, 
                                X=X, 
                                seed=self.seed)

        # Hack to use GridSearchCV with a single fold.        
        test_idx = ([-1] * self.n_tuning_training_samples) + ([0] * self.n_tuning_eval_samples)
        cv = PredefinedSplit(test_idx) # A CV object with 1-fold.

        self._gridsearch = GridSearchCV(LinearSVC(), param_grid=self.param_grid, 
                                          cv=cv,
                                          scoring=self.scoring,
                                          refit=False)
        
        X_tmp = np.concatenate((X_pairwise_tr_prime, X_pairwise_tune))
        y_tmp = np.concatenate((y_pairwise_tr_prime, y_pairwise_tune))

        # TODO : Remove this block
#        self.fit_X = X
#        self.fit_y = y
#        self.X_pairwise_tr_prime = X_pairwise_tr_prime
#        self.y_pairwise_tr_prime = y_pairwise_tr_prime
#        self.X_pairwise_tune = X_pairwise_tune
#        self.y_pairwise_tune = y_pairwise_tune
#        self.pairs = pairs
#        self.sp_tr_prime = sp_tr_prime
#        self.sp_tune = sp_tune
#        #
        
        self._gridsearch.fit(X_tmp, y_tmp)

        #
        # Final tranining. Re-sample pairwise traing set from all pairs X
        # and use best hyperparameters 
        #
        if (self.verbose):
            print("Training final model with best_params: {}".format(self._gridsearch.best_params_))
            
        X_pairwise_tr, Y_pairwise_tr, _ = sample_pairwise_examples(n=self.n_training_samples,
                                pairs=pairs, X=X, 
                                seed=self.seed)
        self._pairwise_clf = LinearSVC(**self._gridsearch.best_params_)
        self._pairwise_clf.fit(X_pairwise_tr, Y_pairwise_tr)

        # TODO : Remove this block
#        self.X_pairwise_tr = X_pairwise_tr
#        self.Y_pairwise_tr = Y_pairwise_tr
        #########        
    
    def decision_function(self, X):
        return self._pairwise_clf.decision_function(X)
    
    def _get_pairs(self, y):
        if self.pairs_filename is None or not exists(self.pairs_filename):
            pairs = create_ranking_pairs(y, verbose=self.verbose)
            if not self.pairs_filename is None:
                save_pairs(self.pairs_filename, pairs)
        else:
            pairs = load_pairs(self.pairs_filename, verbose=self.verbose)
        return pairs
        
def save_pairs(filename, pairs, verbose=True):
    """ 
    Save pairwise ranking indexes to a file

    pairs - N-by-2 numpy array of index values    
    """
    assert type(pairs) is np.ndarray
    assert pairs.ndim == 2 and pairs.shape[1] == 2
    d = {"pairs" : pairs}
    np.savez(filename, **d)
    if verbose:
        print("{} pairwise ranking indexes saved to {}".format(pairs.shape[0], filename))
    
def load_pairs(filename, verbose=False):
    """ Load ranking indexes, previously saved with save_pairs(), from file"""
    if verbose:
        print("loading pairs from {}".format(filename))
    result = np.load(filename)["pairs"]
    return result

def create_ranking_pairs(y, elgible=None, verbose=False):
    """Return pairs of indexes elgible for training a pairwise ranking classifier.
    
    Parameters
    ----------
        y : listlike, shape = [n_examples]
        elgible : function(y1, y2), optional
            Elgibility function that returns True when a valid ranking
            pair can be made from examples with labels y1 and y2.  
            Default is lambda y1, y2: y1 != y2
            
            
    Return
    ------
        array : shape = [-1, 2]
            An array with pairs in the rows. Returned pair (i, j) means 
            that y[i] < y[j]
    

    Example
    -------
        >>> create_ranking_pairs([0,1,1,0])
            [(0, 1), (0, 2), (3, 1), (3, 2)]
            
        >>> create_ranking_pairs([False, True, False])
            [(0, 1), (2, 1)]
    """
    if verbose:
        print("creating_ranking_pairs()")
    y = np.array(y)
    if y.ndim > 2 or (y.ndim == 2 and y.shape[1] != 1):
        raise ValueError("y should be 1-dim or N-by-1")
        
    elgible = elgible if not elgible is None else lambda y1, y2: y1 != y2
    result = []
    comb = itertools.combinations(range(y.size), 2)
    N_check = y.size * (y.size -1) / 2

    for idx, (i, j) in enumerate(comb):
        if verbose and idx % 1e6 == 0:
            print("checking pair {} of  {} ({:.2f}%) : ({}, {})".format(idx, N_check, 100.0*idx/N_check, i, j))
        yi, yj = y[i], y[j]
        if elgible(yi, yj):
            if yi < yj:
                result.append((i, j))
            else:
                result.append((j, i))
    return np.array(result)

def sample_pairwise_examples(n, X, pairs, 
                             with_replacement=True, 
                             whitelist=None, 
                             seed=None):
    """Sample pairwise examples
    
    Parameters
    ----------
    n : int, number of pairwise examples to sample
    X : array, shape = [n_examples, n_features] 
        array of original feature values
    pairs : array, shape = [n_pairs, 2]
        Pairs to sample from. For format see create_ranking_pairs()
    with_replacement : bool, optional
        If True sample with replacement
    whitelist : Iterable, optional
        If set provides a list of elgible example indexes. A pair (i, j) will
        only be returned if both i and j are in the elgible list. Helpful when
        splitting examples X for cross-validation purposes.
    seed : int, default=None
        If not None, np.random.seed(seed) is called prior to sampling
    
    Returns
    -------
    X_pairwise : array, shape = [n, num_features]
        The pairwise examples. 
    Y_pairwise : array, shape = [n]
        Class values Y_out will be approximately balanced.
    sampled_pairs : array, shape = [n, 2]
        The list of sample example indexes. If kth element in sample_pairs 
        is (i, j) means X_pairwise[k,:] = X[i, :] - X[j, :] 
    
    """
    if whitelist is None:
        pairs = pairs
    else:
        whitelist = set(whitelist)
        pairs = np.array([(p[0], p[1]) for p in pairs if p[0] in whitelist and p[1] in whitelist])

    if not seed is None:
        np.random.seed(seed)
        
    N = pairs.shape[0]
    if with_replacement:
        indexes = np.random.randint(N, size=n)
    else:
        if N > n:
            raise ValueError("Cannot sample n times without replacement from set smaller than n")
        indexes = np.random.permutation(N)[:n]
    # TODO : remov this block
#    if True:
#        DEBUG.random_perm.append(indexes)
#        DEBUG.elgible_pairs.append(pairs)
        
    X_pairwise = np.zeros((n, X.shape[1])) + np.nan
    Y_pairwise = np.zeros((n,)) + np.nan
    sampled_pairs = np.zeros((n, 2), dtype=int) 

    for ii, idx in enumerate(indexes):            
        i, j = pairs[idx, :]
        sampled_pairs[ii, :] = i, j
        if ii % 2 == 0:  # keep Y balanced
            X_pairwise[ii, :] = X[j, :] - X[i, :]
            Y_pairwise[ii] = 1
            sampled_pairs[ii, :] = i, j
        else:
            X_pairwise[ii, :] = X[i, :] - X[j, :]
            Y_pairwise[ii] = -1
            sampled_pairs[ii, :] = j,  i
            
    assert np.isnan(X_pairwise).sum() == 0
    assert np.isnan(Y_pairwise).sum() == 0
    return X_pairwise, Y_pairwise, sampled_pairs


