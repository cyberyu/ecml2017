#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 08:59:35 2017

@author: Joe Bockhorst
"""
import numpy as np
import os
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import GridSearchCV

from rsir import RankScoreIsoRegression

def load_data():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data")
    _tmp = np.load(os.path.join(data_dir, "data.npz"))
    X_tr = _tmp["X_tr"]
    X_te = _tmp["X_te"]
    y_tr = _tmp["y_tr"]
    y_te = _tmp["y_te"]
    return X_tr, X_te, y_tr, y_te

if __name__ == "__main__":
    print("This example reproduces the table in Figure 3 of ")
    print("'Predicting Self-reported Customer Satisfaction of")
    print("Interactions with a Corporate Call Center'." )
    print("by J. Bockhorst, S. Yu, L. Polania and G.Fung. ECML 2017")
    print("")
    
    X_tr, X_te, y_tr, y_te = load_data()
    print("data loaded")
    print("X_tr shape: {}".format(X_tr.shape))
    print("X_te shape: {}".format(X_te.shape))
    print("y_tr shape: {}".format(y_tr.shape))
    print("y_te shape: {}".format(y_te.shape))
    
    pr_args = { 
       "verbose":False,
       "n_training_samples":50000, 
       "n_tuning_training_samples":50000,
       "n_tuning_eval_samples":10000,
       "seed" : 1}

    print("")
    print("Training Rank Score + Isotonic Regression model")    
    rsir = RankScoreIsoRegression(pr_args=pr_args)
    rsir.fit(X_tr, y_tr)

    print("RS")    
    y_hat_tr = rsir.rank_scores(X_tr)
    y_hat_te = rsir.rank_scores(X_te)
    print("    TR pearson:{}, spearman:{}".format(
            pearsonr(y_tr, y_hat_tr)[0],
            spearmanr(y_tr, y_hat_tr).correlation))    
    print("    TE pearson:{}, spearman:{}".format(
            pearsonr(y_te, y_hat_te)[0],
            spearmanr(y_te, y_hat_te).correlation))
        
    
    print("RS+IR")    
    y_hat_tr = rsir.predict(X_tr)
    y_hat_te = rsir.predict(X_te)
    print("    TR pearson:{}, spearman:{} MAE:{}".format(
            pearsonr(y_tr, y_hat_tr)[0],
            spearmanr(y_tr, y_hat_tr).correlation,
            np.abs(y_tr-y_hat_tr).mean()))
    print("    TE pearson:{}, spearman:{} MAE:{}".format(
            pearsonr(y_te, y_hat_te)[0],
            spearmanr(y_te, y_hat_te).correlation,
            np.abs(y_te-y_hat_te).mean()))


    # LASSO    
    print("LASSO")
    clf = GridSearchCV(estimator=Lasso(), 
                       param_grid={"alpha" : [2**i for i in range(-8, -1)]})
    clf.fit(X_tr, y_tr)
    print("    Best alpha : 2**{}".format(np.log2(clf.best_params_["alpha"])))
    y_hat_tr = clf.predict(X_tr)
    y_hat_te = clf.predict(X_te)
    print("    TR pearson:{}, spearman:{} MAE:{}".format(
            pearsonr(y_tr, y_hat_tr)[0],
            spearmanr(y_tr, y_hat_tr).correlation,
            np.abs(y_tr-y_hat_tr).mean()))
    print("    TE pearson:{}, spearman:{} MAE:{}".format(
            pearsonr(y_te, y_hat_te)[0],
            spearmanr(y_te, y_hat_te).correlation,
            np.abs(y_te-y_hat_te).mean()))
    
    # RIDGE
    print("RIDGE")
    clf = GridSearchCV(estimator=Ridge(), 
                       param_grid={"alpha" : [2**i for i in range(13, 20)]})
    clf.fit(X_tr, y_tr)
    print("    Best alpha : 2**{}".format(np.log2(clf.best_params_["alpha"])))
    y_hat_tr = clf.predict(X_tr)
    y_hat_te = clf.predict(X_te)
    print("    TR pearson:{}, spearman:{} MAE:{}".format(
            pearsonr(y_tr, y_hat_tr)[0],
            spearmanr(y_tr, y_hat_tr).correlation,
            np.abs(y_tr-y_hat_tr).mean()))
    print("    TE pearson:{}, spearman:{} MAE:{}".format(
            pearsonr(y_te, y_hat_te)[0],
            spearmanr(y_te, y_hat_te).correlation,
            np.abs(y_te-y_hat_te).mean()))
    

