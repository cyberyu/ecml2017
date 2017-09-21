# ecml2017

Synopsis

Rank Score + Isotonic Regression model. This repo contains code and data described in Predicting Self-reported Customer Satisfaction of Interactions with a Corporate Call Center Bockhorst et al., European Conference on Machine Learning (ECML 2017)

Code Example

```
import numpy as np
from rsir import RankScoreIsoRegression
from ecml_2017 import load_data
X_tr, X_te, y_tr, y_te = load_data()
pr_args = { 
   "verbose":False,
   "n_training_samples":50000, 
   "n_tuning_training_samples":50000,
   "n_tuning_eval_samples":10000,
   "seed" : 1}
rsir = RankScoreIsoRegression(pr_args=pr_args)
rsir.fit(X_tr, y_tr)
y_hat_te = rsir.predict(X_te)
print("Test set mean absolute error: {}".format(np.abs(y_te-y_hat_te).mean()))
...
Test set mean absolute error: 0.775229527959
```

Dependencies

Python 2.7.12 External packages: numpy, scipy, sklearn:

```
print(sklearn.__version__)
0.19.0

print(np.__version__)
1.13.1

print(scipy.__version__)
0.19.1
```

Installation

```
> git clone https://github.com/cyberyu/ecml2017
> cd ecml2017/code
> python ecml_2017.py

This example reproduces the table in Figure 3 of 
'Predicting Self-reported Customer Satisfaction of
Interactions with a Corporate Call Center'.
by J. Bockhorst, S. Yu, L. Polania and G.Fung. ECML 2017

data loaded
X_tr shape: (6108, 5501)
X_te shape: (2618, 5501)
y_tr shape: (6108,)
y_te shape: (2618,)

Training Rank Score + Isotonic Regression model
RS
    TR pearson:0.368093643377, spearman:0.400813176115
    TE pearson:0.274846440108, spearman:0.244495675146
RS+IR
    TR pearson:0.429072621209, spearman:0.403490123441 MAE:0.709085031063
    TE pearson:0.327599430132, spearman:0.245288627338 MAE:0.775229527959
LASSO
    Best alpha : 2**-4.0
    TR pearson:0.293582558632, spearman:0.229352880118 MAE:0.804379343987
    TE pearson:0.303111493587, spearman:0.229961530764 MAE:0.818367958069
RIDGE
    Best alpha : 2**15.0
    TR pearson:0.615138530731, spearman:0.459165058423 MAE:0.705994009972
    TE pearson:0.286084145308, spearman:0.212752112383 MAE:0.809126138687
    
```

License

BSD 2-Clause License
