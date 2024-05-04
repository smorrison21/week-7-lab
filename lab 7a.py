#NAMES

import pandas as pd
import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression 

url_to_csv = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv'
df = pd.read_csv(url_to_csv)


# 1) Explore the data & produce some basic summary stats  

# 2) Run a regression of price (y) on carat (x), including an 
#    intercept term.  Report the estimates of the intercept & slope 
#    coefficients using each of the following methods:
#        a) NumPy
#        b) statsmodels (smf) 
#        c) statsmodels (sm)
#        d) scikit-learn (LinearRegression)  
#           Hint:  scikit-learn only works with array-like objects.    
#    Confirm that all four methods produce the same estimates.

# 3) Run a regression of price (y) on carat, the natual logarithm of depth  
#    (log(depth)), and a quadratic polynomial of table (i.e., include table & 
#    table**2 as regressors).  Estimate the model parameters using any Python
#    method you choose, and display the estimates.  

# 4) Run a regression of price (y) on carat and cut.  Estimate the model 
#    parameters using any Python method you choose, and display the estimates.  

# 5) Run a regression of price (y) on whatever predictors (and functions of 
#    those predictors you want).  Try to find the specification with the best
#    fit (as measured by the largest R-squared).  Note that this type of data
#    mining is econometric blasphemy, but is the foundation of machine
#    learning.  Fit the model using any Python method you choose, and display 
#    only the R-squared from each model.  We'll see who can come up with the 
#    best fit by the end of lab.  
