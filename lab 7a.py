#NAMES

import pandas as pd
import numpy as np 
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression 
url_to_csv = 'https://raw.githubusercontent.com/mwaskom/seaborn-data/master/diamonds.csv'
df = pd.read_csv(url_to_csv)


# 1) Explore the data & produce some basic summary stats  
df.describe()

# 2) Run a regression of price (y) on carat (x), including an 
#    intercept term.  Report the estimates of the intercept & slope 
#    coefficients using each of the following methods:
#        a) NumPy
#        b) statsmodels (smf) 
#        c) statsmodels (sm)
#        d) scikit-learn (LinearRegression)  
#           Hint:  scikit-learn only works with array-like objects.    
#    Confirm that all four methods produce the same estimates.
#NumPy
x = df[['carat']]
y = df['price']
m, b = np.polyfit(x, y, deg=1)
m #slope
b #intercept
#smf
model = smf.ols('price ~ carat', data=df)
result = model.fit()
rs = result.summary()
rs
#sm
x_sm = sm.add_constant(df['carat'])
model = sm.OLS(df['price'], x_sm)
result = model.fit()
result.summary()
#LinearRegression    
models = LinearRegression()
X = df[['carat']]
y = df['price']
models.fit(X, y)
models.intercept_
models.coef_[0]


# 3) Run a regression of price (y) on carat, the natual logarithm of depth  
#    (log(depth)), and a quadratic polynomial of table (i.e., include table & 
#    table**2 as regressors).  Estimate the model parameters using any Python
#    method you choose, and display the estimates.  
df['log_depth'] = np.log10(df['depth'])
model3 = smf.ols('price ~ carat + log_depth + table + table**2', data=df)
result3 = model3.fit()
rs3= result3.summary()
rs3
result3.rsquared
# 4) Run a regression of price (y) on carat and cut.  Estimate the model 
#    parameters using any Python method you choose, and display the estimates.  
model4 = smf.ols('price ~ carat + cut', data=df)
result4 = model4.fit()
rs4 = result4.summary()
result4.rsquared
# 5) Run a regression of price (y) on whatever predictors (and functions of 
#    those predictors you want).  Try to find the specification with the best
#    fit (as measured by the largest R-squared).  Note that this type of data
#    mining is econometric blasphemy, but is the foundation of machine
#    learning.  Fit the model using any Python method you choose, and display 
#    only the R-squared from each model.  We'll see who can come up with the 
#    best fit by the end of lab.  
model5 = smf.ols('price ~ carat + log_depth + table + table**2 + clarity + x + y + z + cut', data=df)
result5 = model5.fit()
rs5 = result5.summary()
rs5
result5.rsquared
result5.rsquared
