import pandas as pd
import numpy as np
import pandas_datareader.data as web
import statsmodels.api as sm
import matplotlib.pyplot as plt

series = {'CPMNACSCAB1GQDE':'GDPGermany',
          'LRUNTTTTDEQ156S':'EMPGermany',
          'CPMNACSCAB1GQPL':'GDPPoland',
          'LRUNTTTTPLQ156S':'EMPPoland'}
df = web.DataReader(series.keys(), 'fred', start='1995-01-01', end='2019-10-01')

df = df.rename(series, axis=1)

# 1)
# This data is from lecture 18.  Explore it using plots and summary
# statistics. What is wrong with the employment data from Poland? 
# Then, apply an HP filter from the statsmodels library, and filter 
# all four series.  Plot the cycles, trends, and original values to
# see what is happening when you filter.
df.mean()
#Poland is missing two rows
df = df.reset_index()

df_long = pd.wide_to_long(df, ['GDP', 'EMP'], i='DATE',
                                              j='COUNTRY',
                                              suffix='\\w+')
fig, ax = plt.subplots()
for label, d in df_long.reset_index().groupby('COUNTRY'):
    d.plot(x='DATE', y='GDP', label=label, ax=ax)
ax.legend()
ax.set_title('GDP')

g_cycle1, g_trend1 = sm.tsa.filters.hpfilter(df['GDPGermany'], lamb=1600)
p_cycle1, p_trend1 = sm.tsa.filters.hpfilter(df['GDPPoland'], lamb=1600)

fig, axs = plt.subplots(2, 1, figsize=(12,6))
axs[0].plot(g_cycle1.index, g_cycle1, 'b-', label='Germany')
axs[0].plot(g_cycle1.index, p_cycle1, 'r-', label='Poland')
axs[1].plot(g_trend1.index, g_trend1, 'b-')
axs[1].plot(g_trend1.index, p_trend1, 'r-')
axs[0].set_ylabel('Cycle')
axs[1].set_ylabel('Trend')
fig.legend(loc='upper center', ncols=2)
plt.show()

# 2)
# The code from the lecture includes a function that implements the
# Hamilton filter, though we did not go over the code in detail.
# Copy that function over and try to understand most of what it is
# doing (you may have to test it in pieces) and then apply it to
# this data. Modify your plots from question 1 to compare the results
# of the Hamilton and HP filters to the unfiltered values.
def hamilton_filter(data, h=8, p=4):
    def _shift(orig_series, n):
        #implements efficient (positive) shifting for non-Series dtypes
        new_series = np.empty_like(orig_series)
        new_series[:n] = np.NaN
        new_series[n:] = orig_series[:-n]
        return new_series

    new_cols = [_shift(data, s) for s in range(h, h+p)]

    exog = sm.add_constant(np.array(new_cols).transpose())
    model = sm.GLM(endog=data, exog=exog, missing='drop')
    res = model.fit()

    trend = res.fittedvalues
    rand = data - _shift(data, h)
    cycle = res.resid_pearson
    return cycle, trend, rand

g_cycle2, g_trend2, _ = hamilton_filter(df['GDPGermany'])
p_cycle2, p_trend2, _ = hamilton_filter(df['GDPPoland'])

fig, axs = plt.subplots(4, 1, figsize=(12,6))
axs[0].plot(g_cycle2.index, g_cycle2, 'b-', label='Germany')
axs[0].plot(g_cycle2.index, p_cycle2, 'r-', label='Poland')
axs[1].plot(g_trend2.index, g_trend2, 'b-')
axs[1].plot(g_trend2.index, p_trend2, 'r-')
axs[0].set_ylabel('Hamilton Filter Cycle')
axs[1].set_ylabel('Hamilton Filter Trend')
axs[2].plot(g_cycle1.index, g_cycle1, 'g-', label='Germany')
axs[2].plot(g_cycle1.index, p_cycle1, 'm-', label='Poland')
axs[3].plot(g_trend1.index, g_trend1, 'g-')
axs[3].plot(g_trend1.index, p_trend1, 'm-')
axs[2].set_ylabel('HP Filter Cycle')
axs[3].set_ylabel('HP Filter Trend')
fig.legend(loc='upper center', ncols=2)
plt.show()

g_cycle.name = 'Germany_hamilton'
p_cycle.name = 'Poland_hamilton'
