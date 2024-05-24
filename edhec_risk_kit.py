import pandas as pd

def drawdown(return_series: pd.Series):  # specifying the input as a series, and in particular, as a Pandas series
    """
    Takes a time series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percentage drawdowns
    """
    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks
    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama French dataset for the returns of the equal-weighted portfolios of top and bottom deciles by market cap
    """
    me_m = pd.read_csv("Portfolios_Formed_on_ME_monthly_EW.csv",
                      skiprows=1,
                      header=0, index_col=0, parse_dates=True, na_values=-99.99
                      )
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns = ['SmallCap', 'LargeCap']
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period('M')
    return rets



def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index returns
    """
    hfi = pd.read_csv("edhec-hedgefundindices.csv",
                      sep = ';',
                      header=0, index_col=0, parse_dates=True, dayfirst=True)
    hfi = hfi.drop(columns=['Short Selling']) #this column has some NaN values do we will remove it.
    
    # Function to remove '%' and convert to float
    def remove_percent_and_convert(x):
        return float(x.rstrip('%'))
    
    # Apply the function to all elements in the DataFrame
    hfi = hfi.applymap(remove_percent_and_convert)
    hfi = hfi/100
    hfi.index = hfi.index.to_period('M')
    return hfi

def get_ind_returns():
    """
    Load and format the Ken French 30 Industry Portfolio Value Weighted Monthly Returns
    """
    ind = pd.read_csv("ind30_m_rets.csv",
                  skiprows=11,nrows=1173,
                  header=0,index_col=0,
                  parse_dates=True
                  )/100 #to get the values in decimals
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period('M')
    ind.columns = ind.columns.str.strip() #strip out the blank space in the header names
    return ind

def semideviation(r):
    """
    Returns the semideviation aka the negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r<0
    return r[is_negative].std(ddof=0)

def semideviation3(r):
    """
    Function to compute the semideviation according to the formula explained by Professor Martellini
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population std dev, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """
    demeaned_r = r - r.mean()
    # use the population std dev, so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

import scipy.stats
def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Calculates a tuple of statistic and p-value
    Returns True if the hypothesis of normality is accepted, False otherwise
    """
    statistic, p_value = scipy.stats.jarque_bera(r)
    return p_value > level

import numpy as np
def var_historic(r, level=5):
    """
    Returns the historic Value at Risk at a specified level
    i.e. returns the number so that 'level' percentage of the returns
    fall below that number, and (100 - level) percentage of the returns are above that number
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic,level=level)
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level) # so that the results are output as positive numbers
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    

from scipy.stats import norm
def var_gaussian(r, level=5, modified=False):
    """
    Returns the parametric Gaussian VaR of a Series or DataFrame
    Level is set to 5% by default
    If 'modified'=True, then the modified VaR is returned,
    using the Cornish-Fisher modification
    """
    # compute the Z-score assuming it was Gaussian
    z = norm.ppf(level/100)
    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
             (z**2 - 1)*s/6 +
             (z**3 - 3*z)*(k-3)/24 -
             (2*z**3 - 5*z)*(s**2)/36
             )
    return -(r.mean() + z*r.std(ddof=0))

def cvar_historic(r, level=5):
    """
    Computes the Conditional VaR of the Series or Dataframe
    """
    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level=level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level=level)
    else:
        raise TypeError("Expected r to be a Series or DataFrame")
    
def annualize_rets(r, periods_per_year):
    """
    Returns the annualized return from a return Series r
    We should infer the periods per year, currently left as an exercise for the reader
    """
    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return (compounded_growth**(periods_per_year/n_periods))-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the volatility of a set of returns
    We should infer the periods per year, currently left as an exercise for the reader
    """
    return r.std()*(periods_per_year**(0.5))

def sharpe_ratio(r, risk_free_rate, periods_per_year):
    """
    Computes the annualized Sharpe Ratio of a set of returns
    """
    # convert the annual risk free rate to rate per period
    rf_per_period = (1+risk_free_rate)**(1/periods_per_year)-1
    excess_ret = r - rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Takes a vector of asset weights, and a vector of asset returns, calculates the overall portfolio return
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Takes a vector of asset weights, and a covariance matrix of the assets, calculates the overall portfolio std dev
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_ef2(n_points, er, cov, style = ".-"):
    """
    Plots the efficient frontier for a 2-asset portfolio, with the number of points, expected returns, covariance matrix and plot style as inputs
    """
    if er.shape[0] != 2 or er.shape[0] != 2:
        raise ValueError("plot_ef2 can only plot 2-asset frontiers")
    weights = [np.array([w,1-w]) for w in np.linspace(0,1,n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame(
        {"Returns": rets, "Volatility": vols}
        )
    return ef.plot.line(x = "Volatility", y = "Returns", style = style)

# we will use a quadratic optimizer that is built into scipy
from scipy.optimize import minimize
def minimize_vol(target_return, er, cov):
    """
    target return --> weights
    """
    n = er.shape[0] # number of assets is the same as the number of rows in the expected returns array
    init_guess = np.repeat(1/n, n) #weights are inialized to be equal weights for all n assets
    bounds = ((0.0, 1.0),)*n  # specifies the bounds as an n-tuple of the lower (0%) and upper (100%) bounds for the weights
    return_is_target = {
        'type': 'eq', # equality constraint
        'args': (er,),
        'fun': lambda weights, er: target_return - portfolio_return(weights, er) # lambda function that calculates on the fly if the portfolio return equals the target return
    }
    weights_sum_to_1 = {
        'type': 'eq', # equality constraint
        'fun': lambda weights: np.sum(weights) - 1
    }
    results = minimize(portfolio_vol, init_guess,
                       args=(cov,), method="SLSQP", # quadratic programming optimizer
                       options={'disp': False},  # set display to false to avoid results cluttering the screen
                       constraints=(return_is_target, weights_sum_to_1), # set the constraints
                       bounds=bounds
                       )
    return results.x #passes the results to a variable called x

def optimal_weights(n_points, er, cov):
    """
    Generates a list of weights to run the optimizer on, to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points) # takes a number of returns that lie inbetween the minimum and maximum expected returns
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def msr(risk_free_rate, er, cov):
    """
    Returns the weights of the portfolio that gives you the maximum Sharpe Ratio,
    given the risk free rate, expected returns and covariance matrix
    """
    n = er.shape[0] # number of assets is the same as the number of rows in the expected returns array
    init_guess = np.repeat(1/n, n) #weights are inialized to be equal weights for all n assets
    bounds = ((0.0, 1.0),)*n  # specifies the bounds as an n-tuple of the lower (0%) and upper (100%) bounds for the weights
    
    weights_sum_to_1 = {
        'type': 'eq', # equality constraint
        'fun': lambda weights: np.sum(weights) - 1
    }

    def neg_sharpe_ratio(weights, risk_free_rate, er, cov):
        """
        Returns the negative of the Sharpe Ratio, given weights, exected returns and covariance matrix
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r-risk_free_rate)/vol

    results = minimize(neg_sharpe_ratio, init_guess, #minimize the negative of the Sharpe ratio
                       args=(risk_free_rate, er, cov,), method="SLSQP", # quadratic programming optimizer
                       options={'disp': False},  # set display to false to avoid results cluttering the screen
                       constraints=(weights_sum_to_1), # set the constraints
                       bounds=bounds
                       )
    return results.x #passes the results to a variable called x

def gmv(cov):
    """
    Returns the weights of the global minimum variance portfolio, given the covariance matrix
    """
    n=cov.shape[0] # number of assets obtained by counting number of rows in the covariance matrix
    return msr(0, np.repeat(1,n), cov) # dummy risk free rate of 0, dummy expected returns of 1 for each asset

def plot_ef(n_points, er, cov, show_cml=False, style = ".-", risk_free_rate=0, show_ew=False, show_gmv=False):
    """
    Plots the efficient frontier for a N-asset portfolio, with the number of points, expected returns, covariance matrix and plot style as inputs
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame(
        {"Returns": rets, 
        "Volatility": vols}
        )
    ax = ef.plot.line(x = "Volatility", y = "Returns", style = style)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display EW
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize=10)

    if show_gmv:
        w_gmv = gmv(cov) #GMV only depends on the covariance matrix, we do not need expected return assumptions
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display GMV
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize=10)

    if show_cml:
        # Add the CML to the plot
        ax.set_xlim(left=0)
        w_msr = msr(risk_free_rate, er, cov) #weights of the max Sharpe ratio portfolio
        r_msr = portfolio_return(w_msr, er) #return of the max Sharpe ratio portfolio
        vol_msr = portfolio_vol(w_msr, cov) #vol of the max Sharpe ratio portfolio
        cml_x = [0, vol_msr]
        cml_y = [risk_free_rate, r_msr]
        ax.plot(cml_x, cml_y, color = "green", marker="o", linestyle="dashed",markersize=12, linewidth=2)
    return ax
