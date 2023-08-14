import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.metrics import roc_curve,roc_auc_score
from sklearn.calibration import calibration_curve

def cdf(sample, x, sort = False):
    '''
    Return the value of the Cumulative Distribution Function, evaluated for a given sample and a value x.
    
    Args:
        sample: The list or array of observations.
        x: The value for which the numerical cdf is evaluated.
    
    Returns:
        cdf = CDF_{sample}(x)
    '''
    
    # Sorts the sample, if needed
    if sort:
        sample.sort()
    
    # Counts how many observations are below x
    cdf = sum(sample <= x)
    
    # Divides by the total number of observations
    cdf = cdf / len(sample)
    
    return cdf

def model_evaluation(real:np.array,predicted:np.array)-> dict:
    """_summary_

    Args:
        real (np.array): _description_
        predicted (np.array): _description_

    Returns:
        dict: _description_
    """

    df = pd.DataFrame(data={'values':real,'predicted':predicted})
    goods = df[df['values']==0]['predicted']
    bads = df[df['values']==1]['predicted']

    #AUCROC
    fpr, tpr, _ = roc_curve(df['values'].values,  df['predicted'].values)
    auc = roc_auc_score(df['values'].values,  df['predicted'].values)
    plt.plot(fpr,tpr)
    plt.title(f"AUC: {np.round(auc,4)}")
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()

    #Kolmogorov-Smirnov test
    cfd_goods = np.array([cdf(goods.values, x, sort = False) for x in goods])
    cfd_bads = np.array([cdf(bads.values, x, sort = False) for x in bads])
    ks = stats.ks_2samp(goods, bads)
    
    sns.lineplot(x=goods, y = cfd_goods, color = 'b')
    sns.lineplot(x=bads, y = cfd_bads, color = 'r')
    plt.title(f'KS: {np.round(ks.statistic,4)}')
    plt.show()

    #separation
    sns.histplot(goods, bins = 20, kde = False, color = 'g')
    sns.histplot(bads, bins = 20, kde = False, color = 'b')
    plt.legend(['goods','bads'])
    plt.show()

    #Calibration curve
    x, y = calibration_curve(df['values'], df['predicted'], n_bins = 10)
 
    # Plot calibration curve
    plt.plot([0, 1], [0, 1], linestyle = '--', label = 'Ideally Calibrated')
    
    # Plot model's calibration curve
    plt.plot(y, x, marker = '.', label = 'Model')
    
    leg = plt.legend(loc = 'upper left')
    plt.xlabel('Average Predicted Probability in each bin')
    plt.ylabel('Ratio of positives')
    plt.show()