from __future__ import division

from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import seaborn as sns
from statsmodels.nonparametric.smoothers_lowess import lowess
from treeinterpreter import treeinterpreter as ti

blue, green, red, purple, yellow, cyan = sns.color_palette('deep')

def plot_top_feat_contrib(clf, contrib_df, features_df, labels, index,
                          num_features=None, order_by='natural', violin=False):
    """Plots the top features and their contributions for a given
    observation.

    Inputs:
    clf - A Decision Tree or Random Forest classifier object
    contrib_df - A Pandas DataFrame of the feature contributions
    features_df - A Pandas DataFrame with the features
    labels - A Pandas Series of the labels
    index - An integer representing which observation we would like to
            look at
    num_features - The number of features we wish to plot. If None, then
                   plot all features (Default: None)
    order_by - What to order the contributions by. The default ordering
               is the natural one, which takes the original feature
               ordering. (Options: 'natural', 'contribution')

    Returns:
    obs_contrib_df - A Pandas DataFrame that includes the feature values
                     and their contributions
    """

    if order_by not in ['natural', 'contribution']:
        raise Exception('order_by must be either natural or contribution.')

    feature_array = features_df.iloc[index]
    contrib_array = contrib_df.iloc[index]

    obs_contrib_df = pd.DataFrame({'feat_val': feature_array,
                                   'contrib': contrib_array
                                  })
    # Flip rows vertically so that column names are in the same order as
    # the original data set.
    obs_contrib_df = obs_contrib_df.iloc[::-1, :]

    obs_contrib_df['abs_contrib'] = np.abs(obs_contrib_df['contrib'])
    if order_by == 'contribution':
        obs_contrib_df.sort_values('abs_contrib', inplace=True)

    # Trim the contributions if num_features is specified
    if num_features is not None:
        obs_contrib_head = obs_contrib_df.tail(num_features).copy()
    else:
        obs_contrib_head = obs_contrib_df.copy()

    fig, ax = plt.subplots()
    if violin:
        plt.violinplot([contrib_df[i] for i in obs_contrib_head.index],
                       vert=False,
                       positions=np.arange(len(obs_contrib_head))
                      )
        plt.scatter(obs_contrib_head.contrib, 
                    np.arange(obs_contrib_head.shape[0]), 
                    c=red, 
                    s=100 
                   )
        plt.yticks(np.arange(obs_contrib_head.shape[0]),
                   obs_contrib_head.index
                  )
    else:
        obs_contrib_head['contrib'].plot(kind='barh', ax=ax)

    plt.axvline(0, c='black', linestyle='--', linewidth=2)

    true_label = labels.iloc[index]
    if isinstance(clf, DecisionTreeClassifier)\
            or isinstance(clf, RandomForestClassifier):
        score = clf.predict_proba(features_df.iloc[index:index+1])[0][1]
        plt.title('True Value: {}; Score: {:1.3f}'.format(true_label, score))
    elif isinstance(clf, DecisionTreeRegressor)\
            or isinstance(clf, RandomForestRegressor):
        pred = clf.predict(features_df.iloc[index:index+1])[0]
        plt.title('True Value: {}; Predicted Value: {:1.3f}'.format(true_label, pred))
    plt.xlabel('Contribution of feature')

    x_coord = ax.get_xlim()[0]
    for y_coord, feat_val in enumerate(obs_contrib_head['feat_val']):
        t = plt.text(x_coord, y_coord, feat_val)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=blue))

    # Returns in reverse order because it needed to be reversed to plot
    # properly
    return obs_contrib_df.iloc[::-1]

def plot_single_feat_contrib(feat_name, features_df, contrib_df,
                             add_smooth=False, frac=2/3, **kwargs):
    """Plots a single feature's values across all observations against
    their corresponding contributions.

    Inputs:
    feat_name - The name of the feature
    features_df - A Pandas DataFrame that includes the feature values
    contrib_df - A Pandas DataFrame that has the contributions
    add_smooth - Add a lowess smoothing trend line (Default: False)
    frac - The fraction of data used when estimating each y-value
           (Default: 0.666666666)
    """

    plot_df = pd.DataFrame({'feat_value': features_df[feat_name].tolist(),
                            'contrib': contrib_df[feat_name].tolist()
                           })
    plot_df\
        .sort_values('feat_value')\
        .plot(x='feat_value', y='contrib', kind='scatter', **kwargs)

    if add_smooth:
        # Gets lowess fit points
        x_l, y_l = lowess(plot_df.contrib, plot_df.feat_value, frac=frac).T
        # Overlays lowess curve onto data
        plt.plot(x_l, y_l, c='black')

    plt.axhline(0, c='black', linestyle='--', linewidth=2)

    plt.title('Conribution of {}'.format(feat_name))
    plt.xlabel(feat_name)
    plt.ylabel('Contribution')
