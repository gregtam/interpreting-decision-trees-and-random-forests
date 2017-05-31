from IPython.display import display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import seaborn as sns
from treeinterpreter import treeinterpreter as ti

blue, green, red, purple, yellow, cyan = sns.color_palette()

def plot_top_feat_contrib(clf, contrib_df, features_df, labels, index,
                          num_features=20, order_by='natural', violin=False):
    """Plots the top features and their contributions for a given
    observation.

    Inputs:
    clf - The Decision Tree or Random Forest classifier object
    contrib_df - A Pandas DataFrame of the feature contributions
    features_df - A Pandas DataFrame with the features
    labels - A Pandas Series of the labels
    index - An integer representing which observation we would like to
            look at
    num_features - The number of features we wish to plot
    order_by - What to order the contributions by. (Default: 'natural')
               The default ordering is the natural one, which takes the
               original feature ordering

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

    obs_contrib_head = obs_contrib_df.tail(num_features)

    fig, ax = plt.subplots()
    if violin:
        plt.violinplot([contrib_df[i] for i in obs_contrib_head.index],
                       vert=False
                      )
        plt.scatter(obs_contrib_head.contrib, 
                    np.arange(obs_contrib_head.shape[0]) + 1, 
                    c=red, 
                    s=100 
                   )
        plt.yticks(np.arange(obs_contrib_head.shape[0]) + 1,
                   obs_contrib_head.index
                  )
    else:
        obs_contrib_head['contrib'].plot(kind='barh', ax=ax)

    plt.axvline(0, c='black', linestyle='--')

    true_label = labels.iloc[index]
    score = clf.predict_proba(features_df.iloc[index:index+1])[0][1]
    plt.title('Label: {}; Probability: {:1.3f}'.format(true_label, score))

    x_coord = ax.get_xlim()[0]
    for y_coord, feat_val in enumerate(obs_contrib_head['feat_val']):
        t = plt.text(x_coord, y_coord + 1, feat_val)
        t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=blue))

    # Returns in reverse order because it needed to be reversed to plot
    # properly
    return obs_contrib_df.iloc[::-1]

def plot_single_feat_contrib(feat_name, features_df, contrib_df, **kwargs):
    """Plots a single feature's values across all observations against
    their corresponding contributions.

    Inputs:
    feat_name - The name of the feature
    features_df - A Pandas DataFrame that includes the feature values
    contrib_df - A Pandas DataFrame that has the contributions
    """

    _temp_df = pd.DataFrame({'feat_value': features_df[feat_name].tolist(),
                             'contrib': contrib_df[feat_name].tolist()
                            })
    _temp_df\
        .sort_values('feat_value')\
        .plot(x='feat_value', y='contrib', kind='scatter', **kwargs)
    plt.axhline(0, c='black', linestyle='--')

    plt.title('Conribution of {}'.format(feat_name))
    plt.xlabel(feat_name)
    plt.ylabel('Contribution')




