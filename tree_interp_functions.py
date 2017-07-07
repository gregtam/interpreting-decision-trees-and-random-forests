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

sns.set_palette('colorblind')
blue, green, red, purple, yellow, cyan = sns.color_palette('colorblind')

def plot_obs_feature_contrib(clf, contributions, features_df, labels, index, 
                             class_index=0, num_features=None,
                             order_by='natural', violin=False):
    """Plots a single observation's feature contributions.

    Inputs:
    clf - A Decision Tree or Random Forest classifier object
    contributions - The contributions from treeinterpreter
    features_df - A Pandas DataFrame with the features
    labels - A Pandas Series of the labels
    index - An integer representing which observation we would like to
            look at
    class_index - The index of which class to look at (Default: 0)
    num_features - The number of features we wish to plot. If None, then
                   plot all features (Default: None)
    order_by - What to order the contributions by. The default ordering
               is the natural one, which takes the original feature
               ordering. (Options: 'natural', 'contribution')
    violin - Whether to plot violin plots (Default: False)
    
    Returns:
    obs_contrib_df - A Pandas DataFrame that includes the feature values
                     and their contributions
    """

    def _extract_contrib_array():
        # If regression tree
        if len(contributions.shape) == 2:
            contrib_array = contributions[index]
        # If classification tree
        elif len(contributions.shape) == 3:
            if class_index >= contributions.shape[2]:
                raise Exception('class_index exceeds number of classes.')
            contrib_array = contributions[index, :, class_index]
        else:
            raise Exception('contributions is not the right shape.')    

        return contrib_array

    def _plot_contrib():
        """Plot contributions for a given observation. Also plot violin
        plots for all other observations if specified.
        """
        fig, ax = plt.subplots()
        if violin:
            # Get contributions for the class
            if len(contributions.shape) == 2:
                contrib = contributions
            elif len(contributions.shape) == 3:
                contrib = contributions[:, :, class_index]

            contrib_df = pd.DataFrame(contrib, columns=features_df.columns)
            # Plot a violin plot using only variables in obs_contrib_tail
            plt.violinplot([contrib_df[w] for w in obs_contrib_tail.index],
                           vert=False,
                           positions=np.arange(len(obs_contrib_tail))
                          )
            plt.scatter(obs_contrib_tail.contrib, 
                        np.arange(obs_contrib_tail.shape[0]), 
                        color=red, 
                        s=100
                       )
            plt.yticks(np.arange(obs_contrib_tail.shape[0]),
                       obs_contrib_tail.index
                      )
        else:
            obs_contrib_tail['contrib'].plot(kind='barh', ax=ax)

        plt.axvline(0, c='black', linestyle='--', linewidth=2)

        x_coord = ax.get_xlim()[0]
        for y_coord, feat_val in enumerate(obs_contrib_tail['feat_val']):
            t = plt.text(x_coord, y_coord, feat_val)
            t.set_bbox(dict(facecolor='white', alpha=0.5, edgecolor=blue))

    def _edit_axes():
        plt.xlabel('Contribution of feature')
        true_label = labels.iloc[index]
        if isinstance(clf, DecisionTreeClassifier)\
                or isinstance(clf, RandomForestClassifier):
            scores = clf.predict_proba(features_df.iloc[index:index+1])[0]
            scores = [float('{:1.3f}'.format(i)) for i in scores]
            plt.title('True Value: {}\nScores: {}'
                          .format(true_label, scores[class_index]))
            # Returns obs_contrib_df (flipped back), true labels, and scores 
            return obs_contrib_df.iloc[::1], true_label, scores

        elif isinstance(clf, DecisionTreeRegressor)\
                or isinstance(clf, RandomForestRegressor):
            pred = clf.predict(features_df.iloc[index:index+1])[0]
            plt.title('True Value: {}\nPredicted Value: {:1.3f}'
                          .format(true_label, pred))
            # Returns obs_contrib_df (flipped back), true labels, and scores 
            return obs_contrib_df.iloc[::-1], true_label, pred

    feature_array = features_df.iloc[index]
    contrib_array = _extract_contrib_array()

    obs_contrib_df = pd.DataFrame({'feat_val': feature_array,
                                   'contrib': contrib_array
                                  })
    # Flip DataFrame vertically to plot in same order
    obs_contrib_df = obs_contrib_df.iloc[::-1]

    obs_contrib_df['abs_contrib'] = np.abs(obs_contrib_df['contrib'])
    if order_by == 'contribution':
        obs_contrib_df.sort_values('abs_contrib', inplace=True)

    # Trim the contributions if num_features is specified
    if num_features is not None:
        obs_contrib_tail = obs_contrib_df.tail(num_features).copy()
    else:
        obs_contrib_tail = obs_contrib_df.copy()

    _plot_contrib()
    return _edit_axes()


def plot_single_feat_contrib(feat_name, features_df, contrib_df,
                             add_smooth=False, frac=2/3, class_='', **kwargs):
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

    # Create a DataFrame to plot the contributions
    plot_df = pd.DataFrame({'feat_value': features_df[feat_name].tolist(),
                            'contrib': contrib_df[feat_name].tolist()
                           })

    # Set title according to class_
    if class_ == '':
        title = 'Contribution of {}'.format(feat_name)
    else:
        title = 'Conribution of {} ({})'.format(feat_name, class_)

    # If a matplotlib ax is specified in the kwargs, then set ax to it
    # so we can overlay multiple plots together.
    if 'ax' in kwargs:
        ax = kwargs['ax']
        # If size is not specified, set to default matplotlib size
        if 's' not in kwargs:
            kwargs['s'] = 40
        plot_df\
            .sort_values('feat_value')\
            .plot(x='feat_value', y='contrib', kind='scatter', **kwargs)
        ax.axhline(0, c='black', linestyle='--', linewidth=2)
        ax.set_title(title)
        ax.set_xlabel(feat_name)
        ax.set_ylabel('Contribution')
    else:
        plt.scatter(plot_df.feat_value, plot_df.contrib, **kwargs)
        plt.axhline(0, c='black', linestyle='--', linewidth=2)
        plt.title(title)
        plt.xlabel(feat_name)
        plt.ylabel('Contribution')

    if add_smooth:
        # Gets lowess fit points
        x_l, y_l = lowess(plot_df.contrib, plot_df.feat_value, frac=frac).T
        # Overlays lowess curve onto data
        if 'ax' in kwargs:
            ax.plot(x_l, y_l, c='black')
        else:
            plt.plot(x_l, y_l, c='black')
