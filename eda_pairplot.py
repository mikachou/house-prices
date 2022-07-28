"""
Generate a pairplot chart with train values
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('./data/train.csv')
#test = pd.read_csv('./data/test.csv')

sns.set(style="ticks", color_codes=True)
g = sns.pairplot(train, kind="reg", plot_kws={'line_kws':{'color':'red'}})
plt.savefig('eda/pairplot_train.png')
