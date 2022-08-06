import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PowerTransformer, QuantileTransformer

train = pd.read_csv('./data/train.csv')

bc = PowerTransformer(method='box-cox')
yj = PowerTransformer(method='yeo-johnson')
qn = QuantileTransformer(output_distribution='normal')

y = train['SalePrice'].values.reshape(-1, 1)
y_log = np.log1p(y)
y_bc = bc.fit_transform(y)
y_yj = yj.fit_transform(y)
y_qn = qn.fit_transform(y)

fig, axes = plt.subplots(1, 5)
fig.set_size_inches(24, 6)
variables = [y, y_log, y_bc, y_yj, y_qn]
titles = ['SalePrice', 'log', 'box-cox', 'yeo-johnson', 'quantile-normal']
for i, var in enumerate(variables):
    sns.histplot(var, ax=axes[i], legend=False)
    axes[i].set_title(titles[i])
# sns.displot(y)
fig.savefig('eda/dist_saleprice.png')
