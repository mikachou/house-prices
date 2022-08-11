"""
Generate a pairplot chart with train values
"""

import pandas as pd

train = pd.read_csv('./data/train.csv')

# identify outliers
print('LotFrontage > 300')
print(train[train['LotFrontage'] > 300]['Id'].to_string(index=False))
print('BsmtFinSF1 > 4000')
print(train[train['BsmtFinSF1'] > 4000]['Id'].to_string(index=False))
print('BsmtFinSF2 > 1400')
print(train[train['BsmtFinSF2'] > 1400]['Id'].to_string(index=False))
print('TotalBsmtSF > 6000')
print(train[train['TotalBsmtSF'] > 6000]['Id'].to_string(index=False))
print('1stFlrSF > 4000')
print(train[train['1stFlrSF'] > 4000]['Id'].to_string(index=False))
print('EnclosedPorch > 400')
print(train[train['EnclosedPorch'] > 400]['Id'].to_string(index=False))
print('MiscVal > 5000')
print(train[train['MiscVal'] > 5000]['Id'].to_string(index=False))
print('GrLivArea > 5000')
print(train[train['GrLivArea'] > 5000]['Id'].to_string(index=False))

# Salesprices
print('Salesprices')
print(train[train['SalePrice'] > 500000][['Id', 'SalePrice']]
    .sort_values(by='SalePrice', ascending=False).to_string(index=False))

# sns.set(style="ticks", color_codes=True)
# g = sns.pairplot(train, kind="reg", plot_kws={'line_kws':{'color':'red'}})
# plt.savefig('eda/pairplot_train.png')
