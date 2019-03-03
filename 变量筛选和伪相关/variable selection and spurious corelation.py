# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 10:18:18 2019

@author: ranjing
"""


import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签 
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号 
import seaborn as sns

from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
#from sklearn.model_selection import GridSearchCV
import pandas as pd
import warnings 
warnings.filterwarnings("ignore")


# =============================================================================
# 最大相关系数的密度函数与p的关系
# =============================================================================
n = 100
corelations = []
noise_ps = [10, 50, 100, 500, 1000, 2000]
for p in noise_ps:
    corelations.append([])
    for _ in range(100):
        y = np.random.rand(n)
        X = np.random.rand(n, p)
        max_corr = np.max([np.abs(pearsonr(y, X[:, i])[0]) for i in range(p)])
        corelations[-1].append(max_corr)

plt.figure(figsize=(12, 8))
for i, p in enumerate(noise_ps):
    sns.distplot(corelations[i], hist=False, label=str(p))
plt.tight_layout()
plt.savefig('伪相关密度函数.png', dpi=100)



# =============================================================================
# 无关变量对模型的可靠性的影响
# =============================================================================
# 数据
diabetes = load_diabetes()
diabetes_X = diabetes.data
diabetes_y = diabetes.target


# 模型
lasso = Lasso(alpha=0.1)
ridge = Ridge(alpha=0.1)
clf = LinearRegression()
rf = RandomForestRegressor(n_estimators = 500)

score = np.around(np.sqrt(-np.mean(cross_val_score(lasso, diabetes_X, diabetes_y, scoring='neg_mean_squared_error'))), 2)
print('raw score with lasso: %s' % score)

score = np.around(np.sqrt(-np.mean(cross_val_score(ridge, diabetes_X, diabetes_y, scoring='neg_mean_squared_error'))), 2)
print('raw score with ridge: %s' % score)

score = np.around(np.sqrt(-np.mean(cross_val_score(clf, diabetes_X, diabetes_y, scoring='neg_mean_squared_error'))), 2)
print('raw score with linear model: %s' % score)


# Lasso: 训练效果与噪声维度的关系
noise_ps = [10, 50, 100, 500, 1000, 2000]
for p in noise_ps:
    noise_X = np.random.rand(diabetes_X.shape[0], p)
    X = np.column_stack((diabetes_X, noise_X))
    score = np.around(np.sqrt(-np.mean(cross_val_score(lasso, X, diabetes_y, scoring='neg_mean_squared_error'))), 2)
    print('p:%s, score:%s' % (p, score))    


# RandomForest: 训练效果与噪声维度的关系
noise_ps = [10, 20, 50, 100, 500, 1000, 2000]
rf_errors = []
for p in noise_ps:
    noise_X = np.random.rand(diabetes_X.shape[0], p)
    X = np.column_stack((diabetes_X, noise_X))
    error = np.around(np.sqrt(-np.mean(cross_val_score(lasso, X, diabetes_y, scoring='neg_mean_squared_error'))), 2)
    rf_errors.append(error) 

plt.figure(figsize = (12, 7))
plt.plot(noise_ps, rf_errors, 'o-')
plt.xlabel('噪声维度')
plt.ylabel('标准误差')
plt.tight_layout()
plt.savefig('随机森林：误差 vs 噪声维度.png', dpi=150) 


##  # 错误的做法
#params = {'alpha': [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]}
#noise_ps = [10, 50, 100, 500, 1000, 2000]
#clf = GridSearchCV(lasso, param_grid = params, cv=5, scoring ='neg_mean_squared_error')
#
#results = []
#for p in noise_ps:
#    noise_X = np.random.rand(diabetes_X.shape[0], p)
#    X = np.column_stack((diabetes_X, noise_X))
#    
#    clf.fit(X, diabetes_y)
#    score = np.around(np.sqrt(-clf.best_score_), 2)
#    selected_p = (clf.best_estimator_.coef_ !=0).sum()
#    alpha = clf.best_params_['alpha']
#    print('p:%s, alpha:%s, selected variables:%s, score1:%s' % (p, alpha, selected_p, score))
#    result = ['所有变量', p, alpha, selected_p, score]
#    results.append(result)
#
#    # 根据相关性初筛一半变量出去
#    abs_corrs = [np.abs(pearsonr(diabetes_y, X[:, i])[0]) for i in range(X.shape[1])]
#    thresh_corr = np.median(abs_corrs)
#    idx = [i for i in range(X.shape[1]) if abs_corrs[i] >= thresh_corr]
#    selected_X = X[:, idx]
#    clf.fit(selected_X, diabetes_y)
#    score = np.around(np.sqrt(-clf.best_score_), 2)
#    selected_p = (clf.best_estimator_.coef_ !=0).sum()
#    alpha = clf.best_params_['alpha']
#    print('p:%s, alpha:%s, selected variables:%s, score:%s' % (p, alpha, selected_p, score))
#    result = ['初筛一半变量', p, alpha, selected_p, score]
#    results.append(result)
#
#    # 根据相关性保留20个变量
#    abs_corrs = [np.abs(pearsonr(diabetes_y, X[:, i])[0]) for i in range(X.shape[1])]
#    thresh_corr = abs_corrs[np.argsort(abs_corrs)[-20]]
#    idx = [i for i in range(X.shape[1]) if abs_corrs[i] >= thresh_corr]
#    selected_X = X[:, idx]
#    clf.fit(selected_X, diabetes_y)
#    score = np.around(np.sqrt(-clf.best_score_), 2)
#    selected_p = (clf.best_estimator_.coef_ !=0).sum()
#    alpha = clf.best_params_['alpha']
#    print('p:%s, alpha:%s, selected variables:%s, score:%s' % (p, alpha, selected_p, score))
#    result = ['初筛20个变量', p, alpha, selected_p, score]
#    results.append(result)



# 正确的做法
alphas = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
noise_ps = [10, 50, 100, 500, 1000, 2000]

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
kf = KFold(n_splits=5)


results = []
for p in noise_ps:
    noise_X = np.random.rand(diabetes_X.shape[0], p)
    X = np.column_stack((diabetes_X, noise_X))
    # 定义模型
    errors_alpha = []
    for alpha in alphas:
        clf = Lasso(alpha=alpha)
        # 交叉验证
        errors_cv = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = diabetes_y[train_index], diabetes_y[test_index]
            
            # 直接用Lasso选择
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            error1 = np.sqrt(mean_squared_error(y_test, y_pred))
        
            # 根据相关性初筛一半变量出去
            abs_corrs = [np.abs(pearsonr(y_train, X_train[:, i])[0]) for i in range(X_train.shape[1])]
            thresh_corr = np.median(abs_corrs)
            idx = [i for i in range(X_train.shape[1]) if abs_corrs[i] >= thresh_corr]
            selected_X_train = X_train[:, idx]
            selected_X_test = X_test[:, idx]
            clf.fit(selected_X_train, y_train)
            y_pred = clf.predict(selected_X_test)
            error2 = np.sqrt(mean_squared_error(y_test, y_pred))            
            
            # 根据相关性保留20个变量
            abs_corrs = [np.abs(pearsonr(y_train, X_train[:, i])[0]) for i in range(X_train.shape[1])]
            thresh_corr = abs_corrs[np.argsort(abs_corrs)[-20]]
            idx = [i for i in range(X_train.shape[1]) if abs_corrs[i] >= thresh_corr]
            selected_X_train = X_train[:, idx]
            selected_X_test = X_test[:, idx]
            clf.fit(selected_X_train, y_train)
            y_pred = clf.predict(selected_X_test)
            error3 = np.sqrt(mean_squared_error(y_test, y_pred)) 
            
            # 记录误差
            errors_cv.append([error1, error2, error3])
        errors_cv = np.array(errors_cv).mean(0)
        errors_alpha.append(errors_cv.tolist())
    errors_alpha = np.array(errors_alpha).min(0)
    results.append(errors_alpha)
            

# 数据友好
results = pd.DataFrame(results)
results.index = noise_ps
results.columns = ['保留所有变量', '保留一半变量', '保留20个变量']


# =============================================================================
# 可视化
# =============================================================================
plt.figure(figsize = (12, 7))
plt.plot(results, 'o-')
plt.legend(['保留所有变量', '保留一半变量', '保留20个变量'])
plt.ylabel('标准误差')
plt.xlabel('噪声维度')
plt.tight_layout()
plt.savefig('三种模型误差比较.png', dpi=150)

