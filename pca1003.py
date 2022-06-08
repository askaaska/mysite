#%%
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 19:44:35 2019
@author: J0118314

主成分分析の勉強用です。
以下にURLを貼り付けます。
https://qiita.com/maskot1977/items/082557fcda78c4cdb41f
"""
# 数値計算やデータフレーム操作に関するライブラリをインポートする

import sys
print(sys.executable)

import numpy as np
import pandas as pd
import os, tkinter, tkinter.filedialog, tkinter.messagebox
import datetime
import seaborn
import matplotlib as plt # 追加
import sklearn #機械学習のライブラリ
from sklearn.decomposition import PCA #主成分分析器

#現在の時刻
today = datetime.date.today()
todaydetail = datetime.datetime.today()
today_time = str(today) +'_' +str(todaydetail.hour) + '_' + str(todaydetail.minute) + '_'

#実行ファイルのパス取得
import os
cd = (os.getcwd())

#フォルダー作成
result_dir = cd + "\_" + today_time + "PCA"
os.mkdir(result_dir)
print("0-1")
#選択したファイルをinput_fileに格納
input_file = "train.csv"
print("0-2")
#ファイル名の取得
filename = os.path.basename(input_file)
filename = (filename.split('.')[0])
filename = str(filename)

# データセットのロード
df = pd.read_csv(input_file)
# df = df.dropna()
dfcorr = df
print("1")
#説明変数に相関があるか（次元圧縮が有効そうか）
plt.figure(figsize=(10, 10))
#annot:数字ヒートマップに入力　fmt：有効数字　square:図を正方形にする
seaborn.heatmap(dfcorr.corr(), annot=True,vmin=-1,center = 0,vmax=1,square=True,fmt = '.1g')
filename = result_dir + "/" + today_time  + '相関係数ヒートマップ.png'
plt.savefig(filename)
plt.close()
dfcorr.corr().to_excel(result_dir + "/" + today_time  + '相関係数行列.xlsx',sheet_name='corr')
print("2")
#散布図行列
# from pandas import plotting 
# plotting.scatter_matrix(df.iloc[:, 1:], figsize=(8, 8), c=list(df.iloc[:, 0]), alpha=0.5)
# filename = result_dir + "/" + today_time  + '散布図行列.png'
# plt.savefig(filename)

# 行列の標準化
dfs = df.iloc[:, 1:].apply(lambda x: (x-x.mean())/x.std(), axis=0).fillna(0)
dfs = dfs.fillna(0)#nanをゼロで置換
filename = result_dir + "/" + today_time  + '標準化データ.xlsx'
dfs.to_excel(filename)
print("3")
#主成分分析の実行
pca = PCA()
feature = pca.fit(dfs)
# データを主成分空間に写像
feature = pca.transform(dfs)
print("4")
# 第一主成分と第二主成分でプロットする
plt.figure(figsize=(6, 6))
plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=list(df.iloc[:, 0]))
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
filename = result_dir + "/" + today_time  + 'PC1_PC2.png'
plt.savefig(filename)
plt.close()
print("5")
pca_features = pd.DataFrame(feature)
filename = result_dir + "/" + today_time  + 'pca_features.csv'
pca_features.to_csv(filename,index = None)

# 寄与率
kiyoritu_list = pd.DataFrame(pca.explained_variance_ratio_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
print("6")
# 累積寄与率を図示する
import matplotlib.ticker as ticker
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
filename = result_dir + "/" + today_time  + '累積寄与率.png'
plt.savefig(filename)
plt.close()

plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("Number of principal components")
plt.ylabel("Cumulative contribution rate")
plt.grid()
filename = result_dir + "/" + today_time  + '累積寄与率.png'
plt.savefig(filename)
plt.close()
print("7")
# PCA の固有値
koyuchi = pd.DataFrame(pca.explained_variance_, index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
filename = result_dir + "/" + today_time  + '固有値.xlsx'
koyuchi.to_excel(filename)

# PCA の固有ベクトル
koyubekutoru = pd.DataFrame(pca.components_, columns=df.columns[1:], index=["PC{}".format(x + 1) for x in range(len(dfs.columns))])
filename = result_dir + "/" + today_time  + '固有ベクトル.xlsx'
koyubekutoru.to_excel(filename)

# 第一主成分と第二主成分における観測変数の寄与度をプロットする
plt.figure(figsize=(6, 6))
for x, y, name in zip(pca.components_[0], pca.components_[1], df.columns[1:]):
    plt.text(x, y, name)
plt.scatter(pca.components_[0], pca.components_[1], alpha=0.8)
plt.grid()
plt.xlabel("PC1")
plt.ylabel("PC2")
filename = result_dir + "/" + today_time  + '固有ベクトル.png'
plt.savefig(filename)
plt.close()