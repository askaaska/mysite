# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
import random
import time
import gc
import matplotlib.pylab as plt
import seaborn as sns
"""
インプットファイル読み込み
"""
# #Stage1
# df_TDresults_path = "MDataFiles_Stage1/WNCAATourneyDetailedResults.csv"
# df_seeds_path = "MDataFiles_Stage1/MNCAATourneySeeds.csv"
# df_MMOrdinals = "MDataFiles_Stage1/MMasseyOrdinals.csv"
# test_path = "MDataFiles_Stage1/MSampleSubmissionStage1.csv"
# Stage2
df_TDresults_path = "WDataFiles_Stage2/WNCAATourneyDetailedResults.csv"
df_seeds_path = "WDataFiles_Stage2/WNCAATourneySeeds.csv"
# df_MMOrdinals = "WDataFiles_Stage2/MMasseyOrdinals_thruDay128.csv"
test_path = "WDataFiles_Stage2/WSampleSubmissionStage2.csv"

#ボックススコア(時間別)
df_TDresults = pd.read_csv(df_TDresults_path)
#特徴量の組合せて新たな特徴量を作成
WFgrate = (df_TDresults["WFGM"] / df_TDresults["WFGA"])*100
WFgrate3 = (df_TDresults["WFGM3"] / df_TDresults["WFGA3"])*100
WFTrate = (df_TDresults["WFTM"] / df_TDresults["WFTA"])*100
WFTR = (df_TDresults["WFTA"] / df_TDresults["WFGA"])*100
WORrate = (df_TDresults["WOR"]  / (df_TDresults["WOR"]  + df_TDresults["LDR"]))*100
WDRrate = (df_TDresults["WDR"]  / (df_TDresults["WDR"]  + df_TDresults["LOR"]))*100
WTRB = df_TDresults["WOR"] + df_TDresults["WDR"]
WAstrate = (df_TDresults["WAst"] / df_TDresults["WFGM"])*100
#L
LFgrate = (df_TDresults["LFGM"] / df_TDresults["LFGA"])*100
LFgrate3 = (df_TDresults["LFGM3"] / df_TDresults["LFGA3"])*100
LFTrate = (df_TDresults["LFTM"] / df_TDresults["LFTA"])*100
LFTR = (df_TDresults["LFTA"] / df_TDresults["LFGA"])*100
LORrate = (df_TDresults["LOR"]  / (df_TDresults["LOR"]  + df_TDresults["WDR"]))*100
LDRrate = (df_TDresults["LDR"]  / (df_TDresults["LDR"]  + df_TDresults["WOR"]))*100
LTRB = df_TDresults["LOR"] + df_TDresults["LDR"]
LAstrate = (df_TDresults["LAst"] / df_TDresults["LFGM"])*100
#W&L
WTRBrate = (WTRB / (WTRB + LTRB))*100
LTRBrate = (LTRB / (LTRB + WTRB))*100
#連結
df_newfeautures = pd.concat([WFgrate,WFgrate3,WFTrate,WFTR,WORrate,WDRrate,
                             WTRB,WAstrate,WTRBrate,LFgrate,LFgrate3,LFTrate,LFTR,
                             LORrate,LDRrate,LTRB,LAstrate,LTRBrate
                             ],axis=1)
df_newfeautures.columns = ["WFgrate","WFgrate3","WFTrate","WFTR","WORrate","WDRrate",
                             "WTRB","WAstrate","WTRBrate","LFgrate","LFgrate3","LFTrate","LFTR",
                             "LORrate","LDRrate","LTRB","LAstrate","LTRBrate"]
df_TDresults = pd.concat([df_TDresults,df_newfeautures],axis = 1)
                             
box_col = ["FGM","FGA","FGM3","FTM","FTA","OR","DR","Ast","TO","Stl","Blk","PF",
           "Fgrate","Fgrate3","FTrate","FTR","ORrate","DRrate","TRB","Astrate","TRBrate"]
df_boxW = df_TDresults[["Season","WTeamID"]+["W" + col for col in box_col]]
df_boxL = df_TDresults[["Season","LTeamID"]+["L" + col for col in box_col]]
df_boxW = df_boxW.rename(columns={"WTeamID":"TeamID"})
df_boxW = df_boxW.rename(columns={("W"+ col):col for col in box_col})
df_boxL = df_boxL.rename(columns={"LTeamID":"TeamID"})
df_boxL = df_boxL.rename(columns={("L"+ col):col for col in box_col})
df_box = pd.merge(df_boxW,df_boxL,on = ["Season","TeamID"]+box_col,how="outer")
df_box = df_box.groupby(["Season","TeamID"])[box_col].agg(np.mean).reset_index()
df_box_max = df_box.groupby(["Season","TeamID"])[box_col].agg(np.max).reset_index()#new
df_box_min = df_box.groupby(["Season","TeamID"])[box_col].agg(np.min).reset_index()#new
df_box_median = df_box.groupby(["Season","TeamID"])[box_col].agg(np.median).reset_index()#new

df_TDresults2 = df_TDresults
df_TDresults = df_TDresults.rename(columns={"WTeamID":"Team1ID","LTeamID":"Team2ID","WScore":"T1Score","LScore":"T2Score"})
df_TDresults = df_TDresults.rename(columns={f"W{col}":f"T1{col}" for col in box_col})
df_TDresults = df_TDresults.rename(columns={f"L{col}":f"T2{col}" for col in box_col})
df_TDresults2 = df_TDresults2.rename(columns={"WTeamID":"Team2ID","LTeamID":"Team1ID","WScore":"T2Score","LScore":"T1Score"})

features = ["Season","Team1ID","Team2ID","T1Score","T2Score",'target']
df_TDresults['target'] = 1.0
df_TDresults2['target'] = 0.0
train = pd.merge(df_TDresults,df_TDresults2,on = features,how="outer")
train = train[features]

box_T1 = df_box.copy()
box_T2 = df_box.copy()
box_T1.columns = ['Season','Team1ID'] + ["T1"+col+"_mean" for col in box_col]
box_T2.columns = ['Season','Team2ID'] + ["T2"+col+"_mean" for col in box_col]
train = pd.merge(train,box_T1,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,box_T2,on = ["Season","Team2ID"],how = "left")

box_T3 = df_box_max.copy()
box_T4 = df_box_max.copy()
box_T3.columns = ['Season','Team1ID'] + ["T1"+col+"_max" for col in box_col]
box_T4.columns = ['Season','Team2ID'] + ["T2"+col+"_max" for col in box_col]
train = pd.merge(train,box_T3,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,box_T4,on = ["Season","Team2ID"],how = "left")

box_T5 = df_box_min.copy()
box_T6 = df_box_min.copy()
box_T5.columns = ['Season','Team1ID'] + ["T1"+col+"_min" for col in box_col]
box_T6.columns = ['Season','Team2ID'] + ["T2"+col+"_min" for col in box_col]
train = pd.merge(train,box_T5,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,box_T6,on = ["Season","Team2ID"],how = "left")

box_T7 = df_box_median.copy()
box_T8 = df_box_median.copy()
box_T7.columns = ['Season','Team1ID'] + ["T1"+col+"_median" for col in box_col]
box_T8.columns = ['Season','Team2ID'] + ["T2"+col+"_median" for col in box_col]
train = pd.merge(train,box_T7,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,box_T8,on = ["Season","Team2ID"],how = "left")

#チームのシード情報
df_seeds = pd.read_csv(df_seeds_path)
#前処理
df_seeds["seed"] = df_seeds['Seed'].apply(lambda x: int(x[1:3]))
seeds_T1 = df_seeds[['Season','TeamID','seed']].copy()
seeds_T2 = df_seeds[['Season','TeamID','seed']].copy()
seeds_T1.columns = ['Season','Team1ID','T1_seed']
seeds_T2.columns = ['Season','Team2ID','T2_seed']
train = pd.merge(train,seeds_T1,on = ["Season","Team1ID"],how = "left")
train = pd.merge(train,seeds_T2,on = ["Season","Team2ID"],how = "left")
train["seeddiff"] = train["T1_seed"] - train["T2_seed"]

#各種ランキング
# df_MMOrdinals = pd.read_csv(df_MMOrdinals)
# #前処理
# df_rank_mean = df_MMOrdinals.groupby(["Season","TeamID"])["OrdinalRank"].agg(np.mean).reset_index()
# df_rankmax  = df_MMOrdinals.groupby(["Season","TeamID"])["OrdinalRank"].agg(np.max).reset_index()#new
# df_rankmin = df_MMOrdinals.groupby(["Season","TeamID"])["OrdinalRank"].agg(np.min).reset_index()#new
# df_rankmedian = df_MMOrdinals.groupby(["Season","TeamID"])["OrdinalRank"].agg(np.median).reset_index()#new
# df_rank = pd.concat([df_rank_mean,df_rankmax.iloc[:,2],df_rankmin.iloc[:,2],df_rankmedian.iloc[:,2]],axis = 1)
# df_rank.columns = ["Season","TeamID","rank_mean","rank_max","rank_min","rank_median"]

# ranks_T1 = df_rank.copy()
# ranks_T2 = df_rank.copy()
# ranks_T1.columns = ['Season','Team1ID','T1_rank_mean','T1_rank_max','T1_rank_min','T1_rank_median']
# ranks_T2.columns = ['Season','Team2ID','T2_rank_mean','T2_rank_max','T2_rank_min','T2_rank_median']
# train = pd.merge(train,ranks_T1,on = ["Season","Team1ID"],how = "left")
# train = pd.merge(train,ranks_T2,on = ["Season","Team2ID"],how = "left")
# train["rank_mean_diff"] = train["T1_rank_mean"] - train["T2_rank_mean"]
# train["rank_max_diff"] = train["T1_rank_max"] - train["T2_rank_max"]
# train["rank_min_diff"] = train["T1_rank_min"] - train["T2_rank_min"]
# train["rank_median_diff"] = train["T1_rank_median"] - train["T2_rank_median"]

#各年のブラケット構造
test = pd.read_csv(test_path)

#IDをアンダーバー区切り、SeasonやTeamIDとして保存
test["Season"] = test['ID'].apply(lambda x: int(x[0:4]))
test["Team1ID"] = test['ID'].apply(lambda x: int(x[5:9]))
test["Team2ID"] = test['ID'].apply(lambda x: int(x[10:14]))

test = pd.merge(test,box_T1,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,box_T2,on = ["Season","Team2ID"],how = "left")
#max min median追加
test = pd.merge(test,box_T3,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,box_T4,on = ["Season","Team2ID"],how = "left")
test = pd.merge(test,box_T5,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,box_T6,on = ["Season","Team2ID"],how = "left")
test = pd.merge(test,box_T7,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,box_T8,on = ["Season","Team2ID"],how = "left")
test = pd.merge(test,seeds_T1,on = ["Season","Team1ID"],how = "left")
test = pd.merge(test,seeds_T2,on = ["Season","Team2ID"],how = "left")
test["seeddiff"] = test["T1_seed"] - test["T2_seed"]

# test = pd.merge(test,ranks_T1,on = ["Season","Team1ID"],how = "left")
# test = pd.merge(test,ranks_T2,on = ["Season","Team2ID"],how = "left")

# test["rank_mean_diff"] = test["T1_rank_mean"] - test["T2_rank_mean"]
# test["rank_max_diff"] = test["T1_rank_max"] - test["T2_rank_max"]
# test["rank_min_diff"] = test["T1_rank_min"] - test["T2_rank_min"]
# test["rank_median_diff"] = test["T1_rank_median"] - test["T2_rank_median"]

# test = test.drop(columns = ['ID','Pred'])

#T1の試合の勝率
train_T1winrate = train.groupby(["Team1ID"])["target"].mean()#new
train_T1winrate_index = train_T1winrate.index
#T2の試合の勝率
train_T2winrate = train.groupby(["Team2ID"])["target"].mean()#new
train_T2winrate_index = train_T2winrate.index
#T1 vs T2 の試合の勝率
train_T1winrate_vsT2 = train.groupby(["Team1ID","Team2ID"])["target"].mean()#new
train_T1winrate_vsT2_index = train_T1winrate_vsT2.index

# #☆調整☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆
train = train.drop(columns = ["T1Score","T2Score"])


#trainのT1の試合の勝率データの作成
for i in range(len(train_T1winrate)):
    train.loc[train['Team1ID'] == train_T1winrate_index[i], 'T1winrate'] = train_T1winrate.iloc[i]
#trainのT2の試合の勝率データの作成
for k in range(len(train_T2winrate)):
    train.loc[train['Team2ID'] == train_T2winrate_index[k], 'T2winrate'] = train_T2winrate.iloc[k]
#trainのT1 vs T2 の試合の勝率の勝率データの作成
for j in range(len(train_T1winrate_vsT2)):
    train.loc[(train['Team1ID'] == train_T1winrate_vsT2_index[j][0])&(train['Team2ID'] == train_T1winrate_vsT2_index[j][1]), 'T1winrate_vsT2'] = train_T1winrate_vsT2.iloc[j]
#testのT1の試合の勝率データの作成
for i in range(len(train_T1winrate)):
    test.loc[test['Team1ID'] == train_T1winrate_index[i], 'T1winrate'] = train_T1winrate.iloc[i]
#testのT2の試合の勝率データの作成
for k in range(len(train_T2winrate)):
    test.loc[test['Team2ID'] == train_T2winrate_index[k], 'T2winrate'] = train_T2winrate.iloc[k]
#testのT1 vs T2 の試合の勝率の勝率データの作成
for j in range(len(train_T1winrate_vsT2)):
    test.loc[(test['Team1ID'] == train_T1winrate_vsT2_index[j][0])&(test['Team2ID'] == train_T1winrate_vsT2_index[j][1]), 'T1winrate_vsT2'] = train_T1winrate_vsT2.iloc[j]
#勝率の差を算出
test["winratediff"] = test["T1winrate"] - test["T2winrate"]
train["winratediff"] = train["T1winrate"] - train["T2winrate"]

#T1の各種データの平均値
train_cols = train.columns
train_col = train_cols[4:]

df_T1_mean = pd.DataFrame()

for param in train_col:
    #T1の試合のparam平均値
    current_param = train.groupby(["Team1ID"])[param].mean()#new
    current_param_index = current_param.index
    #paramの名前の設定
    current_param_T1name = param + "_ALL"
    #[train]にT1の試合のparam平均値データ追加
    for i in range(len(current_param)):
        train.loc[train['Team1ID'] == current_param_index[i], current_param_T1name] = current_param.iloc[i]
    #[test]にT1の試合のparam平均値データ追加
    for i in range(len(current_param)):
        test.loc[test['Team1ID'] == current_param_index[i], current_param_T1name] = current_param.iloc[i]
    #T1 vs T2の試合のparam平均値
    current_param_vsT2 = train.groupby(["Team1ID","Team2ID"])[param].mean()#new
    current_param_vsT2_index = current_param_vsT2.index
    #paramの名前の設定
    current_param_T1vsT2name = param + "_vsT2_ALL"
    #[train]にT1 vs T2 の試合のparam平均値データの追加
    for j in range(len(current_param_vsT2)):
        train.loc[(train['Team1ID'] == current_param_vsT2_index[j][0])&(train['Team2ID'] == current_param_vsT2_index[j][1]), current_param_T1vsT2name] = current_param_vsT2.iloc[j]
    #[test]にT1 vs T2 の試合のparam平均値データの追加
    for j in range(len(current_param_vsT2)):
        test.loc[(test['Team1ID'] == current_param_vsT2_index[j][0])&(test['Team2ID'] == current_param_vsT2_index[j][1]), current_param_T1vsT2name] = current_param_vsT2.iloc[j]        

#まとめて削除
train2 = train.drop(train.columns[4:172],axis=1)
test2 = test.drop(test.columns[5:173],axis=1)

#学習・テストデータの出力
train2.to_csv("train.csv",index = None)
test2.to_csv("test.csv",index = None)
