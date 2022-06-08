# -*- coding: utf-8 -*-
"""

"""
import pandas as pd
import numpy as np
import random
import time
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import lightgbm as lgb
import matplotlib.pylab as plt
import seaborn as sns

class config:    
    
    SEED = 42
    FOLD_TO_RUN = 0
    FOLDS = 5
    EARLY_STOPPING_ROUNDS = 200
    VERBOSE = 1000
    
    model_type = 'lgb_regression'  
    EPOCHS = 20000
    LR = 2e-4
    message='baseline'
    TARGET = 'target'
    # TARGET = 'WinA'
    
    TEST = False
    
if config.TEST:
    config.EPOCHS = 100


lgb_params = {
    'objective': 'regression',
    'n_estimators': config.EPOCHS,
    'random_state': config.SEED,
    'learning_rate': config.LR,
    'subsample': 0.44,
    'subsample_freq': 1,
    'colsample_bytree': 0.64,
    'reg_alpha': 0.07,
    'reg_lambda': 0.07,
    'max_depth':100,
    'num_leaves':356,
    'min_child_weight': 256,
    'min_child_samples': 72,
    'device':'gpu'
}
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# train = pd.read_csv("train_goodbaseline.csv")
# test = pd.read_csv("test_goodbaseline.csv")

test = test.drop(columns = ['Pred','ID'])


features = list(train.columns)
features.remove("target")
# features.remove("WinA")

lgb_oof = np.zeros(train.shape[0])
val_pred = np.zeros(train.shape[0])
lgb_pred = np.zeros(test.shape[0])
lgb_importances = pd.DataFrame()
models = []
skf = StratifiedKFold(n_splits=config.FOLDS, shuffle = True , random_state=config.SEED)
RMSE_list = []
y_valid_df = pd.DataFrame()
val_pred_df = pd.DataFrame()

for fold, (trn_idx, val_idx) in enumerate(skf.split(X=train, y=train['Season'])):
    print(f"===== fold {fold} =====")
    X_train,y_train = train[features].iloc[trn_idx],train[config.TARGET].iloc[trn_idx]
    X_valid,y_valid = train[features].iloc[val_idx],train[config.TARGET].iloc[val_idx]
    start = time.time()
    model = LGBMRegressor(**lgb_params)
    model.fit(
        X_train,
        y_train,
        eval_set = (X_valid,y_valid),
        eval_metric = 'rmse',
        early_stopping_rounds=config.EARLY_STOPPING_ROUNDS,
        verbose=config.VERBOSE,
    )
    fi_tmp = pd.DataFrame()
    fi_tmp['feature'] = model.feature_name_
    fi_tmp['importance'] = model.feature_importances_
    fi_tmp['fold'] = fold
    fi_tmp['seed'] = config.SEED
    lgb_importances = lgb_importances.append(fi_tmp)
    val_pred = model.predict(X_valid) 
    lgb_pred += model.predict(test)/config.FOLDS
    elapsed = time.time() - start
    rmse = np.sqrt(mean_squared_error(y_valid, val_pred))
    RMSE_list.append(rmse)
    y_valid_df = pd.concat([y_valid_df,y_valid], ignore_index=True)
    val_pred_df = pd.concat([val_pred_df,pd.DataFrame(val_pred)], ignore_index=True)
    print(f"fold {fold} - lgb rmse: {rmse:.6f}, elapsed time: {elapsed:.2f}sec\n")
    models.append(model)
    
order = list(lgb_importances.groupby('feature').mean().sort_values('importance', ascending=False).index)
y_valid_df.columns = ["anser_pred"]
val_pred_df.columns = ["result_pred"]

#現在の時刻
import datetime
today = datetime.date.today()
todaydetail = datetime.datetime.today()
today_time = str(today) +'_' +str(todaydetail.hour) + '_' + str(todaydetail.minute) + '_'
#実行ファイルのパス取得
import os
cd = (os.getcwd())

#フォルダー作成
result_dir = cd + "\_" + today_time + "lightGBMver2"
os.mkdir(result_dir)

fig = plt.figure(figsize=(16, 32), tight_layout=True)
sns.barplot(x="importance", y="feature", data=lgb_importances.groupby('feature').mean().reset_index(), order=order)
plt.title("LightGBM feature importances")
out_filename = result_dir+ "\_" + "LightGBM feature importances"
plt.savefig(out_filename)
plt.show()

for pred_num in range(len(lgb_pred)):
    if lgb_pred[pred_num]<0:
        lgb_pred[pred_num]=0
    if lgb_pred[pred_num]>1:
        lgb_pred[pred_num]=1
#Stage1        
# df_submission = pd.read_csv("MDataFiles_Stage1/MSampleSubmissionStage1.csv")
#Stage2        
df_submission = pd.read_csv("WDataFiles_Stage2/WSampleSubmissionStage2.csv")

df_submission['Pred'] = lgb_pred
out_filename = result_dir+ "\_" + "submission.csv"
df_submission.to_csv(out_filename,index = None)

df_lgb_importances = lgb_importances.groupby('feature').mean().reset_index()
out_filename = result_dir+ "\_" + "lgb_importances.csv"
df_lgb_importances.to_csv(out_filename,index = None)

df_RMSE = pd.DataFrame(RMSE_list).T
RMSE_mean = df_RMSE.mean(axis='columns') # 行ごとの平均を求める
df_RMSE = pd.concat([df_RMSE,RMSE_mean],axis = 1)
df_RMSE.columns = ["fold1_RMSE","fold2_RMSE","fold3_RMSE","fold4_RMSE","fold5_RMSE","RMSE_mean"]
out_filename = result_dir+ "\_" + "RMSE_result.csv"
df_RMSE.to_csv(out_filename,index = None)

answer_pred = y_valid_df
result_pred = val_pred_df

#ヒストグラム作成
plt.figure()
plt.hist(answer_pred, bins=100, alpha=0.3, label='answer_pred', histtype='stepfilled', color='r')
plt.hist(result_pred, bins=100, alpha=0.3, label='result_pred', histtype='stepfilled', color='b')
plt.legend(loc='upper left')
plt.grid(True)
plt.title('RMSE : '+str(RMSE_mean))
out_filename = result_dir+ "\_" + "pred_hist"
plt.savefig(out_filename)
plt.show()

#散布図作成
plt.figure()
plt.scatter(answer_pred.index,answer_pred,color='r',label="answer_pred", alpha=0.3)
plt.scatter(result_pred.index,result_pred,color='b',label="result_pred", alpha=0.3)
plt.ylim(-0.1,1.1)
plt.legend(loc='upper right',
           bbox_to_anchor=(1.05, 0.5, 0.5, .100), 
           borderaxespad=0.,)
plt.grid(True)
plt.title('RMSE : '+str(RMSE_mean))
out_filename = result_dir+ "\_" +  "pred_scatter"
plt.savefig(out_filename, bbox_inches='tight')
plt.show()

scatter_df = pd.concat([y_valid_df,val_pred_df])
scatter_df.columns = ["answer_pred","result_pred"]
out_filename = result_dir+ "\_" + "scatter_df.csv"
scatter_df.to_csv(out_filename,index = None)

# """
# 様々な可視化
# """
# import lightgbm
# lightgbm.plot_tree(model, tree_index=0)
# result = lightgbm.booster_.trees_to_dataframe()


import dill
#保存
out_filename = result_dir+ "\_" + "lightgbm.pkl"
dill.dump_session(out_filename)
#読込
# dill.load(open('lightlgbm0309.pkl','rb'))