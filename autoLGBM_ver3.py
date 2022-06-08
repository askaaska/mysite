# -*- coding: utf-8 -*-
"""
URL:
"""
import pandas as pd
import numpy as np
import random
import time
import gc
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import matplotlib.pylab as plt
import seaborn as sns
from autolgbm import AutoLGBM

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
    
    TEST = False
    
if config.TEST:
    config.EPOCHS = 100

#testのほうはIDとPredを削除しないといけないから再度作成
test = pd.read_csv("test.csv")
test = test.drop(columns = ['ID','Pred'])

test.to_csv("test_del_IDPred.csv",index = None)

"""
学習
"""
train_filename = "train.csv"
output = "output"
test_filename = "test_del_IDPred.csv"
task = None
targets = [config.TARGET]
features = None
categorical_features = None
use_gpu = False
num_folds = 10
seed = 42
num_trials = 10000
time_limit = 7200
fast = False

algbm = AutoLGBM(
    train_filename=train_filename,
    output=output,
    test_filename=test_filename,
    task=task,
    targets=targets,
    features=features,
    categorical_features=categorical_features,
    use_gpu=use_gpu,
    num_folds=num_folds,
    seed=seed,
    num_trials=num_trials,
    time_limit=time_limit,
    fast=fast,
)

algbm.train()

#現在の時刻
import datetime
today = datetime.date.today()
todaydetail = datetime.datetime.today()
today_time = str(today) +'_' +str(todaydetail.hour) + '_' + str(todaydetail.minute) + '_'
#実行ファイルのパス取得
import os
cd = (os.getcwd())

#フォルダー作成
result_dir = cd + "\_" + today_time + "AutoGBMver3"
os.mkdir(result_dir)

"""
予測結果の出力
"""
# #Stage1
# submission = pd.read_csv("MDataFiles_Stage1/MSampleSubmissionStage1.csv")
#Stage2
submission = pd.read_csv("WDataFiles_Stage2/WSampleSubmissionStage2.csv")

autolgb_pred = pd.read_csv("output/test_predictions.csv")

submission['Pred'] = autolgb_pred['1']
# submission.rename(columns = {'1':'Pred'}, inplace = True)
out_filename = result_dir+ "\_" + "submission.csv"
submission.to_csv(out_filename, index=False)

import dill
out_filename = result_dir+ "\_" + "autolgbm.pkl"
dill.dump_session(out_filename)
#読込
# dill.load(open('autolgbm0308.pkl','rb'))