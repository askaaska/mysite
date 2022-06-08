import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
list_dataframes = []
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        list_dataframes.append(os.path.join(dirname, filename))
        
        
dict_dataframes = {}
for df in list_dataframes:
    if "WDataFiles_Stage2" in df:
        dict_dataframes[os.path.split(df)[-1].split(".")[0]] = pd.read_csv(df, encoding = "cp1252")
    elif "538" in df:
        dict_dataframes[os.path.split(df)[-1].split(".")[0]] = pd.read_csv(df, encoding = "cp1252")
        
for df in dict_dataframes.keys():
    print(df)
    print(dict_dataframes[df].head())
    
list_seasons = dict_dataframes["WRegularSeasonDetailedResults"].Season.unique()

dict_dataframes["WRegularSeasonDetailedResults"].columns

list_stats = ['Score', 'FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR',
       'Ast', 'TO', 'Stl', 'Blk', 'PF']

def other_team(type_team):
    if type_team=="W":
        return "L"
    else:
        return "W"

def add_dict_team(dict_teams, row, type_team):
    id_team = row[type_team+"TeamID"]
    if id_team not in dict_teams.keys():
        dict_teams[id_team] = {}
        for stat in list_stats:
            dict_teams[id_team][stat]=0
        dict_teams[id_team]["Game played"]=0
        dict_teams[id_team]["Game Home"]=0
        dict_teams[id_team]["Game Away"]=0
        dict_teams[id_team]["Game OT"]=0
        dict_teams[id_team]["Wins"]=0
        dict_teams[id_team]["WinsHome"]=0
        dict_teams[id_team]["WinsAway"]=0
        dict_teams[id_team]["WinsOT"]=0
        dict_teams[id_team]["Point diff"] = 0
        dict_teams[id_team]["Point diff win"] = 0
        dict_teams[id_team]["Point diff loss"] = 0
        dict_teams[id_team]["Losses"] = 0
        dict_teams[id_team]["LossesHome"] = 0
        dict_teams[id_team]["LossesAway"] = 0
        dict_teams[id_team]["LossesOT"] = 0
    dict_teams[id_team]["Game played"]+=1
    point_diff = row[type_team+"Score"] - row[other_team(type_team)+"Score"]
    dict_teams[id_team]["Point diff"]+=point_diff
    if type_team=="W":
        dict_teams[id_team]["Wins"]+=1
        dict_teams[id_team]["Point diff win"]+=point_diff
        if row['WLoc']=="H":
            dict_teams[id_team]["Game Home"]+=1
            dict_teams[id_team]["WinsHome"]+=1
        else:
            dict_teams[id_team]["Game Away"]+=1
            dict_teams[id_team]["WinsAway"]+=1
        if row['NumOT']>0:
            dict_teams[id_team]["WinsOT"]+=1
            dict_teams[id_team]["Game OT"]+=1
    else:
        dict_teams[id_team]["Losses"]+=1
        dict_teams[id_team]["Point diff loss"]+=point_diff
        if row['WLoc']=="H":
            dict_teams[id_team]["Game Away"]+=1
            dict_teams[id_team]["LossesAway"]+=1
        else:
            dict_teams[id_team]["Game Home"]+=1
            dict_teams[id_team]["LossesHome"]+=1
        if row['NumOT']>0:
            dict_teams[id_team]["LossesOT"]+=1
            dict_teams[id_team]["Game OT"]+=1
    for stat in list_stats:
        dict_teams[id_team][stat]+=row[type_team+stat]
    point_diff = row[type_team+"Score"] - row[other_team(type_team)+"Score"]
    dict_teams[id_team]["Point diff"]+=point_diff
    
    return dict_teams

def compute_averages(dict_teams):
    dict_averages = {}
    for team in dict_teams.keys():
        dict_averages[team] = {}
        games_played = dict_teams[team]["Game played"]
        games_played_home = dict_teams[team]["Game Home"]
        if games_played_home==0:
            games_played_home=1
        games_played_away = dict_teams[team]["Game Away"]
        if games_played_away==0:
            games_played_away=1
        games_played_ot = dict_teams[team]["Game OT"]
        if games_played_ot==0:
            games_played_ot=1
        for stat in dict_teams[team].keys():
            if stat not in ["Game played", "Game Home", "Game Away", "Game OT", "WinsHome", "WinsAway", "LossesHome", "LossesAway", "WinsOT", "LossesOT"]:
                dict_averages[team][stat] = dict_teams[team][stat] / games_played
            elif stat in ["WinsHome", "LossesHome"]:
                dict_averages[team][stat] = dict_teams[team][stat] / games_played_home
            elif stat in ["WinsAway", "LossesAway"]:
                dict_averages[team][stat] = dict_teams[team][stat] / games_played_away
            elif stat in ["WinsOT", "LossesOT"]:
                dict_averages[team][stat] = dict_teams[team][stat] / games_played_ot

    return dict_averages

from tqdm import tqdm
dict_teams_seasons = {}
dict_teams_averages_seasons={}
for season in tqdm(list_seasons):
    df_season = dict_dataframes["WRegularSeasonDetailedResults"][dict_dataframes["WRegularSeasonDetailedResults"]["Season"]==season]
    dict_teams = {}
    for index, row in df_season.iterrows():
        dict_teams = add_dict_team(dict_teams, row, "W")
        dict_teams = add_dict_team(dict_teams, row, "L")
    
    dict_teams_averages = compute_averages(dict_teams)
    dict_teams_seasons[season] = dict_teams
    dict_teams_averages_seasons[season] = dict_teams_averages
    
list_total_metrics = list(dict_teams_seasons[2016][3104].keys())
list_total_metrics = ["Total_" + x for x in list_total_metrics]
df_totals = pd.DataFrame(columns=["Season", "Team ID"] + list_total_metrics)
for season in dict_teams_seasons.keys():
    for team in dict_teams_seasons[season].keys():
        dict_row = {}
        dict_row["Season"] = int(season)
        dict_row["Team ID"] = int(team)
        for key in dict_teams_seasons[season][team].keys():
            dict_row["Total_"+key] = dict_teams_seasons[season][team][key]
        df_totals = df_totals.append(dict_row, ignore_index=True)

list_averages_metrics = list(dict_teams_averages_seasons[2016][3104].keys())
list_averages_metrics = ["Average_" + x for x in list_averages_metrics]
df_averages = pd.DataFrame(columns=["Season", "Team ID"] + list_averages_metrics)
for season in dict_teams_averages_seasons.keys():
    for team in dict_teams_averages_seasons[season].keys():
        dict_row = {}
        dict_row["Season"] = int(season)
        dict_row["Team ID"] = int(team)
        for key in dict_teams_averages_seasons[season][team].keys():
            dict_row["Average_"+key] = dict_teams_averages_seasons[season][team][key]
        df_averages = df_averages.append(dict_row, ignore_index=True)

def get_team_seed(team_id, season):
    df = dict_dataframes["WNCAATourneySeeds"]
    df_filtered = df[(df["Season"]==season) & (df["TeamID"]==team_id)]
    
    if len(df_filtered)>0:
        return(int(''.join(i for i in df_filtered["Seed"].iloc[0] if i.isdigit())))
    else:
        return 17
    
dict_seed = {}
for season in dict_teams_seasons.keys():
    dict_seed[season]={}
    for team_id in dict_teams_seasons[season].keys():
        dict_seed[season][team_id]=get_team_seed(team_id, season)
        
df_seed = pd.DataFrame(columns=["Season", "Team ID", "Seed"])
for season in dict_seed.keys():
    for team in dict_seed[season].keys():
        dict_row = {"Season":season, "Team ID": team, "Seed": dict_seed[season][team]}
        df_seed = df_seed.append(dict_row, ignore_index=True)
        
df_538 = dict_dataframes["538ratingsWomen"].copy()
df_538 = df_538.rename(columns={"TeamID": "Team ID"})

target_df = pd.DataFrame(columns=["Season", "Team A", "Team B", "WinTeamA", "ScoreDiff"])
for index, row in dict_dataframes["WNCAATourneyCompactResults"].iterrows():
    team_w_id = row["WTeamID"]
    team_l_id = row["LTeamID"]
    team_A = min(team_w_id, team_l_id)
    team_B = max(team_w_id, team_l_id)
    win_team_A = 1 if row["WTeamID"]==team_A else 0
    scoreDiff = row["WScore"] - row["LScore"]
    scoreDiff_team_A = scoreDiff if row["WTeamID"]==team_A else -scoreDiff
    dict_row = {"Season":row["Season"], "Team A": team_A, "Team B": team_B, "WinTeamA": win_team_A, "ScoreDiff": scoreDiff_team_A}
    target_df = target_df.append(dict_row, ignore_index=True)
    
df_target = target_df.copy()
df_target = df_target[df_target["Season"]>=2010]
df_target = pd.merge(df_target, df_seed, how="left", left_on=["Season", "Team A"], right_on=["Season", "Team ID"]).drop(["Team ID"], axis=1)
df_target = pd.merge(df_target, df_seed, how="left", left_on=["Season", "Team B"], right_on=["Season", "Team ID"], suffixes = ("_A", "_B")).drop(["Team ID"], axis=1)
df_target = pd.merge(df_target, df_averages, how="left", left_on=["Season", "Team A"], right_on=["Season", "Team ID"]).drop(["Team ID"], axis=1)
df_target = pd.merge(df_target, df_averages, how="left", left_on=["Season", "Team B"], right_on=["Season", "Team ID"], suffixes = ("_A", "_B")).drop(["Team ID"], axis=1)
df_target = pd.merge(df_target, df_538, how="left", left_on=["Season", "Team A"], right_on=["Season", "Team ID"]).drop(["Team ID", "TeamName"], axis=1)
df_target = pd.merge(df_target, df_538, how="left", left_on=["Season", "Team B"], right_on=["Season", "Team ID"], suffixes = ("_A", "_B")).drop(["Team ID", "TeamName"], axis=1)
df_target = pd.merge(df_target, df_totals, how="left", left_on=["Season", "Team A"], right_on=["Season", "Team ID"]).drop(["Team ID"], axis=1)
df_target = pd.merge(df_target, df_totals, how="left", left_on=["Season", "Team B"], right_on=["Season", "Team ID"], suffixes = ("_A", "_B")).drop(["Team ID"], axis=1)

df_target[df_target["Season"]==2022]

list_feature_columns = list(set(["_".join(x.split("_")[:-1]) for x in list(df_target.columns)]))
list_feature_columns.remove("")

def compute_A_minus_B(df_target, column):
    df_target[column+"_AmenoB"] = df_target[column+"_A"] - df_target[column+"_B"]
    return df_target

for col in list_feature_columns:
    df_target = compute_A_minus_B(df_target, col)
    
df_target_train =  df_target[df_target["Season"]<2021]
df_target_test =  df_target[df_target["Season"]==2021]

for col in df_target_train.columns:
    print(col)
    print(pd.isna(df_target_train[col]).sum())
    
df_target_train["ID"] = df_target_train["Season"].astype(str) + "_" + df_target_train["Team A"].astype(str) + "_" + df_target_train["Team B"].astype(str)

df_full_train = df_target_train.copy()
df_target_train = df_target_train.drop(["Season", "Team A", "Team B", "ScoreDiff"], axis=1)
df_target_train = df_target_train.rename(columns = {"WinTeamA": "Pred"})


df_target_train.to_csv("train_dataset.csv", index=False)