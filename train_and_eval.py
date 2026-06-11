import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import pygeohash
import warnings

warnings.filterwarnings('ignore')
SEED = 42
np.random.seed(SEED)
sns.set_theme(style='whitegrid')

# Create metrics directory
os.makedirs('metrics', exist_ok=True)

print("Loading data...")
# Load data
train = pd.read_csv('data/train.csv')
test  = pd.read_csv('data/test.csv')

def parse_slot(ts):
    h, m = ts.split(':')
    return int(h) * 4 + int(m) // 15

for df in [train, test]:
    df['slot'] = df['timestamp'].apply(parse_slot)
    df['hour'] = df['timestamp'].apply(lambda x: int(x.split(':')[0]))

def build_features(target_df, source_df):
    gm  = source_df['demand'].mean()
    gsl  = source_df.groupby(['geohash','slot'])['demand'].mean().to_dict()
    gl   = source_df.groupby('geohash')['demand'].mean().to_dict()
    sl   = source_df.groupby('slot')['demand'].mean().to_dict()
    hl   = source_df.groupby('hour')['demand'].mean().to_dict()
    
    s2 = source_df.copy()
    s2['geo4'] = s2['geohash'].str[:4]
    s2['geo5'] = s2['geohash'].str[:5]
    g4l  = s2.groupby('geo4')['demand'].mean().to_dict()
    g5l  = s2.groupby('geo5')['demand'].mean().to_dict()
    g4sl = s2.groupby(['geo4','slot'])['demand'].mean().to_dict()
    
    d = target_df.copy()
    d['geo4'] = d['geohash'].str[:4]
    d['geo5'] = d['geohash'].str[:5]
    
    d['geo_slot_te']  = [gsl.get((g,s), gl.get(g, sl.get(s, gm)))
                         for g,s in zip(d['geohash'], d['slot'])]
    d['geo_mean']     = d['geohash'].map(gl).fillna(gm)
    d['slot_mean']    = d['slot'].map(sl).fillna(gm)
    d['hour_mean']    = d['hour'].map(hl).fillna(gm)
    d['geo4_mean']    = d['geo4'].map(g4l).fillna(gm)
    d['geo5_mean']    = d['geo5'].map(g5l).fillna(gm)
    d['geo4_slot_te'] = [g4sl.get((g,s), g4l.get(g, sl.get(s, gm)))
                         for g,s in zip(d['geo4'], d['slot'])]
    
    d['geo_slot_resid'] = d['geo_slot_te'] - d['geo_mean']
    d['slot_geo_ratio'] = d['slot_mean'] / (d['geo_mean'] + 1e-6)
    d['geo4_te_diff']   = d['geo_slot_te'] - d['geo4_slot_te']
    
    binary  = {'Yes':1,'No':0,'Allowed':1,'Not Allowed':0}
    road    = {'Highway':3,'Street':2,'Residential':1}
    weather = {'Sunny':0,'Foggy':1,'Rainy':2,'Snowy':3}
    
    d['lv_enc']   = d.get('LargeVehicles', pd.Series([0]*len(d))).map(binary).fillna(0)
    d['lm_enc']   = d.get('Landmarks', pd.Series([0]*len(d))).map(binary).fillna(0)
    d['road_enc'] = d.get('RoadType', pd.Series([0]*len(d))).map(road).fillna(0)
    d['wx_enc']   = d.get('Weather', pd.Series([0]*len(d))).map(weather).fillna(-1)
    
    tm = source_df['Temperature'].median()
    if 'Temperature' in d.columns:
        d['temp']      = d['Temperature'].fillna(tm)
        d['temp_miss'] = d['Temperature'].isna().astype(int)
    else:
        d['temp'] = tm
        d['temp_miss'] = 1
        
    d['hour_sin'] = np.sin(2*np.pi*d['hour']/24)
    d['hour_cos'] = np.cos(2*np.pi*d['hour']/24)
    d['slot_sin'] = np.sin(2*np.pi*d['slot']/96)
    d['slot_cos'] = np.cos(2*np.pi*d['slot']/96)
    
    num_lanes = d.get('NumberofLanes', pd.Series([1]*len(d)))
    d['lane_road'] = num_lanes * d['road_enc']
    d['lane_lv']   = num_lanes * d['lv_enc']
    d['NumberofLanes'] = num_lanes
    
    # NEW EXPERIMENT FEATURES
    # Lat Lon
    lat_lon = [pygeohash.decode(g) for g in d['geohash']]
    d['lat'] = [x[0] for x in lat_lon]
    d['lon'] = [x[1] for x in lat_lon]
    
    # Weather Bins
    d['temp_bin'] = pd.cut(d['temp'], bins=[-np.inf, 15, 25, np.inf], labels=[0, 1, 2]).astype(int)
    d['wx_temp_combo'] = d['wx_enc'].astype(str) + "_" + d['temp_bin'].astype(str)
    
    return d

FEATS = [
    'geo_slot_te', 'geo_mean', 'slot_mean', 'hour_mean',
    'geo4_mean', 'geo5_mean', 'geo4_slot_te',
    'geo_slot_resid', 'slot_geo_ratio', 'geo4_te_diff',
    'hour', 'slot', 'NumberofLanes',
    'lv_enc', 'lm_enc', 'road_enc', 'wx_enc',
    'temp', 'temp_miss', 'lane_road', 'lane_lv',
    'hour_sin', 'hour_cos', 'slot_sin', 'slot_cos',
    'lat', 'lon', 'wx_temp_te'
]

train48 = train[train['day']==48].copy()
train49 = train[train['day']==49].copy()

print("Building features...")
val_f = build_features(train49, train48)
tr48_f = build_features(train48, train48)

# Target encode the weather combo
wt_map = tr48_f.groupby('wx_temp_combo')['demand'].mean().to_dict()
gm = tr48_f['demand'].mean()
tr48_f['wx_temp_te'] = tr48_f['wx_temp_combo'].map(wt_map).fillna(gm)
val_f['wx_temp_te']  = val_f['wx_temp_combo'].map(wt_map).fillna(gm)

y_train_raw = train48['demand'].values
y_val_raw = train49['demand'].values

# Regression Balancing: Log transformation
y_train = np.log1p(y_train_raw * 10)
y_val = np.log1p(y_val_raw * 10)

# Regression Balancing: Sample Weights
w_train = 1.0 + (y_train_raw * 10)

print("Training LightGBM...")
lgb_m = lgb.LGBMRegressor(
    n_estimators=3000, learning_rate=0.02,
    num_leaves=127, min_child_samples=10,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.5,
    random_state=SEED, verbose=-1
)
lgb_m.fit(
    tr48_f[FEATS], y_train,
    sample_weight=w_train,
    eval_set=[(val_f[FEATS], y_val)],
    callbacks=[lgb.early_stopping(100, verbose=False)]
)
lgb_val = np.clip(np.expm1(lgb_m.predict(val_f[FEATS])) / 10, 0, 1)

print("Training XGBoost...")
xgb_m = xgb.XGBRegressor(
    n_estimators=3000, learning_rate=0.02, max_depth=7,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=0.5,
    random_state=SEED, tree_method='hist', verbosity=0,
    early_stopping_rounds=100, eval_metric='rmse'
)
xgb_m.fit(
    tr48_f[FEATS], y_train,
    sample_weight=w_train,
    eval_set=[(val_f[FEATS], y_val)],
    verbose=False
)
xgb_val = np.clip(np.expm1(xgb_m.predict(val_f[FEATS])) / 10, 0, 1)

print("Training CatBoost...")
cat_m = CatBoostRegressor(
    iterations=3000, learning_rate=0.02,
    depth=8, l2_leaf_reg=3,
    random_seed=SEED, verbose=0,
    early_stopping_rounds=100
)
cat_m.fit(
    tr48_f[FEATS], y_train,
    sample_weight=w_train,
    eval_set=(val_f[FEATS], y_val),
    verbose=False
)
cat_val = np.clip(np.expm1(cat_m.predict(val_f[FEATS])) / 10, 0, 1)

print("Calculating metrics...")

def calc_metrics(y_true, y_pred, prefix):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    y_true_cls = (y_true > 0.5).astype(int)
    y_pred_cls = (y_pred > 0.5).astype(int)
    acc = accuracy_score(y_true_cls, y_pred_cls)
    f1 = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    prec = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    
    return {
        f"{prefix}_MSE": mse,
        f"{prefix}_RMSE": rmse,
        f"{prefix}_MAE": mae,
        f"{prefix}_R2": r2,
        f"{prefix}_Accuracy(Threshold 0.5)": acc,
        f"{prefix}_F1(Threshold 0.5)": f1,
        f"{prefix}_Precision(Threshold 0.5)": prec,
        f"{prefix}_Recall(Threshold 0.5)": rec
    }

metrics = {}
metrics.update(calc_metrics(y_val_raw, lgb_val, "LGBM"))
metrics.update(calc_metrics(y_val_raw, xgb_val, "XGB"))
metrics.update(calc_metrics(y_val_raw, cat_val, "CAT"))

# Ensemble
best_score, best_w = 0, (1, 0, 0)
for w1 in np.arange(0, 1.1, 0.1):
    for w2 in np.arange(0, 1.1-w1, 0.1):
        w3 = round(1 - w1 - w2, 2)
        if w3 < 0: continue
        ens = w1*lgb_val + w2*xgb_val + w3*cat_val
        s   = r2_score(y_val_raw, np.clip(ens, 0, 1))
        if s > best_score:
            best_score, best_w = s, (w1, w2, w3)

ens_val = np.clip(best_w[0]*lgb_val + best_w[1]*xgb_val + best_w[2]*cat_val, 0, 1)
metrics.update(calc_metrics(y_val_raw, ens_val, "Ensemble"))
metrics['Ensemble_Weights'] = f"LGB:{best_w[0]:.2f}, XGB:{best_w[1]:.2f}, CAT:{best_w[2]:.2f}"

with open('metrics/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=4)

with open('metrics/metrics.txt', 'w') as f:
    for k, v in metrics.items():
        if isinstance(v, float):
            f.write(f"{k}: {v:.6f}\n")
        else:
            f.write(f"{k}: {v}\n")

print("Generating plots...")
plt.figure(figsize=(10, 8))
fi = pd.DataFrame({'feature': FEATS, 'importance': lgb_m.feature_importances_})
fi = fi.sort_values('importance', ascending=True)
plt.barh(fi['feature'], fi['importance'], color='steelblue')
plt.title('LightGBM Feature Importance (Weighted)')
plt.tight_layout()
plt.savefig('metrics/feature_importance.png')
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(y_val_raw, ens_val, alpha=0.3, s=5)
plt.plot([0,1], [0,1], color='red', linestyle='--')
plt.xlabel('Actual Demand')
plt.ylabel('Predicted Demand')
plt.title('Actual vs Predicted Demand (Weighted Ensemble)')
plt.tight_layout()
plt.savefig('metrics/actual_vs_predicted.png')
plt.close()

plt.figure(figsize=(10, 6))
sns.histplot(y_val_raw, color='blue', alpha=0.5, label='Actual', kde=True, bins=50)
sns.histplot(ens_val, color='orange', alpha=0.5, label='Predicted', kde=True, bins=50)
plt.title('Demand Distribution (Actual vs Predicted)')
plt.legend()
plt.tight_layout()
plt.savefig('metrics/distribution_comparison.png')
plt.close()

plt.figure(figsize=(8, 6))
residuals = y_val_raw - ens_val
plt.scatter(ens_val, residuals, alpha=0.3, s=5)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Predicted Demand')
plt.ylabel('Residuals')
plt.title('Residuals vs Predicted')
plt.tight_layout()
plt.savefig('metrics/residuals_plot.png')
plt.close()

print("Done! All metrics and plots are saved in the 'metrics' folder.")
