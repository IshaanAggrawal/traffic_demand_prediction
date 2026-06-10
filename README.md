# 🚦 Traffic Demand Prediction Pipeline

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-F3F3F3.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-F3F3F3.svg)
![CatBoost](https://img.shields.io/badge/CatBoost-F3F3F3.svg)

A robust, production-ready machine learning pipeline for predicting urban traffic demand based on spatial-temporal factors, weather conditions, and road infrastructure. This project demonstrates advanced feature engineering, target encoding, and ensemble modeling techniques to solve complex distributional shifts in time-series forecasting.

## 🎯 Project Objective
The goal of this project is to accurately predict traffic demand at specific geographic locations (geohashes) and 15-minute time slots. The system learns historical patterns to forecast future demand, which is crucial for urban planning, traffic management, and dynamic resource allocation.

## 📊 Dataset Size
* **Traffic Demand Dataset** – ~9.6 MB / ~119K records total
  * `train.csv`: 6.75 MB / 77,299 rows
  * `test.csv`: 2.87 MB / 41,778 rows

## 🧠 Key Data Insights & Challenges
The primary challenge of this dataset is a structural temporal-spatial split between the training and testing sets:
* **Temporal Shift:** The training data spans historical days (e.g., Day 48 and Day 49), while the test set is strictly derived from Day 49.
* **Spatial Disjoint:** There is **zero overlap** in `(geohash, slot)` combinations between the training Day 49 and the testing Day 49 sets. However, ~89% of the test combinations are present in the training Day 48 set.
* **Solution:** The problem translates to learning the generalizable temporal patterns from Day 48 to reliably forecast for unobserved locations on Day 49, entirely mitigating data leakage.

## 🏗️ Architecture & Pipeline

### 1. Robust Validation Strategy
To mirror the real-world test distribution and prevent data leakage:
* **Training Set:** Day 48
* **Validation Set:** Day 49 (used for early stopping and evaluation)
* **Prediction:** Models trained on Day 48 are validated on Day 49. For the final submission, models are retrained on the entire dataset with early-stopping iterations discovered during the validation phase.

### 2. Advanced Feature Engineering
Extensive feature engineering was crucial for capturing underlying traffic dynamics:
* **Hierarchical Target Encoding:** Precise mapping of demand using `(geohash, slot)` means. Hierarchical fallbacks were implemented using broader spatial areas (`geo4`, `geo5`), `geo_mean`, and `slot_mean` for previously unseen combinations.
* **Cyclical Time Encoding:** Time-of-day features (`hour`, `slot`) transformed via Sine and Cosine functions to capture temporal continuity.
* **Derived Ratios & Residuals:** Features like `geo_slot_resid` and `slot_geo_ratio` to capture deviations from average spatial behaviours.
* **Categorical & Interaction Features:** Encoding `Weather`, `RoadType`, and `LargeVehicles` presence. Interaction features such as `NumberofLanes × RoadType` capture infrastructure capacity.

### 3. Ensemble Modeling Approach
The pipeline utilizes a diversified ensemble of gradient-boosted trees to maximize generalizability:
* **LightGBM:** Fast, leaf-wise growth, highly effective with tabular spatial data.
* **XGBoost:** Depth-wise growth, providing robust regularization.
* **CatBoost:** Naturally handles categorical encodings and symmetrical trees to prevent overfitting.
* **Optimal Blending:** The final predictions are a weighted ensemble of the three models. The optimal weights ($W_{LGB}, W_{XGB}, W_{CAT}$) are mathematically derived using the **SciPy `minimize`** function to maximize the $R^2$ score on the validation set.

## 🔍 Root Cause Analysis & v2 Fixes
This `v2` pipeline addresses critical flaws discovered in early iterations (`v1`):
1. **Wrong CV Strategy:** KFold shuffling leaked day 48 & 49 together. *Fixed by implementing strict time-based splits.*
2. **Target Encoding Leakage:** Complex KFold target encoding led to garbage feature generation. *Fixed by adopting a direct, leak-free lookup dictionary approach.*
3. **Misaligned Validation Proxy:** Standard CV did not reflect the test distribution. *Fixed by designing validation around the `Day 48 -> Day 49` paradigm.*

## 🚀 Future Enhancements (Roadmap)
To push the performance even higher, the following enhancements are planned:
* **Spatial Neighbour Features:** Decoding geohashes to lat/lon and aggregating the demand of the 8 surrounding spatial neighbours.
* **Bayesian Smoothing:** Applying Laplace smoothing to target encodings `(count * mean + k * global_mean) / (count + k)` to penalize rare combinations.
* **Deep Learning Integrations:** Utilizing Neural Networks (MLP) specifically trained on gradient-boosted residuals, or integrating **TabNet** for attention-based tabular learning.

## 💻 How to Run

### Prerequisites
Install the required dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn lightgbm xgboost catboost optuna scipy
```

### Execution
1. Place `train.csv` and `test.csv` in the appropriate data directory.
2. Run the `traffic_demand_v2.ipynb` Jupyter Notebook.
3. The notebook will automatically execute the leak-free feature engineering, train the baseline and gradient-boosting models, optimize the ensemble weights, and output the final `submission.csv`.
