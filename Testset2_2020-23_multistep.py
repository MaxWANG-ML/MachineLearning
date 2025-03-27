import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

test_df_raw = pd.read_csv('2021-2023.csv')

# date format and time features extraction
test_df_raw['date'] = pd.to_datetime(test_df_raw['DATE'], format='%Y%m%d')
test_df_raw['year'] = test_df_raw['date'].dt.year
test_df_raw['month'] = test_df_raw['date'].dt.month
test_df_raw['day'] = test_df_raw['date'].dt.day

# rename the features
column_mapping = {
    'TX': 'max_temp',
    'TN': 'min_temp',
    'TG': 'mean_temp',
    'SS': 'sunshine',
    'SD': 'snow_depth',
    'RR': 'precipitation',
    'QQ': 'global_radiation',
    'PP': 'pressure',
    'CC': 'cloud_cover'
}
test_df = test_df_raw.rename(columns=column_mapping)

# unit transform
test_df['max_temp'] /= 10
test_df['min_temp'] /= 10
test_df['mean_temp'] /= 10
test_df['sunshine'] /= 10
test_df['precipitation'] /= 10
test_df['pressure'] *= 10

# delete humidity
if 'HU' in test_df.columns:
    test_df = test_df.drop(columns=['HU'])

# missing values
test_df = test_df.interpolate(method='linear')

# outliers
Q1 = test_df.quantile(0.25)
Q3 = test_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
test_df = test_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# load scalers and model
feature_scaler = joblib.load('feature_scaler_multi-step.pkl')
target_scaler = joblib.load('target_scaler_multi-step.pkl')
model = load_model('multi-step_model.h5')

feature_columns = ['max_temp', 'min_temp', 'precipitation', 'cloud_cover',
                   'snow_depth', 'pressure', 'sunshine', 'global_radiation',
                   'year', 'month', 'day']
look_back = 30  
future_steps = 5  

# 归一化
X_all = feature_scaler.transform(test_df[feature_columns])
y_all = target_scaler.transform(test_df[['mean_temp']])

# sliding windows
def create_dataset(features, targets, look_back, predict_steps):
    x, y = [], []
    for i in range(len(features) - look_back - predict_steps + 1):
        x.append(features[i:(i + look_back), :])
        y.append(targets[i + look_back: i + look_back + predict_steps, 0])
    return np.array(x), np.array(y)

X_seq, y_seq = create_dataset(X_all, y_all, look_back, future_steps)

# inverse transform
def inverse_transform(predictions, scaler):
    reshaped_preds = np.array([predictions[:, i].reshape(-1, 1) for i in range(predictions.shape[1])])
    inv_preds = np.array([scaler.inverse_transform(reshaped_preds[i]) for i in range(len(reshaped_preds))])
    return np.transpose(inv_preds.reshape(future_steps, -1))

y_true = inverse_transform(y_seq, target_scaler)

dates = []
for i in range(len(y_true)):
    dates.append(test_df['date'].iloc[look_back+i:look_back+i+future_steps].reset_index(drop=True))

y_pred_scaled = model.predict(X_seq)
y_pred = inverse_transform(y_pred_scaled, target_scaler)

rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
r2 = r2_score(y_true.flatten(), y_pred.flatten())

print(f"Overall RMSE: {rmse:.4f}")
print(f"Overall R²  : {r2:.4f}")

# Evaluation for each step
step_rmse = []
step_r2 = []
for i in range(future_steps):
    rmse_step = np.sqrt(mean_squared_error(y_true[:, i], y_pred[:, i]))
    r2_step = r2_score(y_true[:, i], y_pred[:, i])
    step_rmse.append(rmse_step)
    step_r2.append(r2_step)
    print(f'Step {i+1} Test RMSE: {rmse_step:.4f}, R²: {r2_step:.4f}')

import matplotlib.pyplot as plt
import numpy as np

sample_size = min(100, len(y_true))
sample_indices = np.arange(sample_size)

fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
fig.suptitle('Actual vs Predicted Mean Temperature (5 days)', fontsize=18)

axes_flat = axes.flatten()

for i in range(future_steps):
    ax = axes_flat[i]

    ax.plot(sample_indices, y_true[:sample_size, i], label=f'Actual Mean Temp (t+{i + 1})', alpha=0.8)
    ax.plot(sample_indices, y_pred[:sample_size, i], label=f'Predicted Mean Temp (t+{i + 1})',
            linestyle='--', alpha=0.8)

    ax.set_ylabel('Mean Temp (°C)')
    ax.set_xlabel('Time Step') 
    ax.set_title(f'Day {i + 1} (RMSE: {step_rmse[i]:.2f}, R²: {step_r2[i]:.2f})')
    ax.legend()
    ax.grid(True)

if future_steps < 6:
    fig.delaxes(axes_flat[5])

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

import matplotlib.pyplot as plt

rows = 2
cols = 3  
fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
fig.suptitle('Predicted vs Actual Scatter Plot for Each Step', fontsize=16)

for i in range(future_steps):
    row = i // cols
    col = i % cols
    ax = axes[row, col]
    ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.6)
    ax.plot([min(y_true[:, i]), max(y_true[:, i])],
            [min(y_true[:, i]), max(y_true[:, i])], 'r--')
    ax.set_xlabel('Actual Mean Temp (°C)')
    if col == 0:
        ax.set_ylabel('Predicted Mean Temp (°C)')
    ax.set_title(f'Day {i+1}')
    ax.grid(True)

for j in range(future_steps, rows * cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()
