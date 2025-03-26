import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# 1️⃣ 读取数据
test_df_raw = pd.read_csv('2021-2023.csv')

# 2️⃣ 日期处理
test_df_raw['date'] = pd.to_datetime(test_df_raw['DATE'], format='%Y%m%d')
test_df_raw['year'] = test_df_raw['date'].dt.year
test_df_raw['month'] = test_df_raw['date'].dt.month
test_df_raw['day'] = test_df_raw['date'].dt.day

# 3️⃣ 重命名列
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

# 4️⃣ 单位转换
test_df['max_temp'] /= 10
test_df['min_temp'] /= 10
test_df['mean_temp'] /= 10
test_df['sunshine'] /= 10
test_df['precipitation'] /= 10
test_df['pressure'] *= 10

# 5️⃣ 删除无用列
if 'HU' in test_df.columns:
    test_df = test_df.drop(columns=['HU'])

# 6️⃣ 缺失值处理
test_df = test_df.interpolate(method='linear')

# 7️⃣ 异常值处理
Q1 = test_df.quantile(0.25)
Q3 = test_df.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
test_df = test_df.clip(lower=lower_bound, upper=upper_bound, axis=1)

# 8️⃣ 加载 scaler 和模型
feature_scaler = joblib.load('feature_scaler_multi-step.pkl')
target_scaler = joblib.load('target_scaler_multi-step.pkl')
model = load_model('multi-step_model.h5')

# 9️⃣ 滑窗构造输入
feature_columns = ['max_temp', 'min_temp', 'precipitation', 'cloud_cover',
                   'snow_depth', 'pressure', 'sunshine', 'global_radiation',
                   'year', 'month', 'day']
look_back = 30  # 与训练时相同的回看天数
future_steps = 5  # 预测未来5天

# 归一化
X_all = feature_scaler.transform(test_df[feature_columns])
y_all = target_scaler.transform(test_df[['mean_temp']])

# ✅ 滑动窗口构造函数 - 适用于多步预测
def create_dataset(features, targets, look_back, predict_steps):
    x, y = [], []
    for i in range(len(features) - look_back - predict_steps + 1):
        x.append(features[i:(i + look_back), :])
        y.append(targets[i + look_back: i + look_back + predict_steps, 0])
    return np.array(x), np.array(y)

# 构建序列
X_seq, y_seq = create_dataset(X_all, y_all, look_back, future_steps)

# 反归一化函数
def inverse_transform(predictions, scaler):
    # 转换多步预测结果的形状，以便反归一化
    reshaped_preds = np.array([predictions[:, i].reshape(-1, 1) for i in range(predictions.shape[1])])
    # 应用反归一化
    inv_preds = np.array([scaler.inverse_transform(reshaped_preds[i]) for i in range(len(reshaped_preds))])
    # 重新整理形状
    return np.transpose(inv_preds.reshape(future_steps, -1))

# 反归一化真实值
y_true = inverse_transform(y_seq, target_scaler)

# 日期对齐（第 look_back+1 天起，到 look_back+future_steps 天）
dates = []
for i in range(len(y_true)):
    dates.append(test_df['date'].iloc[look_back+i:look_back+i+future_steps].reset_index(drop=True))

# 🔢 模型预测
y_pred_scaled = model.predict(X_seq)
y_pred = inverse_transform(y_pred_scaled, target_scaler)

# ✅ 性能评估 - 整体
rmse = np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))
r2 = r2_score(y_true.flatten(), y_pred.flatten())

print(f"整体测试集 RMSE: {rmse:.4f}")
print(f"整体测试集 R²  : {r2:.4f}")

# 每个预测步骤的RMSE
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
    ax.set_xlabel('Time Step')  # ✅ 每个子图都加
    ax.set_title(f'Day {i + 1} (RMSE: {step_rmse[i]:.2f}, R²: {step_r2[i]:.2f})')
    ax.legend()
    ax.grid(True)

# 删除第6个多余子图
if future_steps < 6:
    fig.delaxes(axes_flat[5])

plt.tight_layout()
plt.subplots_adjust(top=0.93)
plt.show()

import matplotlib.pyplot as plt

rows = 2
cols = 3  # 最多放得下3个图
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

# 移除多余的子图（如果 future_steps < rows * cols）
for j in range(future_steps, rows * cols):
    fig.delaxes(axes[j // cols, j % cols])

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.show()

#
# # ✅ 3. 残差随时间图 (每个步骤一个图)
# fig, axes = plt.subplots(future_steps, 1, figsize=(12, 3*future_steps), sharex=True)
# fig.suptitle('Residuals Over Time for Each Prediction Step', fontsize=16)
#
# for i in range(future_steps):
#     ax = axes[i] if future_steps > 1 else axes
#     residuals = y_true[:, i] - y_pred[:, i]
#     ax.plot(sample_indices, residuals[sample_indices], alpha=0.7)
#     ax.axhline(0, color='red', linestyle='--')
#     ax.set_ylabel(f'Residual Day {i+1}')
#     ax.grid(True)
#
# axes[-1].set_xlabel('Sample Index') if future_steps > 1 else axes.set_xlabel('Sample Index')
# plt.tight_layout()
# plt.subplots_adjust(top=0.95)
# plt.show()
#
# # ✅ 4. 残差直方图 (所有步骤在一个图)
# plt.figure(figsize=(12, 6))
# for i in range(future_steps):
#     residuals = y_true[:, i] - y_pred[:, i]
#     plt.hist(residuals, bins=20, alpha=0.5, label=f'Day {i+1}')
#
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
# plt.title('Distribution of Prediction Errors for All Steps')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

