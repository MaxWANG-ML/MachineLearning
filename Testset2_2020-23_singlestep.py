import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import load_model

# load Test set 2
test_df_raw = pd.read_csv('2021-2023.csv')

# extract time features
test_df_raw['date'] = pd.to_datetime(test_df_raw['DATE'], format='%Y%m%d')
test_df_raw['year'] = test_df_raw['date'].dt.year
test_df_raw['month'] = test_df_raw['date'].dt.month
test_df_raw['day'] = test_df_raw['date'].dt.day

# rename the features
column_mapping = {
    'TX': 'max_temp','TN': 'min_temp','TG': 'mean_temp',
    'SS': 'sunshine','SD': 'snow_depth','RR': 'precipitation',
    'QQ': 'global_radiation','PP': 'pressure','CC': 'cloud_cover'
}
test_df = test_df_raw.rename(columns=column_mapping)
# unit transform
test_df['max_temp'] /= 10
test_df['min_temp'] /= 10
test_df['mean_temp'] /= 10
test_df['sunshine'] /= 10
test_df['precipitation'] /= 10
test_df['pressure'] *= 10
# delete the humidity
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
feature_scaler = joblib.load('feature_scaler_single-step.pkl')
target_scaler = joblib.load('target_scaler_single-step.pkl')
model = load_model('single-step_model.h5')

# 9️⃣ 滑窗构造输入
feature_columns = ['max_temp', 'min_temp', 'precipitation', 'cloud_cover',
                   'snow_depth', 'pressure', 'sunshine', 'global_radiation',
                   'year', 'month', 'day']
look_back = 5

# 归一化
X_all = feature_scaler.transform(test_df[feature_columns])
y_all = target_scaler.transform(test_df[['mean_temp']])

# ✅ 滑动窗口构造函数
def create_dataset(features, targets, look_back):
    x, y = [], []
    for i in range(len(features) - look_back):
        x.append(features[i:(i + look_back), :])
        y.append(targets[i + look_back, 0])
    return np.array(x), np.array(y)

# 构建序列
X_seq, y_seq = create_dataset(X_all, y_all, look_back)
y_true = target_scaler.inverse_transform(y_seq.reshape(-1, 1)).flatten()

# 日期对齐（第 look_back+1 天起）
date_seq = test_df['date'].iloc[look_back:].reset_index(drop=True)

# 🔢 模型预测
y_pred_scaled = model.predict(X_seq)
y_pred = target_scaler.inverse_transform(y_pred_scaled).flatten()

# ✅ 性能评估
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
r2 = r2_score(y_true, y_pred)

print(f"测试集 RMSE: {rmse:.4f}")
print(f"测试集 R²  : {r2:.4f}")

# ✅ 1. 时间序列预测对比图
plt.figure(figsize=(12, 6))
plt.plot(date_seq, y_true, label='True Mean Temp', alpha=0.8)
plt.plot(date_seq, y_pred, label='Predicted Mean Temp', linestyle='--', alpha=0.8)
plt.xlabel('Date')
plt.ylabel('Mean Temperature (°C)')
plt.title('True vs Predicted Mean Temperature')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ✅ 2. 预测 vs 真实值散点图
plt.figure(figsize=(6, 6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)],linestyle="dashed", color="black", label="y = x (Perfect Prediction)")
plt.xlabel('Actual Mean Temp (°C)')
plt.ylabel('Predicted Mean Temp (°C)')
plt.title('Predicted vs Actual Scatter Plot')
plt.grid(True)
plt.tight_layout()
plt.show()



# ✅ 4. 残差直方图
# plt.figure(figsize=(8, 5))
# plt.hist(residuals, bins=30, edgecolor='black', alpha=0.7)
# plt.xlabel('Residual')
# plt.ylabel('Frequency')
# plt.title('Distribution of Prediction Errors')
# plt.grid(True)
# plt.tight_layout()
# plt.show()
