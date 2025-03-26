import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, GRU
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import time

# load data
data = pd.read_csv('/Users/wangzhengzhuo/Desktop/london_weather.csv')

# extract time features
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

filtered_data = data[data['date'] >= '1979-01-01'][[
    'mean_temp', 'max_temp', 'min_temp',
    'precipitation', 'cloud_cover', 'snow_depth', 'pressure',
    'sunshine', 'global_radiation', 'year', 'month', 'day'
]]

# missing value
filtered_data = filtered_data.interpolate(method='linear')

# Winsorization
Q1 = filtered_data.quantile(0.25)
Q3 = filtered_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_data = filtered_data.clip(lower=lower_bound, upper=upper_bound, axis=1)

# extract target and input features
target_df = filtered_data[['mean_temp']]
feature_df = filtered_data.drop('mean_temp', axis=1)

# split the dataset
train_size = int(len(filtered_data) * 0.7)
val_size = int(len(filtered_data) * 0.2)
test_size = len(filtered_data) - train_size - val_size

train_features = feature_df.iloc[:train_size]
val_features = feature_df.iloc[train_size:train_size + val_size]
test_features = feature_df.iloc[train_size + val_size:]

train_targets = target_df.iloc[:train_size]
val_targets = target_df.iloc[train_size:train_size + val_size]
test_targets = target_df.iloc[train_size + val_size:]

# Min-Max Normalization
feature_scaler = MinMaxScaler(feature_range=(0, 1))
train_features_scaled = feature_scaler.fit_transform(train_features)

val_features_scaled = feature_scaler.transform(val_features)
test_features_scaled = feature_scaler.transform(test_features)

target_scaler = MinMaxScaler(feature_range=(0, 1))
train_targets_scaled = target_scaler.fit_transform(train_targets)
val_targets_scaled = target_scaler.transform(val_targets)
test_targets_scaled = target_scaler.transform(test_targets)

#sliding window
def create_dataset(features, targets, look_back, predict_steps):
    x, y = [], []
    for i in range(len(features) - look_back - predict_steps + 1):
        # 只使用特征数据，不包括mean_temp
        x.append(features[i:(i + look_back), :])
        # 预测未来predict_steps天的mean_temp
        y.append(targets[i + look_back: i + look_back + predict_steps, 0])
    return np.array(x), np.array(y)

lookback = 30
future_steps = 5

X_train, Y_train = create_dataset(train_features_scaled, train_targets_scaled, lookback, future_steps)
X_val, Y_val = create_dataset(val_features_scaled, val_targets_scaled, lookback, future_steps)
X_test, Y_test = create_dataset(test_features_scaled, test_targets_scaled, lookback, future_steps)

# adjust learning rate
batch_size = 32
num_epochs = 50
num_train_steps = (len(X_train) // batch_size) * num_epochs

lr_scheduler = PolynomialDecay(
    initial_learning_rate=0.0005,
    end_learning_rate=0.0001,
    decay_steps=num_train_steps
)
# from tensorflow.keras.layers import Layer
# import tensorflow.keras.backend as K
#
# class Attention(Layer):
#     def __init__(self, **kwargs):
#         super(Attention, self).__init__(**kwargs)
#
#     def call(self, inputs):
#         # inputs.shape = (batch_size, time_steps, features)
#         score = K.softmax(K.sum(inputs, axis=-1, keepdims=True), axis=1)
#         context = inputs * score
#         return K.sum(context, axis=1)
# from tensorflow.keras.models import Sequential, Model
# from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
#
# inputs = Input(shape=(lookback, X_train.shape[2]))
# x = LSTM(128, return_sequences=True)(inputs)
# x = Dropout(0.2)(x)
# x = LSTM(64, return_sequences=True)(x)
# x = Dropout(0.2)(x)
# x = LSTM(32, return_sequences=True)(x)
# x = Dropout(0.2)(x)
# x = Attention()(x)
# outputs = Dense(future_steps)(x)
#
# model = Model(inputs, outputs)
# LSTM model
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(lookback, X_train.shape[2])),
    Dropout(0.2),
    LSTM(64, return_sequences=True,activation='tanh'),
    Dropout(0.2),
    LSTM(32, return_sequences=False,activation='tanh'),
    Dropout(0.2),
    Dense(future_steps)
])
# GRU model
# model = Sequential([
#     GRU(128, return_sequences=True, input_shape=(lookback, X_train.shape[2]),activation='tanh'),
#     Dropout(0.2),
#     GRU(64, return_sequences=True,activation='tanh'),
#     Dropout(0.2),
#     GRU(32, return_sequences=False,activation='tanh'),
#     Dropout(0.2),
#     Dense(future_steps)
# ])
# CNN-LSTM model
# model = Sequential([
#     Conv1D(filters=64, kernel_size=3, input_shape=(lookback, X_train.shape[2])),
#     BatchNormalization(),
#     tf.keras.layers.Activation('relu'),
#     MaxPooling1D(pool_size=2),
#     Dropout(0.2),
#
#     # Conv1D(filters=128, kernel_size=3, padding='same'),
#     # BatchNormalization(),
#     # tf.keras.layers.Activation('relu'),
#     # MaxPooling1D(pool_size=2),
#     # Dropout(0.2),
#     LSTM(128, return_sequences=True),
#     Dropout(0.2),
#     LSTM(64, return_sequences=True),
#     Dropout(0.2),
#     LSTM(32, return_sequences=False),
#     Dropout(0.2),
#     Dense(future_steps) ])


model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr_scheduler))
model.summary()

# EarlyStopping set-up
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
#calculate the training time
start_time = time.time()
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=2)
end_time = time.time()
training_time = end_time - start_time
print(f"Total Training Time: {training_time:.2f} seconds")

# prediction
train_predict = model.predict(X_train)
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)

#inverse normalization
def inverse_transform(predictions, scaler):
    reshaped_preds = np.array([predictions[:, i].reshape(-1, 1) for i in range(predictions.shape[1])])
    inv_preds = np.array([scaler.inverse_transform(reshaped_preds[i]) for i in range(len(reshaped_preds))])
    return np.transpose(inv_preds.reshape(future_steps, -1))

train_predict_inverse = inverse_transform(train_predict, target_scaler)
val_predict_inverse = inverse_transform(val_predict, target_scaler)
test_predict_inverse = inverse_transform(test_predict, target_scaler)

Y_train_inverse = inverse_transform(Y_train, target_scaler)
Y_val_inverse = inverse_transform(Y_val, target_scaler)
Y_test_inverse = inverse_transform(Y_test, target_scaler)

# overall RMSE
train_rmse = np.sqrt(mean_squared_error(Y_train_inverse.flatten(), train_predict_inverse.flatten()))
val_rmse = np.sqrt(mean_squared_error(Y_val_inverse.flatten(), val_predict_inverse.flatten()))
test_rmse = np.sqrt(mean_squared_error(Y_test_inverse.flatten(), test_predict_inverse.flatten()))

# RMSE by step
step_rmse = []
for i in range(future_steps):
    rmse = np.sqrt(mean_squared_error(Y_test_inverse[:, i], test_predict_inverse[:, i]))
    step_rmse.append(rmse)
    print(f'Step {i+1} Test RMSE: {rmse:.2f}')

# overall R2
train_r2 = r2_score(Y_train_inverse.flatten(), train_predict_inverse.flatten())
val_r2 = r2_score(Y_val_inverse.flatten(), val_predict_inverse.flatten())
test_r2 = r2_score(Y_test_inverse.flatten(), test_predict_inverse.flatten())

# R2 by step
step_R2 = []
for i in range(future_steps):
    R2 =  r2_score(Y_test_inverse[:, i], test_predict_inverse[:, i])
    step_R2.append(R2)
    print(f'Step {i+1} Test R2: {R2:.2f}')

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Validation RMSE: {val_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f'Train R² Score: {train_r2:.2f}')
print(f'Validation R² Score: {val_r2:.2f}')
print(f'Test R² Score: {test_r2:.2f}')

# save model and scaler
joblib.dump(feature_scaler, "feature_scaler_multi-step.pkl")
joblib.dump(target_scaler, "target_scaler_multi-step.pkl")
model.save("multi-step_model.h5")

# model structure
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_structure_multi-step.png', show_shapes=True, show_layer_names=True)

# train vs validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# residual
plt.figure(figsize=(8, 5))
residuals = Y_test_inverse.flatten() - test_predict_inverse.flatten()
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Prediction Error (°C)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# Actual vs Predicted Scatter Plot
plt.figure(figsize=(8, 8))
sns.scatterplot(x=Y_test_inverse.flatten(), y=test_predict_inverse.flatten(), alpha=0.5)
# y = x perfect prediction
min_val = min(Y_test_inverse.min(), test_predict_inverse.min())
max_val = max(Y_test_inverse.max(), test_predict_inverse.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="dashed", color="black", label="y = x (Perfect Prediction)")

plt.xlabel("Actual Mean Temp (°C)")
plt.ylabel("Predicted Mean Temp (°C)")
plt.title("Scatter Plot: Actual vs Predicted Temperature")
plt.legend()
plt.show()

#Actucal vs Predicted by step
fig, axes = plt.subplots(5, 1, figsize=(12, 15), sharex=True)
fig.suptitle('Actual vs Predicted(5 days ahead)', fontsize=16)
for i in range(future_steps):
    ax = axes[i]
    r2 = r2_score(Y_test_inverse[:, i], test_predict_inverse[:, i])
    ax.plot(Y_test_inverse[:, i], label=f'Actual Mean Temp (t+{i + 1})', color='blue', linewidth=1.5)
    ax.plot(test_predict_inverse[:, i], label=f'Predicted Mean Temp (t+{i + 1})', color='red', linestyle='--', linewidth=1.5)
    ax.set_title(f'Day{i + 1} prediction (R² = {r2:.4f})')
    ax.set_ylabel('Mean Temp (°C)')
    ax.grid(True)
    ax.legend()

axes[-1].set_xlabel('Time Step')
plt.tight_layout()
plt.show()
