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

data = pd.read_csv('london_weather.csv')

# transform date format and extract time features
data['date'] = pd.to_datetime(data['date'], format='%Y%m%d')
data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day

# filter the target features
filtered_data = data[data['date'] >= '1979-01-01'][[
    'mean_temp', 'max_temp', 'min_temp',
    'precipitation', 'cloud_cover', 'snow_depth', 'pressure',
    'sunshine', 'global_radiation', 'year', 'month', 'day'
]]

#plot time distribution of each feature(2010-2020)
import matplotlib.dates as mdates
feature_units = {
    'mean_temp': 'Mean Temperature (°C)',
    'max_temp': 'Maximum Temperature (°C)',
    'min_temp': 'Minimum Temperature (°C)',
    'precipitation': 'Precipitation (mm)',
    'cloud_cover': 'Cloud Cover (Oktas)',
    'snow_depth': 'Snow Depth (cm)',
    'pressure': 'Pressure (Pa)',
    'sunshine': 'Sunshine (Hour)',
    'global_radiation': 'Global Radiation (W/m²)'
}
mask = (data['date'] >= '2010-01-01') & (data['date'] <= '2020-12-31')
data_subset = data.loc[mask]
filtered_subset = filtered_data.loc[mask]

cols_to_plot = list(feature_units.keys())
num_cols = 3
num_rows = int(np.ceil(len(cols_to_plot) / num_cols))

fig, axes = plt.subplots(num_rows, num_cols, figsize=(18, 4 * num_rows), sharex=True)
axes = axes.flatten()

years = mdates.YearLocator()
years_fmt = mdates.DateFormatter('%Y')

for i, col in enumerate(cols_to_plot):
    axes[i].plot(data_subset['date'], filtered_subset[col], label=feature_units[col], linewidth=0.8)
    axes[i].set_title(feature_units[col])
    axes[i].xaxis.set_major_locator(years)
    axes[i].xaxis.set_major_formatter(years_fmt)
    axes[i].set_xlabel('Date')
    axes[i].grid(True)
    axes[i].legend(loc='upper right')

for j in range(len(cols_to_plot), len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Weather Features (2010–2020)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

#correlation analysis
correlation_data = data[data['date'] >= '1979-01-01'][[
    'mean_temp', 'max_temp', 'min_temp',
    'precipitation', 'cloud_cover', 'snow_depth', 'pressure',
    'sunshine', 'global_radiation'
]]
corr_matrix = correlation_data.corr('spearman')

corr_with_temp = corr_matrix['mean_temp'].sort_values(ascending=False)
print(corr_with_temp)

plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".3f", linewidths=0.5)
plt.title("Feature Correlation Heatmap")
plt.show()

# process missing value
filtered_data = filtered_data.interpolate(method='linear')

# Winsorization：process outliers instead of dropping
Q1 = filtered_data.quantile(0.25)
Q3 = filtered_data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
filtered_data = filtered_data.clip(lower=lower_bound, upper=upper_bound, axis=1)

# extract target feature and input features
target_df = filtered_data[['mean_temp']]
feature_df = filtered_data.drop('mean_temp', axis=1)

# use sliding window to transfer time series data for supervised learning
def create_dataset(features, targets, input_steps):
    x, y = [], []
    for i in range(len(features) - input_steps):
        x.append(features[i:(i + input_steps), :])
        y.append(targets[i + input_steps, 0])
    return np.array(x), np.array(y)

# split the data set into training, validation and testing
train_size = int(len(filtered_data) * 0.7)
val_size = int(len(filtered_data) * 0.2)
test_size = len(filtered_data) - train_size - val_size

# get the input features
train_features = feature_df.iloc[:train_size]
val_features = feature_df.iloc[train_size:train_size + val_size]
test_features = feature_df.iloc[train_size + val_size:]
# get the target features
train_targets = target_df.iloc[:train_size]
val_targets = target_df.iloc[train_size:train_size + val_size]
test_targets = target_df.iloc[train_size + val_size:]

# normalization for input features
feature_scaler = MinMaxScaler(feature_range=(0, 1))
train_features_scaled = feature_scaler.fit_transform(train_features)
val_features_scaled = feature_scaler.transform(val_features)
test_features_scaled = feature_scaler.transform(test_features)

# normalization for target feature
target_scaler = MinMaxScaler(feature_range=(0, 1))
train_targets_scaled = target_scaler.fit_transform(train_targets)
val_targets_scaled = target_scaler.transform(val_targets)
test_targets_scaled = target_scaler.transform(test_targets)

# generate supervised data
inputsteps = 5
X_train, Y_train = create_dataset(train_features_scaled, train_targets_scaled, inputsteps)
X_val, Y_val = create_dataset(val_features_scaled, val_targets_scaled, inputsteps)
X_test, Y_test = create_dataset(test_features_scaled, test_targets_scaled, inputsteps)

# adjust the learning rate（polynomial）
batch_size = 32
num_epochs = 50
num_train_steps = (len(X_train) // batch_size) * num_epochs

lr_scheduler = PolynomialDecay(
    initial_learning_rate=0.0005,
    end_learning_rate=0.0001,
    decay_steps=num_train_steps)

# model structure
model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(inputsteps, X_train.shape[2]),activation='tanh'),
    Dropout(0.2),
    LSTM(64, return_sequences=True,activation='tanh'),
    Dropout(0.2),
    LSTM(32, return_sequences=False,activation='tanh'),
    Dropout(0.2),
    Dense(1)
])
# model = Sequential([
#     GRU(128, return_sequences=True, input_shape=(inputsteps, X_train.shape[2])),
#     Dropout(0.2),
#     GRU(64, return_sequences=True),
#     Dropout(0.2),
#     GRU(32, return_sequences=False),
#     Dropout(0.2),
#     Dense(1)
#     ])
# model = Sequential([
#     Conv1D(filters=128, kernel_size=5, padding='same',activation='relu', input_shape=(inputsteps, X_train.shape[2])),
#     BatchNormalization(),
#     MaxPooling1D(pool_size=2),
#     Dropout(0.2),
#     # Conv1D(filters=64, kernel_size=5, padding='same'),
#     # BatchNormalization(),
#     # tf.keras.layers.Activation('relu'),
#     # MaxPooling1D(pool_size=2),
#     # Dropout(0.2),
#     LSTM(100, return_sequences=True),
#     Dropout(0.2),
#     LSTM(50, return_sequences=False),
#     Dropout(0.2),
#     Dense(1) ])

model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=lr_scheduler))

model.summary()

# EarlyStopping set up
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
start_time = time.time()
history = model.fit(X_train, Y_train, epochs=num_epochs, batch_size=batch_size,
                    validation_data=(X_val, Y_val), callbacks=[early_stopping], verbose=2)
end_time = time.time()
training_time = end_time - start_time
print(f"Total Training Time: {training_time: .2f} seconds")

# prediction process
train_predict = model.predict(X_train)
val_predict = model.predict(X_val)
test_predict = model.predict(X_test)


# inverse_normalization
def inverse_transform(predictions, scaler):
    if len(predictions.shape) == 1:
        predictions = predictions.reshape(-1, 1)
    return scaler.inverse_transform(predictions)

train_predict_inverse = inverse_transform(train_predict, target_scaler)
val_predict_inverse = inverse_transform(val_predict, target_scaler)
test_predict_inverse = inverse_transform(test_predict, target_scaler)

Y_train_inverse = inverse_transform(Y_train.reshape(-1, 1), target_scaler)
Y_val_inverse = inverse_transform(Y_val.reshape(-1, 1), target_scaler)
Y_test_inverse = inverse_transform(Y_test.reshape(-1, 1), target_scaler)

# evaluate performance
train_rmse = np.sqrt(mean_squared_error(Y_train_inverse, train_predict_inverse))
val_rmse = np.sqrt(mean_squared_error(Y_val_inverse, val_predict_inverse))
test_rmse = np.sqrt(mean_squared_error(Y_test_inverse, test_predict_inverse))
train_r2 = r2_score(Y_train_inverse, train_predict_inverse)
val_r2 = r2_score(Y_val_inverse, val_predict_inverse)
test_r2 = r2_score(Y_test_inverse, test_predict_inverse)

print(f'Train RMSE: {train_rmse:.2f}')
print(f'Validation RMSE: {val_rmse:.2f}')
print(f'Test RMSE: {test_rmse:.2f}')
print(f'Train R² Score: {train_r2:.2f}')
print(f'Validation R² Score: {val_r2:.2f}')
print(f'Test R² Score: {test_r2:.2f}')

# save scaler and model parameters
joblib.dump(feature_scaler, "feature_scaler_single-step.pkl")
joblib.dump(target_scaler, "target_scaler_single-step.pkl")
model.save("single-step_model.h5")

# plot model structure
from tensorflow.keras.utils import plot_model

plot_model(model, to_file='model_structure_single-step.png', show_shapes=True, show_layer_names=True)

# train and validation loss
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Actual vs Predicted
all_mean_temp = filtered_data['mean_temp'].values

plt.figure(figsize=(12, 6))
plt.plot(all_mean_temp[inputsteps:], label='Actual', alpha=0.6)
train_offset = inputsteps
val_offset = train_offset + len(train_predict)
test_offset = val_offset + len(val_predict)

plt.plot(range(train_offset, val_offset), train_predict_inverse, label='Train Predict', alpha=0.8)
plt.plot(range(val_offset, test_offset), val_predict_inverse, label='Val Predict', alpha=0.8)
plt.plot(range(test_offset, test_offset + len(test_predict)), test_predict_inverse, label='Test Predict', alpha=0.8)
plt.xlabel("Time Steps")
plt.ylabel("Temperature (°C)")
plt.title(f"Actual vs Predicted Temperature (R²={test_r2:.2f})")
plt.legend()
plt.grid(True)
plt.show()

# Actual vs Predicted in test set
plt.figure(figsize=(12, 6))
plt.plot(Y_test_inverse.flatten(), label='Actual', color='blue', linewidth=2)
plt.plot(test_predict_inverse.flatten(), label='Predicted', color='red', linestyle='--', linewidth=2)
plt.xlabel('Time Steps')
plt.ylabel('Mean Temperature')
plt.title(f'Test Set: Actual vs Predicted\n(R² = {test_r2:.2f}, RMSE = {test_rmse:.2f})')
plt.legend()
plt.grid(True)
plt.show()

# residual
plt.figure(figsize=(8, 5))
residuals = Y_test_inverse.flatten() - test_predict_inverse.flatten()
sns.histplot(residuals, bins=50, kde=True)
plt.xlabel("Prediction Error (°C)")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# scatter plot
plt.figure(figsize=(8, 8))
sns.scatterplot(x=Y_test_inverse.flatten(), y=test_predict_inverse.flatten(), alpha=0.5)

# y=x perfect prediction
min_val = min(Y_test_inverse.min(), test_predict_inverse.min())
max_val = max(Y_test_inverse.max(), test_predict_inverse.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="dashed", color="black", label="y = x (Perfect Prediction)")

plt.xlabel("Actual Mean Temp (°C)")
plt.ylabel("Predicted Mean Temp (°C)")
plt.title("Scatter Plot: Actual vs Predicted Temperature")
plt.legend()
plt.show()
