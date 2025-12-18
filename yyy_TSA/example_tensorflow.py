import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# 生成一个模拟的时间序列数据
date_range = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
n = len(date_range)
np.random.seed(42)
trend = np.linspace(0, 100, n)  # 模拟趋势
seasonality = 10 * np.sin(np.linspace(0, 3.14 * 2, n))  # 模拟季节性
noise = np.random.normal(scale=5, size=n)  # 模拟噪音
data = trend + seasonality + noise
df = pd.DataFrame({'ds': date_range, 'y': data})

# 使用 Prophet 模型进行趋势和季节性建模
prophet_model = Prophet()
prophet_model.fit(df)
future = prophet_model.make_future_dataframe(periods=0)
forecast = prophet_model.predict(future)

# 可视化 Prophet 的预测结果
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Actual Data')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet Prediction', linestyle='--')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3)
plt.title('Prophet Prediction vs Actual Data')
plt.legend()
plt.show()

# 计算 Prophet 的残差
df['residual'] = df['y'] - forecast['yhat']

# 准备数据并训练 LSTM 模型来预测残差
scaler = MinMaxScaler(feature_range=(0, 1))
residual_scaled = scaler.fit_transform(df['residual'].values.reshape(-1, 1))

# 创建LSTM输入格式
X = []
y = []
look_back = 10  # 使用过去10天的数据来预测残差
for i in range(len(residual_scaled) - look_back):
    X.append(residual_scaled[i:i + look_back, 0])
    y.append(residual_scaled[i + look_back, 0])
X, y = np.array(X), np.array(y)

X = X.reshape((X.shape[0], X.shape[1], 1))

# 构建 LSTM 模型
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=False, input_shape=(X.shape[1], 1)))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# 训练 LSTM 模型
lstm_model.fit(X, y, epochs=20, batch_size=32, verbose=1)

# 使用 LSTM 模型进行残差预测
predicted_residual = lstm_model.predict(X)
predicted_residual = scaler.inverse_transform(predicted_residual)

# 合并 Prophet 的预测和 LSTM 的残差预测
final_prediction = forecast['yhat'] + np.concatenate([np.zeros(look_back), predicted_residual[:, 0]])

# 计算并打印均方误差
mse = mean_squared_error(df['y'][look_back:], final_prediction[look_back:])
print(f'Mean Squared Error: {mse}')

# 可视化最终的预测结果
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='Actual Data')
plt.plot(forecast['ds'], final_prediction, label='Final Prediction (Prophet + LSTM)', linestyle='--')
plt.title('Final Prediction (Prophet + LSTM) vs Actual Data')
plt.legend()
plt.show()
