import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# 定义 PyTorch 自定义数据集
class TimeSeriesDataset(Dataset):
    """时间序列数据集，用于 LSTM 输入"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义 PyTorch LSTM 模型
class LSTMModel(nn.Module):
    """LSTM 模型定义"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=1, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def calculate_hybrid_confidence_interval(prophet_forecast, predicted_residual, look_back, confidence_level=0.95):
    """
    计算混合模型的置信区间
    参数:
        prophet_forecast (DataFrame): Prophet 预测结果，包含 yhat, yhat_lower, yhat_upper
        predicted_residual (array): LSTM 预测的残差
        look_back (int): LSTM 回溯时间步长
        confidence_level (float): 置信水平，默认为 95%
    返回:
        tuple: (混合预测下界, 混合预测上界)
    """
    prophet_lower = prophet_forecast['yhat_lower'].values
    prophet_upper = prophet_forecast['yhat_upper'].values
    prophet_yhat = prophet_forecast['yhat'].values
    residual_std = np.std(predicted_residual)
    z_score = 1.96
    residual_uncertainty = z_score * residual_std
    hybrid_lower = prophet_lower + np.concatenate([np.zeros(look_back), predicted_residual[:, 0] - residual_uncertainty])
    hybrid_upper = prophet_upper + np.concatenate([np.zeros(look_back), predicted_residual[:, 0] + residual_uncertainty])
    return hybrid_lower, hybrid_upper

# 生成模拟时间序列数据
date_range = pd.date_range(start='2020-01-01', end='2021-12-31', freq='D')
n = len(date_range)
trend = np.linspace(0, 100, n)
seasonality = 10 * np.sin(np.linspace(0, 3.14 * 2, n))
noise = np.random.normal(scale=5, size=n)
data = trend + seasonality + noise
df = pd.DataFrame({'ds': date_range, 'y': data})

# 划分训练集和测试集
test_size = 0.2
train_size = int(len(df) * (1 - test_size))
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# 使用 Prophet 模型进行趋势和季节性建模
print("训练 Prophet 模型...")
prophet_model = Prophet()
prophet_model.fit(train_df)
future = prophet_model.make_future_dataframe(periods=len(test_df), freq='D')
forecast = prophet_model.predict(future)

# 计算 Prophet 训练损失（MSE）
prophet_train_mse = mean_squared_error(train_df['y'], forecast['yhat'].iloc[:train_size])

# 可视化 Prophet 的预测结果
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='实际数据')
plt.plot(forecast['ds'], forecast['yhat'], label='Prophet 预测', linestyle='--')
plt.fill_between(forecast['ds'], forecast['yhat_lower'], forecast['yhat_upper'], color='orange', alpha=0.3, label='Prophet 95% 置信区间')
plt.title('Prophet 预测 vs 实际数据')
plt.xlabel('日期')
plt.ylabel('值')
plt.legend()
output_dir = "C:/Users/fangxiang/Desktop/yyy_TSA/examples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
plt.savefig(os.path.join(output_dir, 'prophet_prediction.png'), dpi=300)

# 计算 Prophet 的残差
df['residual'] = df['y'] - forecast['yhat']
train_residual = df['residual'].iloc[:train_size]
test_residual = df['residual'].iloc[train_size:]

# 准备数据并训练 LSTM 模型来预测残差
scaler = MinMaxScaler(feature_range=(0, 1))
residual_scaled = scaler.fit_transform(df['residual'].values.reshape(-1, 1))

# 创建 LSTM 输入格式
X = []
y = []
look_back = 10
for i in range(len(residual_scaled) - look_back):
    X.append(residual_scaled[i:i + look_back, 0])
    y.append(residual_scaled[i + look_back, 0])
X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# 划分 LSTM 训练和测试数据
train_size_lstm = int(len(X) * (1 - test_size))
X_train, X_test = X[:train_size_lstm], X[train_size_lstm:]
y_train, y_test = y[:train_size_lstm], y[train_size_lstm:]

# 创建数据集和数据加载器
train_dataset = TimeSeriesDataset(X_train, y_train)
test_dataset = TimeSeriesDataset(X_test, y_test)
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# 初始化模型、损失函数和优化器
lstm_model = LSTMModel(input_size=1, hidden_size=50, num_layers=1, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(lstm_model.parameters(), lr=0.001)

# 训练 LSTM 模型
print("训练 LSTM 模型...")
num_epochs = 20
lstm_train_losses = []
for epoch in range(num_epochs):
    lstm_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = lstm_model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_dataset)
    lstm_train_losses.append(epoch_loss)
    print(f'轮次 {epoch + 1}/{num_epochs}, 损失: {epoch_loss:.6f}')

# 使用 LSTM 模型进行残差预测
lstm_model.eval()
with torch.no_grad():
    X_tensor = torch.FloatTensor(X).to(device)
    predicted_residual_scaled = lstm_model(X_tensor).cpu().numpy()
predicted_residual = scaler.inverse_transform(predicted_residual_scaled)

# 合并 Prophet 的预测和 LSTM 的残差预测
final_prediction = forecast['yhat'].values + np.concatenate([np.zeros(look_back), predicted_residual[:, 0]])

# 计算混合模型的置信区间
hybrid_lower, hybrid_upper = calculate_hybrid_confidence_interval(forecast, predicted_residual, look_back)

# 计算混合模型测试集均方误差
hybrid_test_mse = mean_squared_error(df['y'][look_back:], final_prediction[look_back:])

# 可视化最终的预测结果
plt.figure(figsize=(10, 6))
plt.plot(df['ds'], df['y'], label='实际数据')
plt.plot(forecast['ds'], final_prediction, label='最终预测 (Prophet + LSTM)', linestyle='--')
plt.fill_between(forecast['ds'], hybrid_lower, hybrid_upper, color='green', alpha=0.3, label='混合模型 95% 置信区间')
plt.title('最终预测 (Prophet + LSTM) vs 实际数据')
plt.xlabel('日期')
plt.ylabel('值')
plt.legend()
plt.savefig(os.path.join(output_dir, 'hybrid_prediction.png'), dpi=300)

# 输出训练损失和均方误差（带标签）
print(f"Prophet Train MSE: {prophet_train_mse:.6f}")
print(f"Hybrid Test MSE: {hybrid_test_mse:.6f}")