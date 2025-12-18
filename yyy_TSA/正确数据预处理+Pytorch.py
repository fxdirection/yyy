import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import plotly.graph_objects as go
import plotly.offline as py
import holidays
import warnings

# 忽略警告信息
warnings.filterwarnings('ignore')

# 设置随机种子以确保可重复性
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set(style="whitegrid")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# 检查GPU可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

def load_turnstile_data(years=None):
    """
    加载多年的地铁闸机数据
    参数:
        years (list): 需要加载的年份列表，默认为所有可用数据(2014-2018)
    返回:
        DataFrame: 合并后的数据框
    """
    if years is None:
        years = ['2014', '2015', '2016', '2017', '2018']

    all_data = []
    desktop_path = "C:/Users/fangxiang/Desktop/yyy_TSA/TS/"

    for year in years:
        file_name = f'turnstile-usage-data-{year}.csv'
        file_path = os.path.join(desktop_path, file_name)

        if os.path.exists(f'{file_path}.csv'):
            df = pd.read_csv(f'{file_path}.csv')
        elif os.path.exists(file_path):
            df = pd.read_csv(file_path)
        elif os.path.exists(f'{file_path}.txt'):
            df = pd.read_csv(f'{file_path}.txt')
        else:
            print(f"警告: 找不到{year}年的数据文件")
            continue

        print(f"成功加载{year}年数据: {df.shape[0]}行, {df.shape[1]}列")
        all_data.append(df)

    if not all_data:
        raise FileNotFoundError("未能找到任何数据文件")

    df = pd.concat(all_data, ignore_index=True)
    print(f"合并后数据总量: {df.shape[0]}行, {df.shape[1]}列")
    return df

def preprocess_data(df):
    """
    数据预处理
    参数:
        df (DataFrame): 原始数据框
    返回:
        DataFrame: 预处理后的数据框
    """
    print("开始数据预处理...")

    df.columns = df.columns.str.strip()
    print("修正后的列名:", df.columns.tolist())

    df['Date'] = pd.to_datetime(df['Date'])
    df['Datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['Time'])

    df['Entries'] = pd.to_numeric(df['Entries'], errors='coerce')
    df['Exits'] = pd.to_numeric(df['Exits'], errors='coerce')

    print("计算客流量增量...")
    df = df.sort_values(by=['C/A', 'Unit', 'SCP', 'Datetime'])

    df['ENTRIES_DIFF'] = df.groupby(['C/A', 'Unit', 'SCP'])['Entries'].diff()
    df['EXITS_DIFF'] = df.groupby(['C/A', 'Unit', 'SCP'])['Exits'].diff()

    df['ENTRIES_DIFF'] = df['ENTRIES_DIFF'].clip(lower=0)
    df['EXITS_DIFF'] = df['EXITS_DIFF'].clip(lower=0)

    max_reasonable_count = 10000
    df['ENTRIES_DIFF'] = df['ENTRIES_DIFF'].clip(upper=max_reasonable_count)
    df['EXITS_DIFF'] = df['EXITS_DIFF'].clip(upper=max_reasonable_count)

    df['DOW'] = df['Datetime'].dt.dayofweek
    df['HOUR'] = df['Datetime'].dt.hour
    df['MONTH'] = df['Datetime'].dt.month
    df['YEAR'] = df['Datetime'].dt.year

    df = df.dropna(subset=['ENTRIES_DIFF', 'EXITS_DIFF'])

    print(f"预处理后数据量: {df.shape[0]}行")
    return df

def aggregate_data(df, freq='D'):
    """
    将数据聚合到指定频率
    参数:
        df (DataFrame): 预处理后的数据框
        freq (str): 聚合频率，'D'表示按天，'H'表示按小时
    返回:
        DataFrame: 聚合后的数据框
    """
    print(f"聚合数据到{freq}频率...")

    date_col = 'Date' if freq == 'D' else 'Datetime'

    df_agg = df.groupby([date_col, 'Station']).agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum'
    }).reset_index()

    df_agg['TOTAL_TRAFFIC'] = df_agg['ENTRIES_DIFF'] + df_agg['EXITS_DIFF']

    print(f"聚合后数据量: {df_agg.shape[0]}行")
    return df_agg

def prepare_time_series(df_agg, target_station=None):
    """
    准备时间序列数据用于Prophet模型
    参数:
        df_agg (DataFrame): 聚合后的数据框
        target_station (str): 指定分析的目标站点名称，默认为None
    返回:
        DataFrame: 符合Prophet要求格式的数据框
    """
    print("准备时间序列数据...")

    if target_station:
        df_station = df_agg[df_agg['Station'] == target_station].copy()
        if df_station.empty:
            print(f"警告: 未找到站点 '{target_station}'，将使用总体数据")
            df_station = df_agg.copy()
    else:
        df_station = df_agg.copy()

    ts_data = df_station.groupby('Date').agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum',
        'TOTAL_TRAFFIC': 'sum'
    }).reset_index()

    prophet_df = ts_data.rename(columns={'Date': 'ds', 'TOTAL_TRAFFIC': 'y'})

    print(f"时间序列数据范围: {prophet_df['ds'].min()} 到 {prophet_df['ds'].max()}")
    return prophet_df

def add_holidays(df):
    """
    添加美国假日信息到数据框
    参数:
        df (DataFrame): 包含时间序列的数据框
    返回:
        DataFrame: 包含假日信息的DataFrame
    """
    print("添加假日信息...")

    df['ds'] = pd.to_datetime(df['ds'])
    start_year = df['ds'].min().year
    end_year = df['ds'].max().year

    us_holidays = holidays.US(years=range(start_year, end_year + 1))

    holiday_df = pd.DataFrame(
        [(pd.Timestamp(date), name) for date, name in us_holidays.items()],
        columns=['ds', 'holiday']
    )

    holiday_df['lower_window'] = -1
    holiday_df['upper_window'] = 1

    important_holidays = ['New Year', 'Independence Day', 'Thanksgiving', 'Christmas Day']
    for holiday in important_holidays:
        mask = holiday_df['holiday'].str.contains(holiday, case=False, na=False)
        holiday_df.loc[mask, 'lower_window'] = -2
        holiday_df.loc[mask, 'upper_window'] = 2

    print("假日数据预览:")
    print(holiday_df.head())
    print("假日数据列名:", holiday_df.columns.tolist())
    return holiday_df

def build_prophet_model(data, target_col='y', test_size=0.2, holidays_df=None):
    """
    构建Prophet模型用于预测趋势和季节性成分
    参数:
        data (DataFrame): 时间序列数据
        target_col (str): 预测目标列名
        test_size (float): 测试集比例
        holidays_df (DataFrame): 假日数据框
    返回:
        tuple: (Prophet模型, 预测结果, 包含预测和残差的完整数据, Prophet训练MSE)
    """
    print("构建Prophet模型...")

    prophet_data = data.reset_index()
    prophet_data = prophet_data.rename(columns={'DATETIME': 'ds', target_col: 'y'})

    train_size = int(len(prophet_data) * (1 - test_size))
    train_data = prophet_data.iloc[:train_size].copy()
    test_data = prophet_data.iloc[train_size:].copy()

    model = Prophet(
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        seasonality_mode='multiplicative',
        daily_seasonality=True,
        weekly_seasonality=True,
        yearly_seasonality=True
    )

    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

    if holidays_df is not None:
        model.add_country_holidays(country_name='US')
        model.holidays = holidays_df

    model.fit(train_data)

    future = model.make_future_dataframe(periods=len(test_data), freq='D')
    forecast = model.predict(future)

    train_forecast = forecast.iloc[:train_size].copy()
    train_data['yhat'] = train_forecast['yhat'].values
    train_data['residual'] = train_data['y'] - train_data['yhat']


    test_forecast = forecast.iloc[train_size:].copy()
    test_data['yhat'] = test_forecast['yhat'].values

    full_data = pd.concat([train_data, test_data], ignore_index=True)

    prophet_train_mse = mean_squared_error(train_data['y'], train_data['yhat'])

    return model, forecast, full_data, prophet_train_mse

def create_lstm_dataset(data, lookback=7):
    """
    为LSTM准备时间序列数据（滑动窗口）
    参数:
        data (array): 输入数据
        lookback (int): 回溯时间步长
    返回:
        tuple: (特征数组, 目标数组)
    """
    X, y = [], []  # 初始化为列表
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i, 0])  # 添加到列表
        y.append(data[i])  # 添加到列表
    X = np.array(X)  # 循环结束后转换为 NumPy 数组
    y = np.array(y)  # 循环结束后转换为 NumPy 数组
    X = X.reshape((X.shape[0], X.shape[1], 1))  # 为 LSTM 重塑形状
    return X, y

class TimeSeriesDataset(Dataset):
    """时间序列数据集，用于 LSTM 输入"""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X).to(device)
        self.y = torch.FloatTensor(y).to(device)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class LSTMModel(nn.Module):
    """LSTM 模型定义"""
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def build_lstm_model(data, residual_col='residual', test_size=0.2, lookback=7, output_dir="output_models"):
    """
    构建LSTM模型用于预测残差序列
    参数:
        data (DataFrame): 包含残差的数据框
        residual_col (str): 残差列名
        test_size (float): 测试集比例
        lookback (int): 回溯时间步长
        output_dir (str): 模型保存目录
    返回:
        dict: 包含模型、预测结果、训练损失等信息的字典
    """
    print("构建LSTM残差模型...")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    save_path = os.path.join(output_dir, 'best_lstm_model.pth')

    data = data.sort_values('ds')
    residuals = data[residual_col].values.reshape(-1, 1)

    print("残差数据统计:", pd.Series(residuals.flatten()).describe())
    print("残差数据是否包含NaN:", pd.Series(residuals.flatten()).isna().sum())

    scaler = MinMaxScaler(feature_range=(0, 1))
    residuals_scaled = scaler.fit_transform(residuals)

    X, y = create_lstm_dataset(residuals_scaled, lookback=lookback)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # 与Prophet的train_size对齐
    prophet_train_size = int(len(X) * (1 - test_size))
    train_size = prophet_train_size
    # train_size = prophet_train_size - lookback
    # if train_size <= 0:
    #     raise ValueError("训练集大小不足，请减少lookback或增加数据量")

    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    y_train, y_test = y_train.flatten(), y_test.flatten()
    print(f"LSTM训练集大小: {len(X_train)}, 测试集大小: {len(X_test)}")

    train_dataset = TimeSeriesDataset(X_train, y_train)
    test_dataset = TimeSeriesDataset(X_test, y_test)

    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 20
    train_losses = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(train_dataset)
        train_losses.append(epoch_loss)
        print(f'轮次 {epoch + 1}/{num_epochs}, 训练损失: {epoch_loss:.6f}')

    model.eval()
    with torch.no_grad():
        train_predict = model(torch.FloatTensor(X_train).to(device)).cpu().numpy()
        test_predict = model(torch.FloatTensor(X_test).to(device)).cpu().numpy()

    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)

    torch.save(model.state_dict(), save_path)
    print(f"模型已保存到 {save_path}")

    lstm_results = {
        'model': model,
        'scaler': scaler,
        'train_predict': train_predict,
        'test_predict': test_predict,
        'y_train': y_train_inv,
        'y_test': y_test_inv,
        'lookback': lookback,
        'train_losses': train_losses
    }

    return lstm_results

def hybrid_forecast(prophet_forecast, lstm_results, prophet_data, test_size=0.2, weights=None):
    """
    融合Prophet和LSTM预测结果
    参数:
        prophet_forecast (DataFrame): Prophet模型预测结果
        lstm_results (dict): LSTM模型预测结果
        prophet_data (DataFrame): Prophet输入数据
        test_size (float): 测试集比例
        weights (float): 融合权重，None时自动优化
    返回:
        dict: 包含混合预测结果和评估指标的字典
    """
    train_size = int(len(prophet_data) * (1 - test_size))-1
    train_data = prophet_data.iloc[:train_size].copy()
    test_data = prophet_data.iloc[train_size:].copy()

    lstm_train_residuals = lstm_results['train_predict'].flatten()
    lstm_test_residuals = lstm_results['test_predict'].flatten()

    lookback = lstm_results['lookback']
    prophet_train_preds = train_data['yhat'].values[lookback:]
    prophet_test_preds = test_data['yhat'].values

    train_actual = train_data['y'].values[lookback:]
    test_actual = test_data['y'].values

    # # 确保lstm_train_residuals与prophet_train_preds长度一致
    # if len(lstm_train_residuals) > len(prophet_train_preds):
    #     lstm_train_residuals = lstm_train_residuals[:len(prophet_train_preds)]
    # elif len(lstm_train_residuals) < len(prophet_train_preds):
    #     prophet_train_preds = prophet_train_preds[:len(lstm_train_residuals)]
    #     train_actual = train_actual[:len(lstm_train_residuals)]
    #
    # # 确保lstm_test_residuals与prophet_test_preds长度一致
    # if len(lstm_test_residuals) > len(prophet_test_preds):
    #     lstm_test_residuals = lstm_test_residuals[:len(prophet_test_preds)]
    # elif len(lstm_test_residuals) < len(prophet_test_preds):
    #     prophet_test_preds = prophet_test_preds[:len(lstm_test_residuals)]
    #     test_actual = test_actual[:len(lstm_test_residuals)]

    print(f"训练集预测长度: Prophet={len(prophet_train_preds)}, LSTM={len(lstm_train_residuals)}, Actual={len(train_actual)}")
    print(f"测试集预测长度: Prophet={len(prophet_test_preds)}, LSTM={len(lstm_test_residuals)}, Actual={len(test_actual)}")

    if weights is None:
        best_weight = optimize_weights(prophet_train_preds, lstm_train_residuals, train_actual)
    else:
        best_weight = weights

    hybrid_train_preds = prophet_train_preds * best_weight + (prophet_train_preds + lstm_train_residuals) * (1 - best_weight)
    hybrid_test_preds = prophet_test_preds * best_weight + (prophet_test_preds + lstm_test_residuals) * (1 - best_weight)

    prophet_train_rmse = np.sqrt(mean_squared_error(train_actual, prophet_train_preds))
    prophet_test_rmse = np.sqrt(mean_squared_error(test_actual, prophet_test_preds))
    hybrid_train_rmse = np.sqrt(mean_squared_error(train_actual, hybrid_train_preds))
    hybrid_test_rmse = np.sqrt(mean_squared_error(test_actual, hybrid_test_preds))
    hybrid_train_mae = mean_absolute_error(train_actual, hybrid_train_preds)
    hybrid_test_mae = mean_absolute_error(test_actual, hybrid_test_preds)

    try:
        hybrid_train_mape = mean_absolute_percentage_error(train_actual, hybrid_train_preds) * 100
        hybrid_test_mape = mean_absolute_percentage_error(test_actual, hybrid_test_preds) * 100
    except:
        hybrid_train_mape = np.nan
        hybrid_test_mape = np.nan

    actual = np.concatenate([train_actual, test_actual])
    hybrid_pred = np.concatenate([hybrid_train_preds, hybrid_test_preds])
    prophet_pred = np.concatenate([prophet_train_preds, prophet_test_preds])

    results = {
        'prophet_train_preds': prophet_train_preds,
        'prophet_test_preds': prophet_test_preds,
        'lstm_train_residuals': lstm_train_residuals,
        'lstm_test_residuals': lstm_test_residuals,
        'hybrid_train_preds': hybrid_train_preds,
        'hybrid_test_preds': hybrid_test_preds,
        'train_actual': train_actual,
        'test_actual': test_actual,
        'best_weight': best_weight,
        'metrics': {
            'prophet_train_rmse': prophet_train_rmse,
            'prophet_test_rmse': prophet_test_rmse,
            'hybrid_train_rmse': hybrid_train_rmse,
            'hybrid_test_rmse': hybrid_test_rmse,
            'hybrid_train_mae': hybrid_train_mae,
            'hybrid_test_mae': hybrid_test_mae,
            'hybrid_train_mape': hybrid_train_mape,
            'hybrid_test_mape': hybrid_test_mape
        },
        'actual': actual,
        'hybrid_pred': hybrid_pred,
        'prophet_pred': prophet_pred
    }

    return results

def optimize_weights(prophet_preds, lstm_residuals, actual, step=0.1):
    """
    优化Prophet和LSTM混合权重
    参数:
        prophet_preds (array): Prophet预测值
        lstm_residuals (array): LSTM残差预测值
        actual (array): 实际值
        step (float): 权重搜索步长
    返回:
        float: 最佳权重
    """
    best_rmse = float('inf')
    best_weight = 0.5

    for weight in np.arange(0, 1.01, step):
        hybrid_preds = prophet_preds * weight + (prophet_preds + lstm_residuals) * (1 - weight)
        rmse = np.sqrt(mean_squared_error(actual, hybrid_preds))
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = weight

    return best_weight

def plot_hybrid_results(hybrid_results, station_name, output_dir="Prophet+LSTM", title=None):
    """
    绘制混合模型的预测结果对比
    参数:
        hybrid_results (dict): 混合模型预测结果
        station_name (str): 站点名称
        output_dir (str): 输出目录
        title (str): 图表标题
    """
    if title is None:
        title = f"{station_name} 车站客流量预测（Prophet-LSTM混合模型）"

    train_actual = hybrid_results['train_actual']
    test_actual = hybrid_results['test_actual']
    prophet_train = hybrid_results['prophet_train_preds']
    prophet_test = hybrid_results['prophet_test_preds']
    hybrid_train = hybrid_results['hybrid_train_preds']
    hybrid_test = hybrid_results['hybrid_test_preds']

    plt.figure(figsize=(15, 6))
    train_indices = np.arange(len(train_actual))
    plt.plot(train_indices, train_actual, label='训练集实际值', color='blue')
    plt.plot(train_indices, prophet_train, label='Prophet预测（训练集）', color='green', linestyle='--')
    plt.plot(train_indices, hybrid_train, label='混合模型预测（训练集）', color='red', linestyle='-.')

    test_indices = np.arange(len(train_actual), len(train_actual) + len(test_actual))
    plt.plot(test_indices, test_actual, label='测试集实际值', color='blue')
    plt.plot(test_indices, prophet_test, label='Prophet预测（测试集）', color='green', linestyle='--')
    plt.plot(test_indices, hybrid_test, label='混合模型预测（测试集）', color='red', linestyle='-.')

    plt.axvline(x=len(train_actual), color='gray', linestyle='-', label='测试集开始')
    plt.text(len(train_actual) + 1, plt.ylim()[1] * 0.9, '测试 训练集大小: {len(train_actual)}', fontsize=12)

    plt.title(title)
    plt.xlabel('时间索引')
    plt.ylabel('客流量')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{station_name}_hybrid_results.png"), dpi=300)
    plt.close()

    train_dates = pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=len(test_actual)),
                                periods=len(train_actual), freq='D')
    test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=len(test_actual), freq='D')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=list(train_dates) + list(test_dates),
        y=list(train_actual) + list(test_actual),
        mode='lines',
        name='实际客流量',
        line=dict(color='royalblue', width=2)
    ))
    fig.add_trace(go.Scatter(
        x=list(train_dates) + list(test_dates),
        y=list(prophet_train) + list(prophet_test),
        mode='lines',
        name='Prophet预测',
        line=dict(color='forestgreen', width=1.5, dash='dash')
    ))
    fig.add_trace(go.Scatter(
        x=list(train_dates) + list(test_dates),
        y=list(hybrid_train) + list(hybrid_test),
        mode='lines',
        name='混合模型预测',
        line=dict(color='firebrick', width=1.5)
    ))

    fig.add_vline(
        x=train_dates[-1], line_width=1.5, line_dash="dash", line_color="gray",
        annotation_text="测试集开始",
        annotation_position="top right"
    )

    fig.update_layout(
        title=title,
        xaxis_title='日期',
        yaxis_title='客流量',
        legend_title='数据类型',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        width=1200,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7天", step="day", stepmode="backward"),
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all", label="全部")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f"{station_name}_hybrid_results.html"))
    fig.show()

def create_interactive_forecast_comparison(df_agg, target_stations, prophet_forecast, lstm_results, prophet_data, output_dir="Comparison", forecast_periods=90):
    """
    创建交互式预测对比图，包含实际值和预测值
    参数:
        df_agg (DataFrame): 聚合后的数据框
        target_stations (list): 需要分析的站点列表
        prophet_forecast (DataFrame): Prophet预测结果
        lstm_results (dict): LSTM预测结果
        prophet_data (DataFrame): Prophet输入数据
        forecast_periods (int): 预测天数
    """
    print("创建交互式预测对比图...")

    all_stations = df_agg.groupby('Date').agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum'
    }).reset_index()
    all_stations['Station'] = 'ALL STATIONS'

    selected_stations = df_agg[df_agg['Station'].isin(target_stations)].copy()
    combined_df = pd.concat([selected_stations, all_stations], ignore_index=True)

    all_forecasts = {}

    for station in target_stations + ['ALL STATIONS']:
        station_data = combined_df[combined_df['Station'] == station].copy()
        entries_df = station_data[['Date', 'ENTRIES_DIFF']].rename(columns={'Date': 'ds', 'ENTRIES_DIFF': 'y'}).dropna()
        exits_df = station_data[['Date', 'EXITS_DIFF']].rename(columns={'Date': 'ds', 'EXITS_DIFF': 'y'}).dropna()

        holidays_df = add_holidays(entries_df)

        try:
            entries_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                seasonality_mode='multiplicative'
            ).fit(entries_df)
            entries_future = entries_model.make_future_dataframe(periods=forecast_periods)
            entries_forecast = entries_model.predict(entries_future)

            exits_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                seasonality_mode='multiplicative'
            ).fit(exits_df)
            exits_future = exits_model.make_future_dataframe(periods=forecast_periods)
            exits_forecast = exits_model.predict(exits_future)

            forecast_df = pd.merge(
                entries_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'yhat': 'entries_pred',
                    'yhat_lower': 'entries_lower',
                    'yhat_upper': 'entries_upper'
                }),
                exits_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                    'yhat': 'exits_pred',
                    'yhat_lower': 'exits_lower',
                    'yhat_upper': 'exits_upper'
                }),
                on='ds',
                how='outer'
            )

            all_forecasts[station] = {
                'data': station_data,
                'forecast': forecast_df
            }

            print(f"{station} 站点预测完成")
        except Exception as e:
            print(f"训练 {station} 站点模型时出错: {str(e)}")
            continue

    fig = go.Figure()
    buttons = []

    for i, station in enumerate(target_stations + ['ALL STATIONS']):
        if station not in all_forecasts:
            continue

        data = all_forecasts[station]['data']
        forecast = all_forecasts[station]['forecast']

        data['Date'] = pd.to_datetime(data['Date'])
        forecast['ds'] = pd.to_datetime(forecast['ds'])

        merged = pd.merge(
            data[['Date', 'ENTRIES_DIFF', 'EXITS_DIFF']],
            forecast,
            left_on='Date',
            right_on='ds',
            how='outer'
        ).sort_values('Date')

        is_visible = (i == 0)

        fig.add_trace(go.Scatter(
            x=merged['Date'],
            y=merged['ENTRIES_DIFF'],
            name=f'{station} - 实际入站',
            visible=is_visible,
            line=dict(color='blue', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>实际入站: %{y:,}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=merged['ds'],
            y=merged['entries_pred'],
            name=f'{station} - 预测入站',
            visible=is_visible,
            line=dict(color='blue', dash='dot', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>预测入站: %{y:,}<extra></extra>',
            connectgaps=True
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([merged['ds'], merged['ds'][::-1]]),
            y=pd.concat([merged['entries_upper'], merged['entries_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(0, 100, 255, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo='skip',
            name='入站95%置信区间',
            visible=is_visible,
            showlegend=False
        ))

        fig.add_trace(go.Scatter(
            x=merged['Date'],
            y=merged['EXITS_DIFF'],
            name=f'{station} - 实际出站',
            visible=is_visible,
            line=dict(color='red', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>实际出站: %{y:,}<extra></extra>'
        ))

        fig.add_trace(go.Scatter(
            x=merged['ds'],
            y=merged['exits_pred'],
            name=f'{station} - 预测出站',
            visible=is_visible,
            line=dict(color='red', dash='dot', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>预测出站: %{y:,}<extra></extra>',
            connectgaps=True
        ))

        fig.add_trace(go.Scatter(
            x=pd.concat([merged['ds'], merged['ds'][::-1]]),
            y=pd.concat([merged['exits_upper'], merged['exits_lower'][::-1]]),
            fill='toself',
            fillcolor='rgba(255, 100, 0, 0.2)',
            line=dict(color='rgba(255, 255, 255, 0)'),
            hoverinfo='skip',
            name='出站95%置信区间',
            visible=is_visible,
            showlegend=False
        ))

        buttons.append(dict(
            label=station,
            method='update',
            args=[{
                'visible': [trace.visible if j // 6 != i else True for j, trace in enumerate(fig.data)]
            }, {
                'title': f'{station} 站流量 - 实际 vs 预测'
            }]
        ))

    fig.update_layout(
        title=f'{target_stations[0]} 站流量 - 实际 vs 预测',
        xaxis_title='日期',
        yaxis_title='客流量',
        hovermode='x unified',
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15,
            'xanchor': 'left',
            'yanchor': 'top'
        }],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=700,
        margin=dict(l=50, r=50, b=100, t=100)
    )

    if output_dir:
        py.plot(fig, filename=os.path.join(output_dir, 'interactive_forecast_comparison.html'), auto_open=False)
    fig.show()

def print_metrics(hybrid_results):
    """
    打印评估指标
    参数:
        hybrid_results (dict): 混合模型预测结果
    """
    metrics = hybrid_results['metrics']
    print('\n预测评估指标:')
    print('-' * 50)
    print(f"Prophet训练集RMSE: {metrics['prophet_train_rmse']:.2f}")
    print(f"Prophet测试集RMSE: {metrics['prophet_test_rmse']:.2f}")
    print(f"混合模型训练集RMSE: {metrics['hybrid_train_rmse']:.2f}")
    print(f"混合模型测试集RMSE: {metrics['hybrid_test_rmse']:.2f}")
    print(f"混合模型训练集MAE: {metrics['hybrid_train_mae']:.2f}")
    print(f"混合模型测试集MAE: {metrics['hybrid_test_mae']:.2f}")
    print(f"混合模型训练集MAPE: {metrics['hybrid_train_mape']:.2f}%")
    print(f"混合模型测试集MAPE: {metrics['hybrid_test_mape']:.2f}%")
    print(f"最佳混合权重: {hybrid_results['best_weight']:.2f}")
    print('-' * 50)

def main():
    """
    主函数，运行整个预测流程
    """
    try:
        if not os.path.exists("daily_results.csv"):
            # 1. 加载数据
            print("加载数据...")
            df = load_turnstile_data()

            # 2. 数据预处理
            print("预处理数据...")
            df_processed = preprocess_data(df)

            # 3. 按天聚合数据
            print("聚合数据...")
            df_daily = aggregate_data(df_processed, freq='D')
            df_daily.to_csv("daily_results.csv", index=False)
            exit()
        else:
            df_daily = pd.read_csv("daily_results.csv")

        # 4. 选择目标站点
        target_station = "34 ST-PENN STA"
        print(f"选择站点 {target_station} 进行分析...")
        prophet_df = prepare_time_series(df_daily, target_station=target_station)

        # 5. 添加假日信息
        print("添加假日信息...")
        holidays_df = add_holidays(prophet_df)
        prophet_df.to_csv("prophet_forecast_results.csv", index=False)

        # 6. 构建Prophet模型
        print("训练Prophet模型...")
        prophet_model, prophet_forecast, prophet_data, prophet_train_mse = build_prophet_model(prophet_df, test_size=0.2, holidays_df=holidays_df)

        # 7. 构建LSTM残差模型
        print("训练LSTM残差模型...")
        lstm_results = build_lstm_model(prophet_data, test_size=0.2, lookback=7)

        # 8. 混合模型预测
        print("融合模型结果...")
        hybrid_results = hybrid_forecast(prophet_forecast, lstm_results, prophet_data, test_size=0.2)

        # 9. 计算混合模型测试集MSE
        hybrid_test_mse = mean_squared_error(hybrid_results['test_actual'], hybrid_results['hybrid_test_preds'])

        # 10. 打印评估指标
        print_metrics(hybrid_results)

        # 11. 输出训练损失和均方误差（带标签）
        print("\n训练损失和均方误差:")
        print('-' * 50)
        print(f"Prophet Train MSE: {prophet_train_mse:.6f}")
        for epoch, loss in enumerate(lstm_results['train_losses'], 1):
            print(f"LSTM Train Loss Epoch {epoch}: {loss:.6f}")
        print(f"Hybrid Test MSE: {hybrid_test_mse:.6f}")
        print('-' * 50)

        # 12. 可视化结果
        print("生成可视化结果...")
        output_dir = "output_plots"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plot_hybrid_results(hybrid_results, target_station, output_dir=output_dir)

        target_stations = ["14 ST-UNION SQ", "34 ST-HERALD SQ", "34 ST-PENN STA"]
        create_interactive_forecast_comparison(df_daily, target_stations, prophet_forecast, lstm_results, prophet_data)

        # 13. 保存预测结果
        print("保存预测结果...")
        forecast_df = pd.DataFrame({
            'ds': prophet_data['ds'],
            'actual': hybrid_results['actual'],
            'prophet_pred': hybrid_results['prophet_pred'],
            'hybrid_pred': hybrid_results['hybrid_pred']
        })
        forecast_df.to_csv(os.path.join(output_dir, 'hybrid_forecast_results.csv'), index=False)
        print(f"预测结果已保存到 {os.path.join(output_dir, 'hybrid_forecast_results.csv')}")

        print("\n分析完成！已生成可视化图表和预测结果。")

    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

