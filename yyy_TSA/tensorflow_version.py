import pandas as pd
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings('ignore')

import tensorflow as tf

# 强制启用 GPU 加速（忽略 CPU）
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    tf.config.set_visible_devices(physical_devices[0], 'GPU')
    print("GPU 已启用")
else:
    print("未检测到 GPU，回退到 CPU")

# 1. 数据加载与预处理
def load_turnstile_data(file_paths):
    """
    加载MTA地铁闸机数据并合并
    """
    all_data = []

    for file_path in file_paths:
        df = pd.read_csv(file_path)
        all_data.append(df)

    # 合并所有数据集
    df_combined = pd.concat(all_data, ignore_index=True)

    return df_combined


def preprocess_data(df):
    """
    预处理MTA地铁闸机数据
    """
    df.columns = df.columns.str.strip()
    print("修正后的列名:", df.columns.tolist())

    # 转换日期和时间为datetime对象
    df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])

    # 排序数据
    df = df.sort_values(by=['C/A', 'Unit', 'SCP', 'Datetime'])

    # 计算每个闸机每个时间段的实际进出站人数（差值）
    df['PREV_ENTRIES'] = df.groupby(['C/A', 'Unit', 'SCP'])['Entries'].shift(1)
    df['PREV_EXITS'] = df.groupby(['C/A', 'Unit', 'SCP'])['Exits'].shift(1)

    # 计算进出站人数差值
    df['ENTRIES_DIFF'] = df['Entries'] - df['PREV_ENTRIES']
    df['EXITS_DIFF'] = df['Exits'] - df['PREV_EXITS']

    # 处理异常值（负值或过大值）
    df.loc[df['ENTRIES_DIFF'] < 0, 'ENTRIES_DIFF'] = 0
    df.loc[df['EXITS_DIFF'] < 0, 'EXITS_DIFF'] = 0
    df.loc[df['ENTRIES_DIFF'] > 5000, 'ENTRIES_DIFF'] = 0
    df.loc[df['EXITS_DIFF'] > 5000, 'EXITS_DIFF'] = 0

    # 删除缺失值
    df = df.dropna(subset=['ENTRIES_DIFF', 'EXITS_DIFF'])

    # 提取日期特征
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Datetime'].dt.hour
    df['Day'] = df['Datetime'].dt.day
    df['Month'] = df['Datetime'].dt.month
    df['Year'] = df['Datetime'].dt.year
    df['Dayofweek'] = df['Datetime'].dt.dayofweek  # 0=周一，6=周日

    return df


def aggregate_by_station(df, time_interval='D'):
    """
    按车站和时间间隔聚合数据
    time_interval: 'D'为日，'H'为小时
    """
    # 设置时间索引后重采样
    if time_interval == 'D':
        # 日聚合
        df_agg = df.groupby(['Station', pd.Grouper(key='Datetime', freq='D')])[
            ['ENTRIES_DIFF', 'EXITS_DIFF']].sum().reset_index()
    elif time_interval == 'H':
        # 小时聚合
        df_agg = df.groupby(['Station', pd.Grouper(key='Datetime', freq='H')])[
            ['ENTRIES_DIFF', 'EXITS_DIFF']].sum().reset_index()

    # 计算总流量（进站+出站）
    df_agg['TOTAL_FLOW'] = df_agg['ENTRIES_DIFF'] + df_agg['EXITS_DIFF']

    return df_agg


# 2. 特定车站数据提取和特征工程
def prepare_station_data(df_agg, station_name, target_col='TOTAL_FLOW'):
    """
    准备特定车站的数据用于时间序列预测
    """
    # 提取特定车站数据
    station_data = df_agg[df_agg['Station'] == station_name].copy()

    # 确保数据时间上是连续的
    station_data = station_data.set_index('Datetime').sort_index()

    # 重采样以确保没有缺失的时间点
    if 'D' in station_data.index.freq or station_data.index.freq is None:
        station_data = station_data.resample('D').sum()
    else:
        station_data = station_data.resample('H').sum()

    # 线性插值填充缺失值
    station_data = station_data.interpolate(method='linear')

    # 提取要预测的目标列
    ts_data = station_data[[target_col]].copy()

    # 添加时间特征
    ts_data['hour'] = ts_data.index.hour
    ts_data['day'] = ts_data.index.day
    ts_data['month'] = ts_data.index.month
    ts_data['year'] = ts_data.index.year
    ts_data['dayofweek'] = ts_data.index.dayofweek
    ts_data['is_weekend'] = (ts_data.index.dayofweek >= 5).astype(int)

    return ts_data


# 3. 构建Prophet模型
def build_prophet_model(data, target_col='TOTAL_FLOW', test_size=0.1):
    """
    构建Prophet模型用于预测趋势和季节性成分
    """
    # 准备Prophet所需的数据格式
    prophet_data = data.reset_index()
    prophet_data = prophet_data.rename(columns={'DATETIME': 'ds', target_col: 'y'})

    # 划分训练集和测试集
    train_size = int(len(prophet_data) * (1 - test_size))
    train_data = prophet_data.iloc[:train_size].copy()
    test_data = prophet_data.iloc[train_size:].copy()

    # 创建并训练Prophet模型
    model = Prophet(
        changepoint_prior_scale=0.05,  # 控制趋势变化的灵活性
        seasonality_prior_scale=10,  # 控制季节性强度
        seasonality_mode='additive',  # 加法季节性
        daily_seasonality=True,  # 启用日季节性
        weekly_seasonality=True,  # 启用周季节性
        yearly_seasonality=True  # 启用年季节性
    )

    # 添加美国假日
    model.add_country_holidays(country_name='US')

    # 训练模型
    model.fit(train_data)

    # 预测整个数据集
    future = model.make_future_dataframe(periods=len(test_data), freq='D')
    forecast = model.predict(future)

    # 提取训练集的拟合结果和残差
    train_forecast = forecast.iloc[:train_size].copy()
    train_data['yhat'] = train_forecast['yhat'].values
    train_data['residual'] = train_data['y'] - train_data['yhat']

    # 提取测试集的预测结果
    test_forecast = forecast.iloc[train_size:].copy()
    test_data['yhat'] = test_forecast['yhat'].values

    # 合并数据
    full_data = pd.concat([train_data, test_data], ignore_index=True)

    # 返回模型、预测结果和原始数据（包含残差）
    return model, forecast, full_data


# 4. 构建LSTM模型用于预测残差
def create_lstm_dataset(data, lookback=7):
    """
    为LSTM准备时间序列数据（滑动窗口）
    """
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i])
    return np.array(X), np.array(y)


def build_lstm_model(data, residual_col='residual', test_size=0.1, lookback=7):
    """
    构建LSTM模型用于预测残差序列（GPU优化版）
    """
    # 检查GPU是否可用
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("使用GPU加速")
        try:
            # 设置GPU显存按需增长
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("未检测到GPU，使用CPU运行")

    # 确保数据是按时间排序的
    data = data.sort_values('ds')

    # 获取残差序列
    residuals = data[residual_col].values.reshape(-1, 1)

    # 数据归一化
    scaler = MinMaxScaler(feature_range=(-1, 1))
    residuals_scaled = scaler.fit_transform(residuals)

    # 创建LSTM数据集
    X, y = create_lstm_dataset(residuals_scaled, lookback=lookback)

    # 划分训练集和验证集
    train_size = int(len(X) * (1 - test_size))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 构建LSTM模型（GPU会自动加速）
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=(lookback, 1), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    # 使用混合精度训练（需GPU支持）
    policy = tf.keras.mixed_precision.Policy('mixed_float16')
    tf.keras.mixed_precision.set_global_policy(policy)

    model.compile(optimizer='adam', loss='mse')

    # 设置早停机制
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # 训练模型（GPU会自动加速）
    history = model.fit(
        X_train, y_train,
        epochs=100,
        batch_size=64,  # 增大batch_size以充分利用GPU
        validation_data=(X_test, y_test),
        callbacks=[early_stopping],
        verbose=1
    )

    # 预测
    train_predict = model.predict(X_train, batch_size=64)
    test_predict = model.predict(X_test, batch_size=64)

    # 反向转换回原始尺度
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    y_train_inv = scaler.inverse_transform(y_train)
    y_test_inv = scaler.inverse_transform(y_test)

    # 准备结果
    lstm_results = {
        'model': model,
        'scaler': scaler,
        'train_predict': train_predict,
        'test_predict': test_predict,
        'y_train': y_train_inv,
        'y_test': y_test_inv,
        'lookback': lookback,
        'history': history
    }

    return lstm_results


# 5. 混合Prophet-LSTM模型
def hybrid_forecast(prophet_forecast, lstm_results, prophet_data, test_size=0.1, weights=None):
    """
    融合Prophet和LSTM预测结果
    weights: 如果提供，使用固定权重；否则，根据验证集表现确定最优权重
    """
    # 准备Prophet预测和实际值
    train_size = int(len(prophet_data) * (1 - test_size))

    # 分离训练集和测试集
    train_data = prophet_data.iloc[:train_size].copy()
    test_data = prophet_data.iloc[train_size:].copy()

    # 获取LSTM预测的残差
    lstm_train_residuals = lstm_results['train_predict'].flatten()
    lstm_test_residuals = lstm_results['test_predict'].flatten()

    # 对齐索引（由于LSTM需要lookback，会减少部分样本）
    lookback = lstm_results['lookback']

    # Prophet训练集预测
    prophet_train_preds = train_data['yhat'].values[lookback:]

    # Prophet测试集预测
    prophet_test_preds = test_data['yhat'].values

    # 实际值
    train_actual = train_data['y'].values[lookback:]
    test_actual = test_data['y'].values

    # 确定最佳权重
    if weights is None:
        best_weight = optimize_weights(prophet_train_preds, lstm_train_residuals, train_actual)
    else:
        best_weight = weights

    # 融合预测结果
    hybrid_train_preds = prophet_train_preds * best_weight + (prophet_train_preds + lstm_train_residuals) * (
                1 - best_weight)
    hybrid_test_preds = prophet_test_preds * best_weight + (prophet_test_preds + lstm_test_residuals) * (
                1 - best_weight)

    # 计算评估指标
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
        # 处理可能的除零错误
        hybrid_train_mape = np.nan
        hybrid_test_mape = np.nan

    # 准备结果
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
        }
    }

    return results


def optimize_weights(prophet_preds, lstm_residuals, actual, step=0.1):
    """
    优化Prophet和LSTM混合权重
    """
    best_rmse = float('inf')
    best_weight = 0.5  # 默认权重

    # 尝试不同的权重
    for weight in np.arange(0, 1.01, step):
        # 计算混合预测
        hybrid_preds = prophet_preds * weight + (prophet_preds + lstm_residuals) * (1 - weight)

        # 计算RMSE
        rmse = np.sqrt(mean_squared_error(actual, hybrid_preds))

        # 更新最佳权重
        if rmse < best_rmse:
            best_rmse = rmse
            best_weight = weight

    return best_weight


# 6. 可视化函数
def plot_prophet_components(prophet_model, forecast, station_name, output_dir=None):
    """
    绘制Prophet模型的分解组件
    """
    # Matplotlib静态图常规版本
    plt.figure(figsize=(15, 10))
    prophet_model.plot_components(forecast)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{station_name}_prophet_components.png"), dpi=300)
    plt.show()

    # Plotly交互装逼版
    fig = make_subplots(
        rows=4, cols=1,
        subplot_titles=('Trend', 'Week seasonality', 'Annual seasonality', 'Day seasonality'),
        vertical_spacing=0.1,
        shared_xaxes=False
    )

    # 趋势图
    fig.add_trace(
        go.Scatter(x=forecast['ds'], y=forecast['trend'], mode='lines', name='Trend'),
        row=1, col=1
    )

    # 周季节性
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    weekly = forecast[['weekly']].copy()
    weekly['day'] = forecast['ds'].dt.dayofweek
    weekly_avg = weekly.groupby('day')['weekly'].mean().reset_index()
    weekly_avg['day_name'] = weekly_avg['day'].map(lambda x: weekdays[x])

    fig.add_trace(
        go.Bar(x=weekly_avg['day_name'], y=weekly_avg['weekly'], name='Week seasonality', marker_color='indianred'),
        row=2, col=1
    )

    # 年季节性
    if 'yearly' in forecast.columns:
        yearly = forecast[['ds', 'yearly']].copy()
        yearly['month'] = yearly['ds'].dt.month
        yearly_avg = yearly.groupby('month')['yearly'].mean().reset_index()

        fig.add_trace(
            go.Scatter(x=yearly_avg['month'], y=yearly_avg['yearly'], mode='lines+markers', name='Annual Seasonality',
                       line=dict(color='forestgreen')),
            row=3, col=1
        )
        fig.update_xaxes(title_text='Months', tickvals=list(range(1, 13)), row=3, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='markers', name='Annual seasonality is not in use',
                       marker=dict(color='gray')),
            row=3, col=1
        )

    # 日季节性
    if 'daily' in forecast.columns:
        daily = forecast[['ds', 'daily']].copy()
        daily['hour'] = daily['ds'].dt.hour
        daily_avg = daily.groupby('hour')['daily'].mean().reset_index()

        fig.add_trace(
            go.Scatter(x=daily_avg['hour'], y=daily_avg['daily'], mode='lines+markers', name='Day seasonality',
                       line=dict(color='royalblue')),
            row=4, col=1
        )
        fig.update_xaxes(title_text='小时', tickvals=list(range(0, 24)), row=4, col=1)
    else:
        fig.add_trace(
            go.Scatter(x=[0], y=[0], mode='markers', name='Daily seasonality is not in use', marker=dict(color='gray')),
            row=4, col=1
        )

    fig.update_layout(
        height=900,
        title_text=f"{station_name} - Prophet模型分解组件",
        showlegend=False
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f"{station_name}_prophet_components.html"))

    fig.show()


def plot_hybrid_results(hybrid_results, station_name, output_dir=None, title=None):
    """
    绘制混合模型的预测结果对比
    """
    if title is None:
        title = f"{station_name} Prediction results of the prophet-LSTM model"

    train_actual = hybrid_results['train_actual']
    test_actual = hybrid_results['test_actual']
    prophet_train = hybrid_results['prophet_train_preds']
    prophet_test = hybrid_results['prophet_test_preds']
    hybrid_train = hybrid_results['hybrid_train_preds']
    hybrid_test = hybrid_results['hybrid_test_preds']

    # Matplotlib静态图版本
    plt.figure(figsize=(15, 6))

    # 训练集
    train_indices = np.arange(len(train_actual))
    plt.plot(train_indices, train_actual, label='Actual value (training set)', color='blue')
    plt.plot(train_indices, prophet_train, label='Prophet Prediction (Training Set)', color='green', linestyle='--')
    plt.plot(train_indices, hybrid_train, label='Prophrt-LSTM model Prediction (Training set)', color='red',
             linestyle='-.')

    # 测试集
    test_indices = np.arange(len(train_actual), len(train_actual) + len(test_actual))
    plt.plot(test_indices, test_actual, label='Actual value (test set)', color='blue')
    plt.plot(test_indices, prophet_test, label='Prophet Prediction (Test Set)', color='green', linestyle='--')
    plt.plot(test_indices, hybrid_test, label='Prophet-LSTM model prediction(Test set)', color='red', linestyle='-.')

    # 添加分隔线
    plt.axvline(x=len(train_actual), color='gray', linestyle='-')
    plt.text(len(train_actual) + 1, plt.ylim()[1] * 0.9, '测试集开始', fontsize=12)

    plt.title(title)
    plt.xlabel('Time Index')
    plt.ylabel('Ridership')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if output_dir:
        plt.savefig(os.path.join(output_dir, f"{station_name}_hybrid_results.png"), dpi=300)
    plt.show()

    # Plotly交互装逼版
    # 创建日期索引
    train_dates = pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=len(test_actual)),
                                periods=len(train_actual), freq='D')
    test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=len(test_actual), freq='D')

    fig = go.Figure()

    # 添加实际值
    fig.add_trace(go.Scatter(
        x=list(train_dates) + list(test_dates),
        y=list(train_actual) + list(test_actual),
        mode='lines',
        name='Actual Ridership',
        line=dict(color='royalblue', width=2)
    ))

    # 添加Prophet预测值
    fig.add_trace(go.Scatter(
        x=list(train_dates) + list(test_dates),
        y=list(prophet_train) + list(prophet_test),
        mode='lines',
        name='Prophet Prediction',
        line=dict(color='forestgreen', width=1.5, dash='dash')
    ))

    # 添加混合模型预测值
    fig.add_trace(go.Scatter(
        x=list(train_dates) + list(test_dates),
        y=list(hybrid_train) + list(hybrid_test),
        mode='lines',
        name='Prophet-LSTM model Prediction',
        line=dict(color='firebrick', width=1.5)
    ))

    # 添加训练集/测试集分隔线
    fig.add_vline(
        x=train_dates[-1], line_width=1.5, line_dash="dash", line_color="gray",
        annotation_text="测试集开始",
        annotation_position="top right"
    )

    # 更新布局
    fig.update_layout(
        title=title,
        xaxis_title='Date',
        yaxis_title='Ridership',
        legend_title='Type of Data',
        hovermode='x unified',
        template='plotly_white',
        height=600,
        width=1200
    )

    # 添加范围选择器
    fig.update_layout(
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7天", step="day", stepmode="backward"),
                    dict(count=1, label="1月", step="month", stepmode="backward"),
                    dict(count=6, label="6月", step="month", stepmode="backward"),
                    dict(count=1, label="1年", step="year", stepmode="backward"),
                    dict(step="all", label="ALL")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f"{station_name}_hybrid_results.html"))

    fig.show()


def plot_residuals_analysis(hybrid_results, station_name, output_dir=None):
    """
    绘制残差分析图
    """
    # 提取残差
    train_residuals = hybrid_results['train_actual'] - hybrid_results['hybrid_train_preds']
    test_residuals = hybrid_results['test_actual'] - hybrid_results['hybrid_test_preds']
    all_residuals = np.concatenate([train_residuals, test_residuals])

    # Plotly版本
    # 1. 残差分布图
    fig_dist = go.Figure()
    fig_dist.add_trace(go.Histogram(
        x=all_residuals,
        nbinsx=50,
        opacity=0.7,
        marker_color='royalblue',
        name='Residual Distribution'
    ))

    # 添加正态分布拟合线
    x_range = np.linspace(min(all_residuals), max(all_residuals), 100)
    mean = np.mean(all_residuals)
    std = np.std(all_residuals)
    y_norm = np.exp(-(x_range - mean) ** 2 / (2 * std ** 2)) / (std * np.sqrt(2 * np.pi))
    y_norm = y_norm * len(all_residuals) * (max(all_residuals) - min(all_residuals)) / 50  # 缩放到直方图高度

    fig_dist.add_trace(go.Scatter(
        x=x_range,
        y=y_norm,
        mode='lines',
        name='Normal Distribution Fitting',
        line=dict(color='firebrick', width=2)
    ))

    fig_dist.update_layout(
        title=f'{station_name} - 残差分布分析',
        xaxis_title='Residual',
        yaxis_title='Frequency',
        template='plotly_white',
        height=500,
        width=900
    )

    if output_dir:
        fig_dist.write_html(os.path.join(output_dir, f"{station_name}_residual_distribution.html"))

    fig_dist.show()

    # 2. 残差时序图
    # 创建日期索引
    train_dates = pd.date_range(end=pd.Timestamp.today() - pd.Timedelta(days=len(test_residuals)),
                                periods=len(train_residuals), freq='D')
    test_dates = pd.date_range(start=train_dates[-1] + pd.Timedelta(days=1), periods=len(test_residuals), freq='D')

    fig_ts = go.Figure()

    # 添加训练集残差
    fig_ts.add_trace(go.Scatter(
        x=train_dates,
        y=train_residuals,
        mode='lines',
        name='Residual of the training set',
        line=dict(color='royalblue', width=1)
    ))

    # 添加测试集残差
    fig_ts.add_trace(go.Scatter(
        x=test_dates,
        y=test_residuals,
        mode='lines',
        name='Residual of the test set',
        line=dict(color='firebrick', width=1)
    ))

    # 添加零线
    fig_ts.add_hline(
        y=0, line_width=1, line_dash="solid", line_color="black"
    )

    # 训练集/测试集分隔线
    fig_ts.add_vline(
        x=train_dates[-1], line_width=1, line_dash="dash", line_color="gray",
        annotation_text="测试集开始",
        annotation_position="top right"
    )

    fig_ts.update_layout(
        title=f'{station_name} - 残差时序分析',
        xaxis_title='Date',
        yaxis_title='Residual',
        legend_title='Type of Data',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        width=900,
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="7d", step="day", stepmode="backward"),
                    dict(count=1, label="1month", step="month", stepmode="backward"),
                    dict(count=3, label="3months", step="month", stepmode="backward"),
                    dict(step="all", label="All")
                ])
            ),
            rangeslider=dict(visible=True),
            type="date"
        )
    )

    if output_dir:
        fig_ts.write_html(os.path.join(output_dir, f"{station_name}_residual_timeseries.html"))

    fig_ts.show()


def plot_station_traffic_heatmap(station_data, station_name, output_dir=None):
    """
    绘制车站客流量热力图（按小时和星期几）
    """
    # 确保有日期索引和必要的列
    if not isinstance(station_data.index, pd.DatetimeIndex):
        print("数据索引不是日期时间格式，无法创建热力图。")
        return

    # 准备热力图数据
    df_heatmap = station_data.copy()
    df_heatmap['hour'] = df_heatmap.index.hour
    df_heatmap['dayofweek'] = df_heatmap.index.dayofweek

    # 按小时和星期几聚合
    pivot_table = df_heatmap.pivot_table(
        values='TOTAL_FLOW',
        index='hour',
        columns='dayofweek',
        aggfunc='mean'
    )

    # 重命名列以显示星期几
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thusrday', 'Friday', 'Saturday', 'Sunday']
    pivot_table.columns = days

    # Plotly热力图
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='Viridis',
        hoverongaps=False,
        colorbar=dict(title='Average Ridership')
    ))

    fig.update_layout(
        title=f'{station_name} - 客流量热力图 (按小时和星期)',
        xaxis_title='Week',
        yaxis_title='Hour',
        height=600,
        width=900,
        template='plotly_white'
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f"{station_name}_traffic_heatmap.html"))

    fig.show()


def plot_monthly_trend(station_data, station_name, output_dir=None):
    """
    绘制月度客流量趋势图
    """
    # 按月聚合
    monthly_data = station_data['TOTAL_FLOW'].resample('M').sum()

    # Plotly图表
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=monthly_data.index,
        y=monthly_data.values,
        mode='lines+markers',
        name='月度总客流量',
        line=dict(color='royalblue', width=2),
        marker=dict(size=8)
    ))

    # 添加趋势线（简单移动平均）
    rolling_avg = monthly_data.rolling(window=3).mean()
    fig.add_trace(go.Scatter(
        x=rolling_avg.index,
        y=rolling_avg.values,
        mode='lines',
        name='3 Months MA',
        line=dict(color='firebrick', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f'{station_name} - 月度客流量趋势',
        xaxis_title='MOnth',
        yaxis_title='Total Ridership',
        legend_title='指标',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        width=900
    )

    if output_dir:
        fig.write_html(os.path.join(output_dir, f"{station_name}_monthly_trend.html"))

    fig.show()


def print_metrics(hybrid_results):
    """
    打印评估指标
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


# 7. 主函数
def main():
    """
    主函数，运行全部流程
    """
    # 1. 加载数据
    data_files = glob.glob("/content/drive/MyDrive/turnstile/turnstile-usage-data-*.csv")

    if not data_files:
        print("未找到数据文件，请确认数据文件位于桌面上。")
        return

    print(f"找到以下数据文件: {[os.path.basename(f) for f in data_files]}")

    # 加载数据
    print("正在加载数据...")
    df = load_turnstile_data(data_files)
    print(f"加载完成，共 {len(df)} 条记录。")

    # 2. 数据预处理
    print("正在预处理数据...")
    df_processed = preprocess_data(df)
    print(f"预处理完成，剩余 {len(df_processed)} 条有效记录。")

    # 3. 聚合数据（按天）
    print("正在聚合数据...")
    df_daily = aggregate_by_station(df_processed, time_interval='D')

    # 4. 选择一个繁忙的车站进行建模
    # 计算每个车站的总流量
    station_totals = df_daily.groupby('STATION')['TOTAL_FLOW'].sum().sort_values(ascending=False)
    top_stations = station_totals.head(10)

    print("\n流量最大的10个车站:")
    for i, (station, total) in enumerate(top_stations.items(), 1):
        print(f"{i}. {station}: {total:,}")

    # 选择流量最大的车站
    target_station = top_stations.index[0]
    print(f"\n选择 {target_station} 用于建模...")

    # 5. 准备特定车站数据
    station_data = prepare_station_data(df_daily, target_station)
    print(f"准备了 {len(station_data)} 天的数据用于模型训练和测试。")

    # 查看基本统计
    print("\n数据统计摘要:")
    print(station_data['TOTAL_FLOW'].describe())

    # 6. 构建Prophet模型
    print("\n正在构建Prophet模型...")
    prophet_model, prophet_forecast, prophet_data = build_prophet_model(station_data, test_size=0.2)

    # 7. 构建LSTM残差模型
    print("正在构建LSTM残差模型...")
    lstm_results = build_lstm_model(prophet_data, test_size=0.2, lookback=7)

    # 8. 混合模型预测
    print("正在融合模型结果...")
    hybrid_results = hybrid_forecast(prophet_forecast, lstm_results, prophet_data, test_size=0.2)

    # 9. 显示结果
    print_metrics(hybrid_results)

    # 10. 可视化
    print("\n生成可视化结果...")
    plot_prophet_components(prophet_model, prophet_forecast)
    plot_hybrid_results(hybrid_results, title=f"{target_station} 车站客流量预测")

    print("\n建模完成！")

    return prophet_model, prophet_forecast, lstm_results, hybrid_results


if __name__ == "__main__":
    main()