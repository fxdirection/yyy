import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from datetime import datetime, timedelta
import os
import glob
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import plotly.offline as py
import plotly.graph_objs as go
import holidays
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
sns.set(style="whitegrid")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("开始执行地铁客流量分析与预测...")


def load_turnstile_data(years=None):
    """
    加载多年的闸机数据

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
    """数据预处理"""
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

    # 处理异常值
    df['ENTRIES_DIFF'] = df['ENTRIES_DIFF'].clip(lower=0)
    df['EXITS_DIFF'] = df['EXITS_DIFF'].clip(lower=0)

    max_reasonable_count = 10000
    df['ENTRIES_DIFF'] = df['ENTRIES_DIFF'].clip(upper=max_reasonable_count)
    df['EXITS_DIFF'] = df['EXITS_DIFF'].clip(upper=max_reasonable_count)

    # 添加时间特征
    df['DOW'] = df['Datetime'].dt.dayofweek  # 星期几 (0=周一, 6=周日)
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


def analyze_station_traffic(df_agg):
    """分析各站点客流量"""
    print("分析站点客流量...")

    station_traffic = df_agg.groupby('Station').agg({
        'TOTAL_TRAFFIC': 'sum',
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum'
    }).reset_index()

    station_traffic = station_traffic.sort_values('TOTAL_TRAFFIC', ascending=False)

    print("客流量最高的10个站点:")
    print(station_traffic.head(10))

    # 普劳特前10个最繁忙的站点
    plt.figure(figsize=(12, 8))
    top_stations = station_traffic.head(10)

    # 堆叠条形图
    plt.barh(top_stations['Station'], top_stations['ENTRIES_DIFF'], color='skyblue', label='Entries')
    plt.barh(top_stations['Station'], top_stations['EXITS_DIFF'], left=top_stations['ENTRIES_DIFF'],
             color='lightcoral', label='Exits')

    plt.xlabel('Ridership')
    plt.ylabel('Station')
    plt.title('The 10 busiest stations of the New York subway')
    plt.legend()
    plt.tight_layout()
    plt.savefig('top_stations_traffic.png', dpi=300)
    plt.close()

    return station_traffic


def prepare_time_series(df_agg, top_n_stations=None, target_station=None):
    """
    准备时间序列数据用于Prophet模型

    参数:
    df_agg (DataFrame): 聚合后的数据框
    top_n_stations (int): 选取客流量最高的前N个站点，默认为None
    target_station (str): 指定分析的目标站点名称，默认为None

    返回:
    DataFrame: 符合Prophet要求格式的数据框
    """
    print("准备时间序列数据...")

    # 如果指定了目标站点
    if target_station:
        df_station = df_agg[df_agg['Station'] == target_station].copy()
        if df_station.empty:
            print(f"警告: 未找到站点 '{target_station}'，将使用总体数据")
            df_station = df_agg.copy()

    # 如果已经指定要选取top N站点了
    elif top_n_stations:
        # 找出客流量最高的N个站点
        top_stations = analyze_station_traffic(df_agg)['Station'].head(top_n_stations).tolist()
        df_station = df_agg[df_agg['Station'].isin(top_stations)].copy()

    # 否则的话使用所有站点的总和
    else:
        df_station = df_agg.copy()

    # 然后按日期聚合所有选中站点的客流量
    ts_data = df_station.groupby('Date').agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum',
        'TOTAL_TRAFFIC': 'sum'
    }).reset_index()

    # 这是Prophet要求的格式: ds (日期) 和 y (预测目标) 好像要求时间戳格式🤔
    prophet_df = ts_data.rename(columns={'Date': 'ds', 'TOTAL_TRAFFIC': 'y'})

    print(f"时间序列数据范围: {prophet_df['ds'].min()} 到 {prophet_df['ds'].max()}")
    return prophet_df


def add_holidays(df):
    """添加美国假日信息到数据框"""
    print("添加假日信息...")

    df['ds'] = pd.to_datetime(df['ds'])

    start_year = df['ds'].min().year
    end_year = df['ds'].max().year

    # 获取大漂亮的假日
    us_holidays = holidays.US(years=range(start_year, end_year + 1))

    # 先构造一个假日数据框 集中处理
    holiday_df = pd.DataFrame(
        [(pd.Timestamp(date), name) for date, name in us_holidays.items()],
        columns=['ds', 'holiday']
    )

    # 添加假日前后的影响天数
    holiday_df['lower_window'] = -1  # 假日前一天
    holiday_df['upper_window'] = 1  # 假日后一天

    # 重要假日影响应该会大一点 就多一天吧 重要假日是我猜的（like the fourth of July~
    important_holidays = ['New Year', 'Independence Day', 'Thanksgiving', 'Christmas Day']
    for holiday in important_holidays:
        mask = holiday_df['holiday'].str.contains(holiday, case=False, na=False)
        holiday_df.loc[mask, 'lower_window'] = -2  # 重要假日前两天
        holiday_df.loc[mask, 'upper_window'] = 2  # 重要假日后两天

    print("假日数据预览:")
    print(holiday_df.head())
    print("假日数据列名:", holiday_df.columns.tolist())

    return holiday_df


def train_prophet_model(df, forecast_periods=60, holidays_df=None):
    """
    训练Prophet模型并进行预测

    参数:
    df (DataFrame): 符合Prophet格式的数据
    forecast_periods (int): 预测的天数
    holidays_df (DataFrame): 假日数据框

    返回:
    tuple: (Prophet模型, 预测结果)
    """
    print("训练Prophet模型...")

    # Prophet建模
    model = Prophet(
        changepoint_prior_scale=0.05,  # 控制趋势灵活性
        seasonality_prior_scale=10,  # 增强季节性
        seasonality_mode='multiplicative',  # 乘法季节性通常更适合客流量
        daily_seasonality=True,  # 启用日内季节性
        weekly_seasonality=True,  # 启用周季节性
        yearly_seasonality=True  # 启用年季节性
    )

    # 添加月季节性
    model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

    # 添加季度季节性
    model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

    # 添加假日效应
    if holidays_df is not None:
        model.add_country_holidays(country_name='US')
        model.holidays = holidays_df

    model.fit(df)

    future = model.make_future_dataframe(periods=forecast_periods)

    forecast = model.predict(future)

    print(f"模型训练完成，预测未来{forecast_periods}天")
    print("预测结果列:", forecast.columns.tolist())
    return model, forecast


def visualize_forecast(model, forecast, df, title='地铁客流量预测'):
    """可视化预测结果"""
    print("可视化预测结果...")

    # 普劳特预测总图 似乎普劳特不出汉语 但我已经设置过了 仍是一个bug
    fig1 = model.plot(forecast)
    plt.title(f'{title} - Tendency Chart')
    plt.xlabel('Date')
    plt.ylabel('Ridership')
    plt.tight_layout()
    plt.savefig('prophet_forecast.png', dpi=300)
    plt.close()

    # 普劳特组件图
    fig2 = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig('prophet_components.png', dpi=300)
    plt.close()

    # 最近一年的实际值与预测值对比图吧
    plt.figure(figsize=(12, 6))

    # 先筛选出最近一年的数据
    last_date = df['ds'].max()
    one_year_ago = last_date - pd.Timedelta(days=365)

    # 然后提取最近一年的实际值和预测值
    recent_actual = df[df['ds'] >= one_year_ago]
    recent_forecast = forecast[(forecast['ds'] >= one_year_ago) & (forecast['ds'] <= last_date)]

    # 普劳特出实际值
    plt.plot(recent_actual['ds'], recent_actual['y'], 'k.', label='Actual Ridership')

    # 普劳特预测值及其置信区间
    plt.plot(recent_forecast['ds'], recent_forecast['yhat'], 'b-', label='Predict Ridership')
    plt.fill_between(recent_forecast['ds'], recent_forecast['yhat_lower'], recent_forecast['yhat_upper'],
                     color='blue', alpha=0.2, label='95% Confidence Interval')

    plt.title(f'{title} - Trend over the Past Year')
    plt.xlabel('Date')
    plt.ylabel('Ridership')
    plt.legend()
    plt.tight_layout()
    plt.savefig('recent_year_comparison.png', dpi=300)
    plt.close()

    # 普劳特未来预测
    plt.figure(figsize=(12, 6))

    # 提取未来的预测值
    future_forecast = forecast[forecast['ds'] > last_date]

    # 普劳特预测值及其置信区间
    plt.plot(future_forecast['ds'], future_forecast['yhat'], 'r-', label='Future Ridership Forecast')
    plt.fill_between(future_forecast['ds'], future_forecast['yhat_lower'], future_forecast['yhat_upper'],
                     color='red', alpha=0.2, label='95% Confidence Interval')

    plt.title(f'{title} - Future Ridership Forecast')
    plt.xlabel('Date')
    plt.ylabel('Ridership')
    plt.legend()
    plt.tight_layout()
    plt.savefig('future_forecast.png', dpi=300)
    plt.close()

    # 输出为交互式Plotly图表(保存为HTML)
    try:
        fig = plot_plotly(model, forecast)
        fig.update_layout(title=f'{title} - 交互式预测图')
        py.plot(fig, filename='interactive_forecast.html', auto_open=False)

        components_fig = plot_components_plotly(model, forecast)
        components_fig.update_layout(title=f'{title} - 交互式组件分解图')
        py.plot(components_fig, filename='interactive_components.html', auto_open=False)

        print("已生成交互式HTML可视化")
    except Exception as e:
        print(f"生成交互式图表时出错: {e}")


def analyze_holiday_effects(forecast):
    print("分析假日效应...")

    # 查找包含假日效应的列（通常以假日名称或 'holidays' 开头）
    holiday_cols = [col for col in forecast.columns if 'holidays' in col.lower() or any(
        h in col.lower() for h in ['new year', 'independence', 'thanksgiving', 'christmas'])]

    if not holiday_cols:
        print("未找到假日效应列，可能是假日未正确配置")
        return None

    print("找到的假日效应列:", holiday_cols)

    # 假设假日效应汇总在 'holidays' 列或单独的假日列
    holiday_effects = forecast[holiday_cols + ['ds']].copy()
    holiday_effects = holiday_effects[holiday_effects[holiday_cols].notna().any(axis=1)]

    # 如果有多个假日列，汇总总效应
    if len(holiday_cols) > 1:
        holiday_effects['holiday_effect'] = holiday_effects[holiday_cols].sum(axis=1)
    else:
        holiday_effects['holiday_effect'] = holiday_effects[holiday_cols[0]]

    # 按效应绝对值排序
    holiday_effects['abs_effect'] = holiday_effects['holiday_effect'].abs()
    holiday_effects = holiday_effects.sort_values('abs_effect', ascending=False)

    print("假日效应最显著的10个日期:")
    print(holiday_effects[['ds', 'holiday_effect', 'abs_effect']].head(10))

    # 可视化
    plt.figure(figsize=(12, 8))
    top_holidays = holiday_effects.head(15)
    plt.barh(top_holidays['ds'].dt.strftime('%Y-%m-%d'),
             top_holidays['holiday_effect'], color='skyblue')

    plt.xlabel('Ridership Changes')
    plt.ylabel('Date')
    plt.title('The Impact of Holidays on Subway Ridership')
    plt.tight_layout()
    plt.savefig('holiday_effects.png', dpi=300)
    plt.close()

    return holiday_effects


def evaluate_model(df, forecast):
    """评估模型性能"""
    print("评估模型性能...")

    # 将预测结果与实际值合并
    evaluation = pd.merge(
        df[['ds', 'y']],
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        on='ds',
        how='left'
    )

    # 计算评估指标
    evaluation['error'] = evaluation['y'] - evaluation['yhat']
    evaluation['abs_error'] = np.abs(evaluation['error'])
    evaluation['squared_error'] = evaluation['error'] ** 2

    # 计算MAE, RMSE, MAPE
    mae = evaluation['abs_error'].mean()
    rmse = np.sqrt(evaluation['squared_error'].mean())
    # 避免除以零
    evaluation['abs_pct_error'] = evaluation['abs_error'] / evaluation['y'].replace(0, np.nan) * 100
    mape = evaluation['abs_pct_error'].mean()

    print(f"平均绝对误差 (MAE): {mae:.2f}")
    print(f"均方根误差 (RMSE): {rmse:.2f}")
    print(f"平均绝对百分比误差 (MAPE): {mape:.2f}%")

    # 检查预测区间覆盖率
    evaluation['in_range'] = (evaluation['y'] >= evaluation['yhat_lower']) & (
                evaluation['y'] <= evaluation['yhat_upper'])
    coverage = evaluation['in_range'].mean() * 100
    print(f"95%置信区间覆盖率: {coverage:.2f}%")

    # 绘制误差直方图
    plt.figure(figsize=(10, 6))
    plt.hist(evaluation['error'], bins=50, color='skyblue', edgecolor='black')
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel('Forecast Error')
    plt.ylabel('Frequency')
    plt.title('Prediction Error Distribution')
    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300)
    plt.close()

    # 绘制真实值与预测值的散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(evaluation['y'], evaluation['yhat'], alpha=0.5)

    # 添加对角线

    max_val = max(evaluation['y'].max(), evaluation['yhat'].max())
    min_val = min(evaluation['y'].min(), evaluation['yhat'].min())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')

    plt.xlabel('Actual Ridership')
    plt.ylabel('Predict Ridership')
    plt.title('Actual vs Predicted')
    plt.tight_layout()
    plt.savefig('actual_vs_predicted.png', dpi=300)
    plt.close()

    return {
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
        'Coverage': coverage
    }


def plot_station_comparison(df_agg, target_stations, forecast_periods=90):
    """
    绘制特定站点的进出口客流量真实值与预测值对比图

    参数:
    df_agg (DataFrame): 聚合后的数据框
    target_stations (list): 需要分析的站点列表
    forecast_periods (int): 预测的天数
    """
    print(f"开始绘制{target_stations}站点的对比图...")

    # 为每个目标站点创建图表
    for station in target_stations + ['ALL STATIONS']:
        plt.figure(figsize=(14, 8))

        # 准备数据
        if station == 'ALL STATIONS':
            station_df = df_agg.groupby('Date').agg({
                'ENTRIES_DIFF': 'sum',
                'EXITS_DIFF': 'sum'
            }).reset_index()
            title = 'All Stations Total Traffic'
        else:
            station_df = df_agg[df_agg['Station'] == station].copy()
            title = f'{station} Station Traffic'

        # 准备时间序列数据
        entries_df = station_df[['Date', 'ENTRIES_DIFF']].rename(columns={'Date': 'ds', 'ENTRIES_DIFF': 'y'})
        exits_df = station_df[['Date', 'EXITS_DIFF']].rename(columns={'Date': 'ds', 'EXITS_DIFF': 'y'})

        # 添加假日信息
        holidays_df = add_holidays(entries_df)

        # 训练模型并预测
        entries_model, entries_forecast = train_prophet_model(entries_df, forecast_periods, holidays_df)
        exits_model, exits_forecast = train_prophet_model(exits_df, forecast_periods, holidays_df)

        # 合并实际值和预测值
        comparison_df = pd.merge(
            station_df[['Date', 'ENTRIES_DIFF', 'EXITS_DIFF']],
            entries_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'yhat': 'entries_pred',
                'yhat_lower': 'entries_lower',
                'yhat_upper': 'entries_upper'
            }),
            left_on='Date', right_on='ds', how='left'
        )

        comparison_df = pd.merge(
            comparison_df,
            exits_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
                'yhat': 'exits_pred',
                'yhat_lower': 'exits_lower',
                'yhat_upper': 'exits_upper'
            }),
            left_on='Date', right_on='ds', how='left'
        )

        # 筛选最近一年的数据用于可视化
        last_date = comparison_df['Date'].max()
        one_year_ago = last_date - pd.Timedelta(days=365)
        recent_data = comparison_df[comparison_df['Date'] >= one_year_ago]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

        # 绘制进站客流量
        ax1.plot(recent_data['Date'], recent_data['ENTRIES_DIFF'], 'k-', label='Actual Entries', alpha=0.7)
        ax1.plot(recent_data['Date'], recent_data['entries_pred'], 'b-', label='Predicted Entries')
        ax1.fill_between(recent_data['Date'], recent_data['entries_lower'], recent_data['entries_upper'],
                         color='blue', alpha=0.2, label='95% CI')
        ax1.set_title(f'{title} - Entries Comparison')
        ax1.set_ylabel('Entries Count')
        ax1.legend()

        # 绘制出站客流量
        ax2.plot(recent_data['Date'], recent_data['EXITS_DIFF'], 'k-', label='Actual Exits', alpha=0.7)
        ax2.plot(recent_data['Date'], recent_data['exits_pred'], 'r-', label='Predicted Exits')
        ax2.fill_between(recent_data['Date'], recent_data['exits_lower'], recent_data['exits_upper'],
                         color='red', alpha=0.2, label='95% CI')
        ax2.set_title(f'{title} - Exits Comparison')
        ax2.set_ylabel('Exits Count')
        ax2.legend()

        # 格式化x轴
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_locator(mdates.MonthLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()

        # 保存图表
        filename = f"{station.replace(' ', '_').replace('-', '_')}_comparison.png"
        plt.savefig(filename, dpi=300)
        plt.close()

        print(f"已保存 {station} 站点的对比图: {filename}")


def create_interactive_comparison(df_agg, target_stations):
    """
    创建交互式站点客流量对比图

    参数:
    df_agg (DataFrame): 聚合后的数据框
    target_stations (list): 需要分析的站点列表
    """
    print("创建交互式站点对比图...")

    # 准备数据 - 添加所有站点总和
    all_stations = df_agg.groupby('Date').agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum'
    }).reset_index()
    all_stations['Station'] = 'ALL STATIONS'

    # 合并所有目标站点数据
    selected_stations = df_agg[df_agg['Station'].isin(target_stations)].copy()
    combined_df = pd.concat([selected_stations, all_stations], ignore_index=True)

    # 创建Plotly图表
    fig = go.Figure()

    # 添加下拉菜单选项
    buttons = []
    visible = [False] * (len(target_stations) + 1) * 2  # 每个站点有entries和exits两条线

    # 为每个站点添加数据
    for i, station in enumerate(target_stations + ['ALL STATIONS']):
        station_data = combined_df[combined_df['Station'] == station]

        # 添加入站数据
        fig.add_trace(go.Scatter(
            x=station_data['Date'],
            y=station_data['ENTRIES_DIFF'],
            name=f'{station} - Entries',
            visible=(i == 0),  # 默认显示第一个站点
            line=dict(color='blue'),
            hovertemplate='Date: %{x}<br>Entries: %{y:,}<extra></extra>'
        ))

        # 添加出站数据
        fig.add_trace(go.Scatter(
            x=station_data['Date'],
            y=station_data['EXITS_DIFF'],
            name=f'{station} - Exits',
            visible=(i == 0),  # 默认显示第一个站点
            line=dict(color='red'),
            hovertemplate='Date: %{x}<br>Exits: %{y:,}<extra></extra>'
        ))

        # 创建按钮选项
        buttons.append(dict(
            label=station,
            method='update',
            args=[{'visible': [v == i * 2 or v == i * 2 + 1 for v in range(len(visible))]},
                  {'title': f'{station} Station Traffic'}]
        ))

    # 更新布局
    fig.update_layout(
        title=f'{target_stations[0]} Station Traffic',
        xaxis_title='Date',
        yaxis_title='Passenger Count',
        hovermode='x unified',
        updatemenus=[{
            'buttons': buttons,
            'direction': 'down',
            'showactive': True,
            'x': 0.1,
            'y': 1.15
        }],
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    # 保存为HTML文件
    py.plot(fig, filename='interactive_station_comparison.html', auto_open=False)
    print("已生成交互式站点对比图: interactive_station_comparison.html")


def create_interactive_forecast_comparison(df_agg, target_stations, forecast_periods=90):
    """
    创建交互式预测对比图，包含真实值和预测值

    参数:
    df_agg (DataFrame): 聚合后的数据框
    target_stations (list): 需要分析的站点列表
    forecast_periods (int): 预测的天数
    """
    print("创建交互式预测对比图...")

    # 准备数据 - 添加所有站点总和
    all_stations = df_agg.groupby('Date').agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum'
    }).reset_index()
    all_stations['Station'] = 'ALL STATIONS'

    # 合并所有目标站点数据
    selected_stations = df_agg[df_agg['Station'].isin(target_stations)].copy()
    combined_df = pd.concat([selected_stations, all_stations], ignore_index=True)

    # 存储所有预测结果
    all_forecasts = {}

    # 为每个站点训练模型
    for station in target_stations + ['ALL STATIONS']:
        station_data = combined_df[combined_df['Station'] == station].copy()

        # 准备时间序列数据（确保没有缺失值）
        entries_df = station_data[['Date', 'ENTRIES_DIFF']].rename(columns={'Date': 'ds', 'ENTRIES_DIFF': 'y'}).dropna()
        exits_df = station_data[['Date', 'EXITS_DIFF']].rename(columns={'Date': 'ds', 'EXITS_DIFF': 'y'}).dropna()

        # 添加假日信息
        holidays_df = add_holidays(entries_df)

        try:
            # 训练模型并预测入站量
            entries_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                seasonality_mode='multiplicative'
            ).fit(entries_df)
            entries_future = entries_model.make_future_dataframe(periods=forecast_periods)
            entries_forecast = entries_model.predict(entries_future)

            # 训练模型并预测出站量
            exits_model = Prophet(
                changepoint_prior_scale=0.05,
                seasonality_prior_scale=10,
                holidays_prior_scale=10,
                seasonality_mode='multiplicative'
            ).fit(exits_df)
            exits_future = exits_model.make_future_dataframe(periods=forecast_periods)
            exits_forecast = exits_model.predict(exits_future)

            # 合并预测结果
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
                how='outer'  # 使用外连接确保不丢失任何数据
            )

            all_forecasts[station] = {
                'data': station_data,
                'forecast': forecast_df
            }

            print(
                f"{station} 站点预测完成 - 入站量预测数: {len(entries_forecast)}, 出站量预测数: {len(exits_forecast)}")

        except Exception as e:
            print(f"训练 {station} 站点模型时出错: {str(e)}")
            continue

    # 创建Plotly图表
    fig = go.Figure()

    # 添加下拉菜单选项
    buttons = []

    # 为每个站点添加数据
    for i, station in enumerate(target_stations + ['ALL STATIONS']):
        if station not in all_forecasts:
            continue

        data = all_forecasts[station]['data']
        forecast = all_forecasts[station]['forecast']

        # 确保日期列格式一致
        data['Date'] = pd.to_datetime(data['Date'])
        forecast['ds'] = pd.to_datetime(forecast['ds'])

        # 合并实际值和预测值（使用外连接）
        merged = pd.merge(
            data[['Date', 'ENTRIES_DIFF', 'EXITS_DIFF']],
            forecast,
            left_on='Date',
            right_on='ds',
            how='outer'
        ).sort_values('Date')

        # 调试输出（仅检查问题站点）
        if station == '34 ST-PENN STA':
            print(f"\n调试信息 - {station}:")
            print("实际出站量非空数:", merged['EXITS_DIFF'].notnull().sum())
            print("预测出站量非空数:", merged['exits_pred'].notnull().sum())
            print("预测数据样例:")
            print(merged[['Date', 'EXITS_DIFF', 'exits_pred']].tail(10))

        # 设置当前跟踪是否可见（默认显示第一个站点）
        is_visible = (i == 0)

        # 添加入站实际数据
        fig.add_trace(go.Scatter(
            x=merged['Date'],
            y=merged['ENTRIES_DIFF'],
            name=f'{station} - 实际入站',
            visible=is_visible,
            line=dict(color='blue', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>实际入站: %{y:,}<extra></extra>'
        ))

        # 添加入站预测数据
        fig.add_trace(go.Scatter(
            x=merged['ds'],
            y=merged['entries_pred'],
            name=f'{station} - 预测入站',
            visible=is_visible,
            line=dict(color='blue', dash='dot', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>预测入站: %{y:,}<extra></extra>',
            connectgaps=True
        ))

        # 添加入站置信区间
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

        # 添加出站实际数据
        fig.add_trace(go.Scatter(
            x=merged['Date'],
            y=merged['EXITS_DIFF'],
            name=f'{station} - 实际出站',
            visible=is_visible,
            line=dict(color='red', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>实际出站: %{y:,}<extra></extra>'
        ))

        # 添加出站预测数据
        fig.add_trace(go.Scatter(
            x=merged['ds'],
            y=merged['exits_pred'],
            name=f'{station} - 预测出站',
            visible=is_visible,
            line=dict(color='red', dash='dot', width=2),
            hovertemplate='日期: %{x|%Y-%m-%d}<br>预测出站: %{y:,}<extra></extra>',
            connectgaps=True
        ))

        # 添加出站置信区间
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

        # 创建按钮选项
        buttons.append(dict(
            label=station,
            method='update',
            args=[{
                'visible': [trace.visible if j // 6 != i else True for j, trace in enumerate(fig.data)]
            }, {
                'title': f'{station} 站流量 - 实际 vs 预测'
            }]
        ))

    # 更新布局
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

    # 保存为HTML文件
    py.plot(fig, filename='interactive_forecast_comparison.html', auto_open=False)
    print("已生成交互式预测对比图: interactive_forecast_comparison.html")

    return fig


def prepare_selected_stations_data(df_agg, selected_stations):
    """
    为选定的站点准备时间序列数据用于Prophet模型

    参数:
    df_agg (DataFrame): 聚合后的数据框
    selected_stations (list): 选定分析的站点名称列表

    返回:
    dict: 包含每个站点时间序列数据的字典，以及一个所有站点的总和
    """
    print(f"为选定的{len(selected_stations)}个站点准备时间序列数据...")

    stations_data = {}

    # 为每个选定的站点准备数据
    for station in selected_stations:
        df_station = df_agg[df_agg['Station'] == station].copy()
        if df_station.empty:
            print(f"警告: 未找到站点 '{station}'")
            continue

        # 按日期聚合此站点的客流量
        ts_data = df_station.groupby('Date').agg({
            'ENTRIES_DIFF': 'sum',
            'EXITS_DIFF': 'sum',
            'TOTAL_TRAFFIC': 'sum'
        }).reset_index()

        # 按Prophet要求的格式准备数据
        prophet_df = ts_data.rename(columns={'Date': 'ds', 'TOTAL_TRAFFIC': 'y'})
        stations_data[station] = prophet_df
        print(f"已准备站点 '{station}' 的数据，共{len(prophet_df)}条记录")

    # 所有站点的总和
    all_stations_ts = df_agg.groupby('Date').agg({
        'ENTRIES_DIFF': 'sum',
        'EXITS_DIFF': 'sum',
        'TOTAL_TRAFFIC': 'sum'
    }).reset_index()

    all_stations_prophet = all_stations_ts.rename(columns={'Date': 'ds', 'TOTAL_TRAFFIC': 'y'})
    stations_data['ALL_STATIONS'] = all_stations_prophet
    print(f"已准备所有站点的总和数据，共{len(all_stations_prophet)}条记录")

    return stations_data


def train_models_for_stations(stations_data, forecast_periods=60, holidays_df=None):
    """
    为多个站点训练Prophet模型并进行预测

    参数:
    stations_data (dict): 包含每个站点时间序列数据的字典
    forecast_periods (int): 预测的天数
    holidays_df (DataFrame): 假日数据框

    返回:
    dict: 包含每个站点模型和预测结果的字典
    """
    print("为多个站点训练Prophet模型...")

    results = {}

    for station_name, df in stations_data.items():
        print(f"训练站点 '{station_name}' 的模型...")

        # Prophet建模
        model = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10,
            seasonality_mode='multiplicative',
            daily_seasonality=True,
            weekly_seasonality=True,
            yearly_seasonality=True
        )

        # 添加月季节性
        model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

        # 添加季度季节性
        model.add_seasonality(name='quarterly', period=91.25, fourier_order=5)

        # 添加假日效应
        if holidays_df is not None:
            model.add_country_holidays(country_name='US')
            model.holidays = holidays_df

        model.fit(df)

        future = model.make_future_dataframe(periods=forecast_periods)
        forecast = model.predict(future)

        results[station_name] = {
            'model': model,
            'forecast': forecast,
            'data': df
        }

        print(f"站点 '{station_name}' 的模型训练完成，预测未来{forecast_periods}天")

    return results


def create_interactive_station_comparison(stations_results):
    """
    创建多站点客流量对比的交互式可视化

    参数:
    stations_results (dict): 包含每个站点模型和预测结果的字典

    返回:
    None: 函数将生成HTML文件
    """
    print("创建站点客流量对比的交互式可视化...")

    # 创建站点选择的下拉菜单数据
    station_options = list(stations_results.keys())

    # 创建交互式图表
    fig = go.Figure()

    # 初始显示的站点
    initial_station = 'ALL_STATIONS'

    # 为每个站点添加实际值和预测值曲线（默认隐藏）
    for station_name, result in stations_results.items():
        df = result['data']
        forecast = result['forecast']

        # 合并实际值和预测值用于图表
        visible = (station_name == initial_station)

        # 添加实际值曲线
        fig.add_trace(
            go.Scatter(
                x=df['ds'],
                y=df['y'],
                mode='markers',
                name=f'{station_name} - 实际值',
                marker=dict(color='black', size=4),
                visible=visible
            )
        )

        # 添加预测值曲线
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines',
                name=f'{station_name} - 预测值',
                line=dict(color='blue', width=2),
                visible=visible
            )
        )

        # 添加预测置信区间
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(255, 255, 255, 0)'),
                name=f'{station_name} - 95%置信区间',
                visible=visible
            )
        )

    # 创建下拉菜单
    dropdown_buttons = []
    for station in station_options:
        station_index = station_options.index(station)
        station_traces = [False] * len(fig.data)

        # 为当前站点设置可见性
        start_index = station_index * 3
        for i in range(3):  # 每个站点有3个曲线：实际值、预测值和置信区间
            if start_index + i < len(station_traces):
                station_traces[start_index + i] = True

        dropdown_buttons.append(
            dict(
                args=[{'visible': station_traces}],
                label=station,
                method='update'
            )
        )

    # 添加下拉菜单到图表
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction='down',
                showactive=True,
                x=0.1,
                xanchor='left',
                y=1.15,
                yanchor='top'
            )
        ]
    )

    # 添加注释说明下拉菜单
    fig.update_layout(
        annotations=[
            dict(
                text='选择站点:',
                x=0,
                y=1.15,
                xref='paper',
                yref='paper',
                showarrow=False
            )
        ]
    )

    # 更新图表布局
    fig.update_layout(
        title='纽约地铁站点客流量预测对比',
        xaxis_title='日期',
        yaxis_title='客流量',
        template='plotly_white',
        height=700,
        width=1200,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )

    # 保存为HTML文件
    py.plot(fig, filename='station_comparison_interactive.html', auto_open=False)
    print("交互式站点对比可视化已生成：station_comparison_interactive.html")


def visualize_entries_exits_comparison(stations_results):
    """
    创建站点进出口客流量对比的交互式可视化

    参数:
    stations_results (dict): 包含每个站点模型和预测结果的字典

    返回:
    None: 函数将生成HTML文件
    """
    print("创建站点进出口客流量对比的交互式可视化...")

    # 创建站点选择的下拉菜单数据
    station_options = list(stations_results.keys())

    # 创建交互式图表
    fig = go.Figure()

    # 初始显示的站点
    initial_station = 'ALL_STATIONS'

    # 为每个站点添加进出口客流量曲线（默认隐藏）
    for station_name, result in stations_results.items():
        # 直接从结果中获取数据，而不是使用未定义的 stations_data
        df = result['data']

        visible = (station_name == initial_station)

        # 添加进站客流量曲线
        fig.add_trace(
            go.Scatter(
                x=df['ds'],
                y=df['ENTRIES_DIFF'] if 'ENTRIES_DIFF' in df.columns else df['y'],  # 兼容性处理
                mode='lines',
                name=f'{station_name} - 进站量',
                line=dict(color='green', width=2),
                visible=visible
            )
        )

        # 添加出站客流量曲线
        fig.add_trace(
            go.Scatter(
                x=df['ds'],
                y=df['EXITS_DIFF'] if 'EXITS_DIFF' in df.columns else df['y'],  # 兼容性处理
                mode='lines',
                name=f'{station_name} - 出站量',
                line=dict(color='red', width=2),
                visible=visible
            )
        )

    # 创建下拉菜单
    dropdown_buttons = []
    for station in station_options:
        station_index = station_options.index(station)
        station_traces = [False] * len(fig.data)

        # 为当前站点设置可见性
        start_index = station_index * 2
        for i in range(2):  # 每个站点有2个曲线：进站量和出站量
            if start_index + i < len(station_traces):
                station_traces[start_index + i] = True

        dropdown_buttons.append(
            dict(
                args=[{'visible': station_traces}],
                label=station,
                method='update'
            )
        )

    # 添加下拉菜单到图表
    fig.update_layout(
        updatemenus=[
            dict(
                buttons=dropdown_buttons,
                direction='down',
                showactive=True,
                x=0.1,
                xanchor='left',
                y=1.15,
                yanchor='top'
            )
        ]
    )

    # 添加注释说明下拉菜单
    fig.update_layout(
        annotations=[
            dict(
                text='选择站点:',
                x=0,
                y=1.15,
                xref='paper',
                yref='paper',
                showarrow=False
            )
        ]
    )

    # 更新图表布局
    fig.update_layout(
        title='纽约地铁站点进出站客流量对比',
        xaxis_title='日期',
        yaxis_title='客流量',
        template='plotly_white',
        height=700,
        width=1200,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=-0.2,
            xanchor='center',
            x=0.5
        )
    )

    # 保存为HTML文件
    py.plot(fig, filename='entries_exits_comparison_interactive.html', auto_open=False)
    print("交互式进出站对比可视化已生成：entries_exits_comparison_interactive.html")


def create_performance_comparison(stations_results):
    """
    创建各站点模型性能对比的可视化

    参数:
    stations_results (dict): 包含每个站点模型和预测结果的字典

    返回:
    None: 函数将生成可视化图表
    """
    print("创建各站点模型性能对比...")

    # 计算每个站点模型的性能指标
    performance_metrics = {}

    for station_name, result in stations_results.items():
        df = result['data']
        forecast = result['forecast']

        # 将预测结果与实际值合并
        evaluation = pd.merge(
            df[['ds', 'y']],
            forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
            on='ds',
            how='left'
        )

        # 计算评估指标
        evaluation['error'] = evaluation['y'] - evaluation['yhat']
        evaluation['abs_error'] = np.abs(evaluation['error'])
        evaluation['squared_error'] = evaluation['error'] ** 2

        # 计算MAE, RMSE, MAPE
        mae = evaluation['abs_error'].mean()
        rmse = np.sqrt(evaluation['squared_error'].mean())
        # 避免除以零
        evaluation['abs_pct_error'] = evaluation['abs_error'] / evaluation['y'].replace(0, np.nan) * 100
        mape = evaluation['abs_pct_error'].mean()

        # 检查预测区间覆盖率
        evaluation['in_range'] = (evaluation['y'] >= evaluation['yhat_lower']) & (
                    evaluation['y'] <= evaluation['yhat_upper'])
        coverage = evaluation['in_range'].mean() * 100

        performance_metrics[station_name] = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Coverage': coverage
        }

    # 将性能指标转换为DataFrame
    metrics_df = pd.DataFrame.from_dict(performance_metrics, orient='index')

    # 创建交互式条形图
    fig = go.Figure()

    metrics = ['MAE', 'RMSE', 'MAPE', 'Coverage']
    colors = ['royalblue', 'crimson', 'green', 'orange']

    # 初始度量值
    initial_metric = 'MAE'

    for i, metric in enumerate(metrics):
        visible = (metric == initial_metric)

        fig.add_trace(
            go.Bar(
                x=metrics_df.index,
                y=metrics_df[metric],
                name=metric,
                marker_color=colors[i],
                visible=visible
            )
        )

    # 创建按钮用于切换不同的度量
    buttons = []
    for i, metric in enumerate(metrics):
        buttons.append(
            dict(
                args=[{'visible': [i == j for j in range(len(metrics))]}],
                label=metric,
                method='update'
            )
        )

    fig.update_layout(
        updatemenus=[
            dict(
                buttons=buttons,
                direction='down',
                pad={'r': 10, 't': 10},
                showactive=True,
                x=0.1,
                xanchor='left',
                y=1.15,
                yanchor='top'
            )
        ]
    )

    # 添加注释说明按钮
    fig.update_layout(
        annotations=[
            dict(
                text='选择指标:',
                x=0,
                y=1.15,
                xref='paper',
                yref='paper',
                showarrow=False
            )
        ]
    )

    # 更新图表布局
    fig.update_layout(
        title='各站点模型性能指标对比',
        xaxis_title='站点',
        yaxis_title='指标值',
        template='plotly_white',
        height=600,
        width=1000
    )

    # 保存为HTML文件
    py.plot(fig, filename='model_performance_comparison.html', auto_open=False)
    print("模型性能对比可视化已生成：model_performance_comparison.html")


def main():
    """主函数"""
    try:
        # 1. 加载数据
        df = load_turnstile_data()

        # 2. 数据预处理
        df_processed = preprocess_data(df)

        # 3. 按天聚合数据
        df_daily = aggregate_data(df_processed, freq='D')

        # 4. 分析站点客流量
        station_traffic = analyze_station_traffic(df_daily)

        # 5. 选择分析方式
        # 可以修改为分析特定站点或者Top N站点
        # prophet_df = prepare_time_series(df_daily, target_station="14 ST-UNION SQ")
        # prophet_df = prepare_time_series(df_daily, top_n_stations=5)
        prophet_df = prepare_time_series(df_daily)  # 分析所有站点总流量

        # 6. 添加假日信息
        holidays_df = add_holidays(prophet_df)

        # 7. 训练Prophet模型并预测
        model, forecast = train_prophet_model(prophet_df, forecast_periods=90, holidays_df=holidays_df)

        # 8. 可视化预测结果
        visualize_forecast(model, forecast, prophet_df, title='纽约地铁客流量预测')

        # 9. 分析假日效应
        holiday_effects = analyze_holiday_effects(forecast)

        # 10. 评估模型性能
        metrics = evaluate_model(prophet_df, forecast)

        # 11. 院长要求新增: 绘制特定站点的进出口客流量对比图
        # target_stations = ["14 ST-UNION SQ", "34 ST-HERALD SQ", "34 ST-PENN STA"]
        # plot_station_comparison(df_daily, target_stations)

        # 12. 院长要求新增：创建交互式图表
        # target_stations = ["14 ST-UNION SQ", "34 ST-HERALD SQ", "34 ST-PENN STA"]
        # create_interactive_comparison(df_daily, target_stations)
        # create_interactive_forecast_comparison(df_daily, target_stations)
        # print("\n分析完成! 已生成多个可视化图表。")

        # 13. 保存预测结果到CSV
        forecast.to_csv('prophet_forecast_results.csv', index=False)
        print("预测结果已保存到 'prophet_forecast_results.csv'")

        # 14. 新增: 针对特定站点的分析
        # print("\n开始针对特定站点的分析...")

        # 选择要分析的特定站点
        # selected_stations = ['14 ST-UNION SQ', '34 ST-HERALD SQ', '34 ST-PENN STA']
        #
        # stations_data = prepare_selected_stations_data(df_daily, selected_stations)
        #
        # stations_results = train_models_for_stations(stations_data, forecast_periods=90, holidays_df=holidays_df)
        #
        # create_interactive_station_comparison(stations_results)
        #
        # visualize_entries_exits_comparison(stations_results)
        #
        # create_performance_comparison(stations_results)

        print("\n分析完成! 已生成多个可视化图表。")
        print("1. 总体客流量预测: prophet_forecast.png, prophet_components.png")
        print("2. 交互式站点预测对比: station_comparison_interactive.html")
        print("3. 交互式进出站客流量对比: entries_exits_comparison_interactive.html")
        print("4. 各站点模型性能对比: model_performance_comparison.html")

    except Exception as e:
        print(f"程序执行过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()