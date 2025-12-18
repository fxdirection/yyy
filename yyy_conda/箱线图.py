import pandas as pd
import matplotlib.pyplot as plt
import os
import seaborn as sns

# 设置文件夹路径和输出图片路径
folder_path = "C:/Users/fangxiang/Desktop/0428yyy/cs"  # 替换为你的文件夹路径
output_path = 'boxplot_comparison.png'

# 获取所有Excel文件
excel_files = [f for f in os.listdir(folder_path) if f.endswith(('.xlsx', '.xls'))]
if len(excel_files) != 3:
    raise ValueError("文件夹中应包含3个Excel文件")

# 创建图形
plt.figure(figsize=(12, 8))
sns.set_style("whitegrid")

# 读取并绘制每个文件的数据
all_data = []
for file in excel_files:
    # 读取Excel文件
    file_path = os.path.join(folder_path, file)
    df = pd.read_excel(file_path)

    # 假设所有Excel有相同的列结构，取第一列数值数据
    # 可根据实际需求修改列名
    column_name = df.columns[0]  # 取第一列
    data = df[column_name].dropna()

    # 添加到绘图数据
    file_name = os.path.splitext(file)[0]  # 去除扩展名
    temp_df = pd.DataFrame({
        'Value': data,
        'Source': [file_name] * len(data)
    })
    all_data.append(temp_df)

# 合并所有数据
combined_data = pd.concat(all_data, ignore_index=True)

# 绘制箱线图
sns.boxplot(x='Source', y='Value', data=combined_data, palette="Set2")
plt.title('Comparison of Three Excel Files', fontsize=16)
plt.xlabel('Excel File Name', fontsize=14)
plt.ylabel('Value Distribution', fontsize=14)

# 添加数据点
sns.stripplot(x='Source', y='Value', data=combined_data,
              color='black', alpha=0.5, jitter=True)

# 保存并显示图形
plt.tight_layout()
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()