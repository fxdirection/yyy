import pandas as pd
import numpy as np
import os

# 设置文件夹路径
folder_path = 'C:/Users/fangxiang/Downloads/GMM_5+5/'  # 替换为您的文件夹路径
output_path = 'C:/Users/fangxiang/Downloads/reprocess/'
# 遍历文件夹中的所有文件
for filename in os.listdir(folder_path):
    if filename.endswith('.xlsx') or filename.endswith('.xls'):
        # 完整的文件路径
        file_path = os.path.join(folder_path, filename)

        # 读取Excel文件
        df = pd.read_excel(file_path)

        # 计算第14-19列的和 (列索引13-18)
        sum_14_19 = df.iloc[:, 13:18].sum(axis=1)

        # 计算第24-29列的和 (列索引23-28)
        sum_24_29 = df.iloc[:, 23:28].sum(axis=1)

        # 将两个和相加并存入第40列 (索引39)
        # 如果第40列不存在，会自动创建
        df.iloc[:, 39] = sum_14_19 + sum_24_29

        # 根据第40列的值创建标签
        percentiles = df.iloc[:, 39].quantile([0.25, 0.5, 0.75])


        # 创建标签函数
        def create_label(x):
            if x <= percentiles[0.25]:
                return 4
            elif x <= percentiles[0.5]:
                return 3
            elif x <= percentiles[0.75]:
                return 2
            else:
                return 1


        # 应用标签函数并存入第39列 (索引38)
        # 如果第39列不存在，会自动创建
        df.iloc[:, 38] = df.iloc[:, 39].apply(create_label)

        # 保存修改后的文件，添加前缀"processed_"
        processed_filename = filename
        processed_file_path = os.path.join(output_path, processed_filename)

        # 保存为Excel文件
        df.to_excel(processed_file_path, index=False)

        print(f"已处理文件: {filename} -> 保存为: {processed_filename}")

print("所有文件处理完成！")