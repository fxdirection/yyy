import os
import pandas as pd
from tqdm import tqdm


def merge_excels_with_subfolder_marker(root_folder, output_folder):
    """
    合并同名Excel文件，使用子文件夹名称作为区分标识
    功能：
    1. 遍历所有子文件夹中的同名Excel文件
    2. 合并时添加"数据来源"列记录子文件夹名称
    3. 每个同名Excel文件合并为一个单独文件
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有子文件夹（排除隐藏文件夹）
    subfolders = [f for f in os.listdir(root_folder)
                  if os.path.isdir(os.path.join(root_folder, f)) and not f.startswith('.')]

    if not subfolders:
        print("错误：未找到任何有效子文件夹")
        return

    # 获取第一个子文件夹中的Excel文件列表作为模板
    sample_files = [f for f in os.listdir(os.path.join(root_folder, subfolders[0]))
                    if f.endswith(('.xlsx', '.xls')) and not f.startswith('~$')]  # 排除临时文件

    if not sample_files:
        print("错误：子文件夹中没有找到Excel文件")
        return

    # 为每个同名Excel文件创建合并结果
    for filename in tqdm(sample_files, desc="正在合并文件"):
        merged_data = []

        # 遍历所有子文件夹收集数据
        for folder in subfolders:
            file_path = os.path.join(root_folder, folder, filename)

            try:
                if os.path.exists(file_path):
                    # 读取Excel文件
                    df = pd.read_excel(file_path)

                    # 添加子文件夹标识列（放在第一列）
                    df.insert(0, '数据来源', folder)

                    merged_data.append(df)
                else:
                    print(f"警告：{folder} 中缺少文件 {filename}")

            except Exception as e:
                print(f"处理 {file_path} 时出错: {str(e)}")
                continue

        if merged_data:
            # 合并所有数据
            result_df = pd.concat(merged_data, ignore_index=True)

            # 保存结果（原文件名前加"合并_"前缀）
            output_path = os.path.join(output_folder, f"合并_{filename}")

            # 根据文件类型选择保存引擎
            if filename.endswith('.xlsx'):
                result_df.to_excel(output_path, index=False, engine='openpyxl')
            else:
                result_df.to_excel(output_path, index=False)

            print(f"/n{filename} 合并完成：")
            print(f"├─ 合并了 {len(merged_data)} 个子文件夹的数据")
            print(f"├─ 总行数: {len(result_df)}")
            print(f"└─ 已保存到: {output_path}")

            # 显示各子文件夹数据量统计
            print("/n数据来源分布：")
            print(result_df['数据来源'].value_counts().to_string())
        else:
            print(f"/n警告：{filename} 没有可合并的数据")

    print("/n所有文件合并完成！")


# 使用示例
if __name__ == "__main__":
    # 输入路径（包含子文件夹的根目录）
    input_path = "C:/Users/fangxiang/Downloads/XGBoost_res"
    # 输出路径（合并结果保存位置）
    output_path = "C:/Users/fangxiang/Downloads/合并结果"

    merge_excels_with_subfolder_marker(input_path, output_path)