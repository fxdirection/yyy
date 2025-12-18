import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================== Subject Predictor Configuration ==================
subject_predictors = {
    # Natural Sciences
    "physics": {
        "n_clusters": 4,
        "match_papers_L": 100,
        "distance_metric": "euclidean",
        "time_decay_weights": [0.5, 1.0, 1.5],
        "covariance_type": "full",
        "scaling_method": "standard"
    },
    "chemistry": {
        "n_clusters": 3,
        "match_papers_L": 100,
        "distance_metric": "euclidean",
        "covariance_type": "tied",
        "scaling_method": "robust",
        "log_transform": True,
        "dynamic_L": True
    },
    "biology": {
        "n_clusters": 5,
        "match_papers_L": 100,
        "distance_metric": "euclidean",
        "covariance_type": "full",
        "scaling_method": "minmax",
        "dynamic_L": True,
        "use_journal_impact": True
    },

    # Engineering
    "computer_science": {
        "n_clusters": 2,
        "match_papers_L": 100,
        "distance_metric": "manhattan",
        "covariance_type": "diag",
        "scaling_method": "robust",
        "dynamic_L": True
    },
    "mechanical_engineering": {
        "n_clusters": 3,
        "match_papers_L": 100,
        "distance_metric": "euclidean",
        "covariance_type": "tied",
        "scaling_method": "standard",
        "dynamic_L": True,
        "cluster_weighted_mean": True
    },

    # Medicine
    "medical": {
        "n_clusters": 4,
        "match_papers_L": 100,
        "distance_metric": "euclidean",
        "covariance_type": "tied",
        "scaling_method": "robust",
    },

    # Social Sciences
    "economics": {
        "n_clusters": 3,
        "match_papers_L": 100,
        "distance_metric": "cosine",
        "covariance_type": "full",
        "scaling_method": "robust",
        "use_author_hindex": True
    },

    "psychology": {
        "n_clusters": 4,
        "match_papers_L": 100,
        "distance_metric": "cosine",
        "time_decay_weights": [0.8, 1.0, 1.2],
        "covariance_type": "diag",
        "scaling_method": "standard"
    },
    "sociology": {
        "n_clusters": 3,
        "match_papers_L": 100,
        "distance_metric": "cosine",
        "covariance_type": "tied",
        "scaling_method": "minmax",
        "dynamic_L": True,
        "log_transform": True
    },

    # Humanities
    "history": {
        "n_clusters": 2,
        "match_papers_L": 100,
        "distance_metric": "manhattan",
        "covariance_type": "spherical",
        "scaling_method": "minmax",
        "decay_factor": 0.8
    },
    "philosophy": {
        "n_clusters": 2,
        "match_papers_L": 100,
        "distance_metric": "manhattan",
        "covariance_type": "spherical",
        "scaling_method": None,
        "scaling_comment": "standardization disabled"
    }
}

# Chinese to English mapping
chinese_to_english = {
    # Natural Sciences
    "物理学": "physics",
    "化学": "chemistry",
    "生物学": "biology",

    # Engineering
    "计算机科学": "computer_science",
    "工程技术": "mechanical_engineering",

    # Medicine
    "医学": "medical",

    # Social Sciences
    "经济学": "economics",
    "心理学": "psychology",
    "社会学": "sociology",

    # Humanities
    "历史学": "history",
    "哲学": "philosophy",
}


def get_predictor_config(subject_input):
    """Get predictor configuration by subject name"""
    if subject_input in chinese_to_english:
        english_name = chinese_to_english[subject_input]
        return subject_predictors.get(english_name, None), english_name

    lower_input = subject_input.lower()
    for eng_name in subject_predictors:
        if eng_name.lower() == lower_input:
            return subject_predictors[eng_name], eng_name

    return None, None

class CitationPredictor:
    def __init__(self, train_data, test_data, test_indices, n_train_years=3, m_pred_years=7,
                 n_clusters=3, distance_metric="euclidean", scaling_method="standard", **kwargs):
        """初始化预测器"""
        # 训练集数据
        self.train_titles = train_data.iloc[:, 0].values
        self.train_db = train_data.iloc[:, 13:23].values  # 假设13-22列是引用数据

        # 测试集数据
        self.test_titles = test_data.iloc[:, 0].values
        self.test_db = test_data.iloc[:, 13:23].values
        self.test_indices = test_indices  # 存储原始索引

        self.n_train = n_train_years
        self.n_pred = m_pred_years
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.scaling_method = scaling_method
        self.extra_params = kwargs

        # 初始化标准化器
        if scaling_method == "standard":
            self.scaler = StandardScaler()
        elif scaling_method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            from sklearn.preprocessing import RobustScaler
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def _match_papers(self, test_paper, L=100):
        """在训练集中找到与测试论文最相似的L篇论文"""
        train_X = self.train_db[:, :self.n_train]

        if self.distance_metric == "euclidean":
            distances = euclidean_distances([test_paper[:self.n_train]], train_X)[0]
        elif self.distance_metric == "manhattan":
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances([test_paper[:self.n_train]], train_X)[0]
        elif self.distance_metric == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances([test_paper[:self.n_train]], train_X)[0]
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

        return np.argsort(distances)[:L]

    def _predict_with_gmm(self, matched_indices):
        """使用GMM进行预测"""
        future_cites = self.train_db[matched_indices, self.n_train:self.n_train + self.n_pred]

        if len(future_cites) < self.n_clusters:
            return np.zeros(self.n_pred)

        # 数据标准化
        if self.scaler:
            scaled_data = self.scaler.fit_transform(future_cites)
        else:
            scaled_data = future_cites

        # GMM建模
        gmm = GaussianMixture(
            n_components=min(self.n_clusters, len(future_cites)),
            covariance_type=self.extra_params.get("covariance_type", "full"),
            max_iter=self.extra_params.get("gmm_max_iter", 200),
            random_state=42
        )
        gmm.fit(scaled_data)

        # 逆标准化预测结果并四舍五入
        if self.scaler:
            pred = self.scaler.inverse_transform(gmm.means_).mean(axis=0)
        else:
            pred = gmm.means_.mean(axis=0)

        return np.round(pred).astype(int)

    def predict_test_set(self):
        """预测测试集所有论文"""
        predictions = []

        for i in range(len(self.test_db)):
            test_paper = self.test_db[i]
            if len(test_paper) < self.n_train + self.n_pred:
                continue

            # 匹配相似论文
            matched_idx = self._match_papers(test_paper, L=100)

            if len(matched_idx) == 0:
                continue

            try:
                # GMM预测
                gmm_pred = self._predict_with_gmm(matched_idx)
                predictions.append({
                    'Paper Title': self.test_titles[i],
                    'True Citations': test_paper[self.n_train:self.n_train + self.n_pred],
                    'Predicted Citations': gmm_pred,
                    'Matched Papers': len(matched_idx),
                    'Original Index': self.test_indices[i]  # 存储原始索引
                })
            except Exception as e:
                continue

        print(f"\n成功处理 {len(predictions)}/{len(self.test_db)} 篇测试论文")
        return predictions

    def save_predictions(self, predictions, output_path, original_data):
        """保存预测结果到Excel"""
        # 创建原始数据的副本
        result_df = original_data.copy()

        # 添加测试集标记列
        if 'Test_Set_Flag' not in result_df.columns:
            result_df['Test_Set_Flag'] = 0

        # 标记测试集论文
        test_indices = [p['Original Index'] for p in predictions]
        result_df.loc[test_indices, 'Test_Set_Flag'] = 1

        # 移除已有的预测列
        for year in range(self.n_pred):
            col_name = f'GMM Pred Year {self.n_train + year + 1}'
            if col_name in result_df.columns:
                result_df.drop(col_name, axis=1, inplace=True)

        # 插入新的预测列
        for year in range(self.n_pred):
            col_name = f'GMM Pred Year {self.n_train + year + 1}'
            result_df.insert(23 + year, col_name, np.nan)

        # 填充预测结果
        for pred in predictions:
            idx = pred['Original Index']
            for year_idx in range(self.n_pred):
                col_name = f'GMM Pred Year {self.n_train + year_idx + 1}'
                result_df.at[idx, col_name] = pred['Predicted Citations'][year_idx]

        # 确保所有要求的列都存在
        expected_columns = [
            '期刊_ref', '期刊分区', '领域', '学科', '标题', '标题长度', '作者数量',
            '作者h指数', '页码', '总引用次数', '参考文献数量', '五年影响因子', '出版日期',
            'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'year_6',
            'year_7', 'year_8', 'year_9', 'year_10',
            'GMM Pred Year 4', 'GMM Pred Year 5', 'GMM Pred Year 6',
            'GMM Pred Year 7', 'GMM Pred Year 8', 'GMM Pred Year 9',
            'GMM Pred Year 10', '标签', '起始页码', '结束页码', '篇幅', '出版社',
            '十年CNCI', 'Test_Set_Flag'
        ]

        # 添加缺失的列
        for col in expected_columns:
            if col not in result_df.columns:
                result_df[col] = np.nan

        # 重新排列列顺序
        result_df = result_df[expected_columns]

        result_df.to_excel(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")


def process_single_file(input_file, output_file, subject):
    """处理单个文件"""
    try:
        print(f"\n处理文件: {input_file}")
        print(f"学科: {subject}")

        # 读取数据并确保数值列是数字类型
        raw_data = pd.read_excel(input_file)
        for col in raw_data.columns[13:23]:  # 假设13-22列是引用数据
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').fillna(0)

        # 获取原始索引
        indices = np.arange(len(raw_data))

        # 9:1 train/test split
        train_data, test_data = train_test_split(
            raw_data,
            test_size=0.1,
            random_state=42
        )
        test_indices = raw_data.index.difference(train_data.index).values

        # 初始化预测器
        predictor = CitationPredictor(
            train_data=train_data,
            test_data=test_data,
            test_indices=test_indices,
            n_train_years=3,
            m_pred_years=7,
            n_clusters=3,
            distance_metric="euclidean",
            scaling_method="standard"
        )

        # 进行预测
        predictions = predictor.predict_test_set()

        if predictions:
            predictor.save_predictions(predictions, output_file, raw_data)
        else:
            print("未生成有效预测")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    subjects_to_process = ["物理学", "化学", "生物学", "计算机科学", "工程技术", "医学", "经济学", "心理学", "社会学",
                           "历史学", "哲学"]
    print("开始批量处理...")

    for subject in subjects_to_process:
        input_file = f"/predictions/predicted_{subject}.xlsx"
        output_file = f"/with_gmm/predicted_{subject}.xlsx"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        process_single_file(input_file, output_file, subject)

    print("\n批量处理完成!")