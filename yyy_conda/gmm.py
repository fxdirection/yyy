# GMM
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
import warnings

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)

# ================== 学科预测器配置 ==================
subject_predictors = {
    # 自然科学类
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

    # 工程类
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

    # 医学类
    "medical": {
        "n_clusters": 4,
        "match_papers_L": 100,
        "distance_metric": "euclidean",
        "covariance_type": "tied",
        "scaling_method": "robust",
    },

    # 社会科学类
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

    # 人文学科类
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

# ================== 中英文映射 ==================
chinese_to_english = {
    # 自然科学
    "物理学": "physics",
    "化学": "chemistry",
    "生物学": "biology",

    # 工程
    "计算机科学": "computer_science",
    "工程技术": "mechanical_engineering",

    # 医学
    "医学": "medical",

    # 社会科学
    "经济学": "economics",
    "心理学": "psychology",
    "社会学": "sociology",

    # 人文学科
    "历史学": "history",
    "哲学": "philosophy",
}


def get_predictor_config(subject_input):
    """获取学科配置"""
    if subject_input in chinese_to_english:
        english_name = chinese_to_english[subject_input]
        return subject_predictors.get(english_name, None), english_name

    lower_input = subject_input.lower()
    for eng_name in subject_predictors:
        if eng_name.lower() == lower_input:
            return subject_predictors[eng_name], eng_name

    return None, None


class CitationPredictor:
    def __init__(self, train_data, test_data, test_indices, n_train_years=5, m_pred_years=5,
                 n_clusters=3, distance_metric="euclidean", scaling_method="standard", **kwargs):
        """初始化预测器"""
        # 训练集数据
        self.train_titles = train_data.iloc[:, 0].values
        self.train_db = train_data.iloc[:, 13:23].values  # 假设13-22列是引用数据
        self.train_indices = train_data.index.values  # 存储训练集原始索引

        # 测试集数据
        self.test_titles = test_data.iloc[:, 0].values
        self.test_db = test_data.iloc[:, 13:23].values
        self.test_indices = test_indices  # 存储测试集原始索引

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
            self.scaler = MinMaxScaler()
        elif scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            self.scaler = None

    def _match_papers(self, query_paper, paper_db, L=100):
        """在指定论文集中找到与查询论文最相似的L篇论文"""
        db_X = paper_db[:, :self.n_train]

        if self.distance_metric == "euclidean":
            distances = euclidean_distances([query_paper[:self.n_train]], db_X)[0]
        elif self.distance_metric == "manhattan":
            from sklearn.metrics.pairwise import manhattan_distances
            distances = manhattan_distances([query_paper[:self.n_train]], db_X)[0]
        elif self.distance_metric == "cosine":
            from sklearn.metrics.pairwise import cosine_distances
            distances = cosine_distances([query_paper[:self.n_train]], db_X)[0]
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

        return np.argsort(distances)[:L]

    def _predict_with_gmm(self, matched_indices, paper_db):
        """使用GMM进行预测"""
        future_cites = paper_db[matched_indices, self.n_train:self.n_train + self.n_pred]

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

    def predict_train_set(self):
        """预测训练集所有论文的后7年引用量"""
        train_predictions = []

        for i in range(len(self.train_db)):
            train_paper = self.train_db[i]
            if len(train_paper) < self.n_train + self.n_pred:
                continue

            # 匹配相似论文（从训练集中排除自己）
            all_indices = np.arange(len(self.train_db))
            other_indices = all_indices[all_indices != i]
            matched_idx = self._match_papers(train_paper, self.train_db[other_indices], L=100)

            # 调整索引，因为排除了自己
            matched_idx = other_indices[matched_idx]

            if len(matched_idx) == 0:
                continue

            try:
                # GMM预测
                gmm_pred = self._predict_with_gmm(matched_idx, self.train_db)
                train_predictions.append({
                    'Paper Title': self.train_titles[i],
                    'True Citations': train_paper[self.n_train:self.n_train + self.n_pred],
                    'Predicted Citations': gmm_pred,
                    'Matched Papers': len(matched_idx),
                    'Original Index': self.train_indices[i]  # 存储原始索引
                })
            except Exception as e:
                continue

        print(f"\n成功处理 {len(train_predictions)}/{len(self.train_db)} 篇训练论文")
        return train_predictions

    def predict_test_set(self):
        """预测测试集所有论文的后7年引用量"""
        test_predictions = []

        for i in range(len(self.test_db)):
            test_paper = self.test_db[i]
            if len(test_paper) < self.n_train + self.n_pred:
                continue

            # 匹配相似论文（从训练集中）
            matched_idx = self._match_papers(test_paper, self.train_db, L=100)

            if len(matched_idx) == 0:
                continue

            try:
                # GMM预测
                gmm_pred = self._predict_with_gmm(matched_idx, self.train_db)
                test_predictions.append({
                    'Paper Title': self.test_titles[i],
                    'True Citations': test_paper[self.n_train:self.n_train + self.n_pred],
                    'Predicted Citations': gmm_pred,
                    'Matched Papers': len(matched_idx),
                    'Original Index': self.test_indices[i]  # 存储原始索引
                })
            except Exception as e:
                continue

        print(f"\n成功处理 {len(test_predictions)}/{len(self.test_db)} 篇测试论文")
        return test_predictions

    def save_predictions(self, train_predictions, test_predictions, output_path, original_data):
        """保存预测结果到Excel，包含训练集和测试集的预测"""
        # 创建原始数据的副本
        result_df = original_data.copy()

        # 1. 添加AN列（N-P列和X-AD列的和）
        # 假设N-P列是13-15列，X-AD列是23-29列
        result_df[39] = result_df.iloc[:, 13:18].sum(axis=1) + result_df.iloc[:, 28:30].sum(axis=1)

        # 2. 添加AM列标签（基于AN列的四分位数）
        quantiles = result_df[39].quantile([0.25, 0.5, 0.75])
        result_df[38] = pd.cut(
            result_df[39],
            bins=[-np.inf, quantiles[0.25], quantiles[0.5], quantiles[0.75], np.inf],
            labels=[4, 3, 2, 1]  # 注意顺序是反的，因为cut是从小到大
        ).astype(int)

        # 3. 添加训练集/测试集标记
        result_df['Train_Set_Flag'] = 0
        result_df['Test_Set_Flag'] = 0

        # 标记训练集论文
        if train_predictions:
            train_indices = [p['Original Index'] for p in train_predictions]
            result_df.loc[train_indices, 'Train_Set_Flag'] = 1

        # 标记测试集论文
        if test_predictions:
            test_indices = [p['Original Index'] for p in test_predictions]
            result_df.loc[test_indices, 'Test_Set_Flag'] = 1

        # 4. 更新预测列
        for year in range(self.n_pred):
            col_name = f'GMM Pred Year {self.n_train + year + 1}'
            if col_name in result_df.columns:
                result_df.drop(col_name, axis=1, inplace=True)
            result_df.insert(23 + year, col_name, np.nan)

        # 填充训练集预测结果
        for pred in train_predictions:
            idx = pred['Original Index']
            for year_idx in range(self.n_pred):
                col_name = f'GMM Pred Year {self.n_train + year_idx + 1}'
                result_df.at[idx, col_name] = pred['Predicted Citations'][year_idx]

        # 填充测试集预测结果
        for pred in test_predictions:
            idx = pred['Original Index']
            for year_idx in range(self.n_pred):
                col_name = f'GMM Pred Year {self.n_train + year_idx + 1}'
                result_df.at[idx, col_name] = pred['Predicted Citations'][year_idx]

        # 5. 确保所有要求的列都存在
        expected_columns = [
            '期刊_ref', '期刊分区', '领域', '学科', '标题', '标题长度', '作者数量',
            '作者h指数', '页码', '总引用次数', '参考文献数量', '五年影响因子', '出版日期',
            'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'year_6',
            'year_7', 'year_8', 'year_9', 'year_10',
            'GMM Pred Year 4', 'GMM Pred Year 5', 'GMM Pred Year 6',
            'GMM Pred Year 7', 'GMM Pred Year 8', 'GMM Pred Year 9',
            'GMM Pred Year 10', '标签', '起始页码', '结束页码', '篇幅', '出版社',
            '十年CNCI', 'Train_Set_Flag', 'Test_Set_Flag', '新标签', '预测总引用量'
        ]

        # 添加缺失的列
        for col in expected_columns:
            if col not in result_df.columns:
                result_df[col] = np.nan

        # 重新排列列顺序
        result_df = result_df[expected_columns]

        # 保存主结果文件
        result_df.to_excel(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")

        # 6. 计算并保存评估指标
        if '标签' in result_df.columns:
            self._save_evaluation_metrics(result_df, output_path)

    def _save_evaluation_metrics(self, df, output_path):
        """计算并保存AE和AM列的评估指标"""
        # 计算指标
        true = pd.to_numeric(df['标签'], errors='coerce').fillna(0).astype(int)
        pred = pd.to_numeric(df['新标签'], errors='coerce').fillna(0).astype(int)

        metrics = {
            'Weighted Kappa': cohen_kappa_score(true, pred, weights='quadratic'),
            'Weighted F1': f1_score(true, pred, average='weighted'),
            'MAE': mean_absolute_error(true, pred),
            'File': os.path.basename(output_path)
        }

        # 创建评估目录
        eval_dir = os.path.join(os.path.dirname(output_path), "evaluation")
        os.makedirs(eval_dir, exist_ok=True)

        eval_file = os.path.join(eval_dir, "an_am_metrics.xlsx")

        # 如果文件已存在，读取并追加新数据
        if os.path.exists(eval_file):
            existing_df = pd.read_excel(eval_file)
            metrics_df = pd.concat([existing_df, pd.DataFrame([metrics])], ignore_index=True)
        else:
            metrics_df = pd.DataFrame([metrics])

        metrics_df.to_excel(eval_file, index=False)
        print(f"评估指标已保存至: {eval_file}")


def process_single_file(input_file, output_file, subject):
    """处理单个文件，同时预测训练集和测试集"""
    try:
        print(f"\n处理文件: {input_file}")
        print(f"学科: {subject}")

        # 读取数据并确保数值列是数字类型
        raw_data = pd.read_excel(input_file)
        for col in raw_data.columns[13:23]:  # 假设13-22列是引用数据
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').fillna(0)

        # 获取配置参数
        config, eng_name = get_predictor_config(subject)
        if not config:
            print(f"未找到学科 {subject} 的配置，使用默认参数")
            config = {
                "n_clusters": 3,
                "distance_metric": "euclidean",
                "scaling_method": "standard"
            }

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
            n_train_years=5,
            m_pred_years=5,
            **config
        )

        # 进行训练集和测试集预测
        train_predictions = predictor.predict_train_set()
        test_predictions = predictor.predict_test_set()

        if train_predictions or test_predictions:
            predictor.save_predictions(train_predictions, test_predictions, output_file, raw_data)
            print(f"训练集预测: {len(train_predictions)}篇, 测试集预测: {len(test_predictions)}篇")
        else:
            print("未生成有效预测")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    subjects_to_process = ["物理学", "化学", "生物学", "计算机科学", "工程技术", "医学",
                           "经济学", "心理学", "社会学", "历史学", "哲学"]
    print("开始批量处理...")

    # 创建输出目录
    os.makedirs("/gmm", exist_ok=True)
    os.makedirs("/gmm/evaluation", exist_ok=True)

    for subject in subjects_to_process:
        input_file = f"/data/{chinese_to_english[subject]}.xlsx"
        output_file = f"/gmm/{chinese_to_english[subject]}.xlsx"
        process_single_file(input_file, output_file, subject)

    print("\n批量处理完成!")