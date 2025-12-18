import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import warnings
import torch

# 忽略特定警告
warnings.filterwarnings("ignore", category=UserWarning)


# Check GPU availability and display connection status
def check_gpu_status():
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0)
        print(f"GPU Connected: {gpu_count} GPU(s) available. Using {gpu_name}.")
        return True
    else:
        print("GPU Not Connected: Running on CPU.")
        return False


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
    """Get predictor configuration by subject name"""
    if subject_input in chinese_to_english:
        english_name = chinese_to_english[subject_input]
        return subject_predictors.get(english_name, subject_predictors['_default']), english_name

    lower_input = subject_input.lower()
    for eng_name in subject_predictors:
        if eng_name.lower() == lower_input:
            return subject_predictors[eng_name], eng_name

    return subject_predictors['_default'], None


class CitationPredictor:
    def __init__(self, train_data, test_data, train_indices, test_indices, n_train_years=3, m_pred_years=7,
                 n_clusters=3, distance_metric="euclidean", scaling_method="standard", **kwargs):
        """初始化预测器"""
        self.train_titles = train_data.iloc[:, 0].values
        self.train_db = train_data.iloc[:, 13:23].values
        self.test_titles = test_data.iloc[:, 0].values
        self.test_db = test_data.iloc[:, 13:23].values
        self.train_indices = train_indices
        self.test_indices = test_indices
        self.n_train = n_train_years
        self.n_pred = m_pred_years
        self.n_clusters = n_clusters
        self.distance_metric = distance_metric
        self.scaling_method = scaling_method
        self.extra_params = kwargs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

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

    def _match_papers(self, paper, L=100):
        """在训练集中找到与给定论文最相似的L篇论文（GPU加速）"""
        train_X = self.train_db[:, :self.n_train]

        # Convert to torch tensors and move to GPU
        paper_tensor = torch.tensor(paper[:self.n_train], dtype=torch.float32).to(self.device)
        train_X_tensor = torch.tensor(train_X, dtype=torch.float32).to(self.device)

        if self.distance_metric == "euclidean":
            # Euclidean distance on GPU
            distances = torch.norm(train_X_tensor - paper_tensor, dim=1)
        elif self.distance_metric == "cosine":
            # Cosine distance on GPU
            norm_paper = torch.norm(paper_tensor)
            norm_train = torch.norm(train_X_tensor, dim=1)
            cosine_sim = torch.matmul(train_X_tensor, paper_tensor) / (norm_train * norm_paper + 1e-8)
            distances = 1 - cosine_sim
        elif self.distance_metric == "manhattan":
            # Manhattan distance on GPU
            distances = torch.sum(torch.abs(train_X_tensor - paper_tensor), dim=1)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")

        # Get indices of L nearest papers
        _, indices = torch.topk(distances, L, largest=False)
        return indices.cpu().numpy()

    def _predict_with_gmm(self, matched_indices):
        """使用GMM进行预测"""
        future_cites = self.train_db[matched_indices, self.n_train:self.n_train + self.n_pred]

        if len(future_cites) < self.n_clusters:
            return np.zeros(self.n_pred)

        if self.scaler:
            scaled_data = self.scaler.fit_transform(future_cites)
        else:
            scaled_data = future_cites

        gmm = GaussianMixture(
            n_components=min(self.n_clusters, len(future_cites)),
            covariance_type=self.extra_params.get("covariance_type", "full"),
            max_iter=self.extra_params.get("gmm_max_iter", 200),
            random_state=42
        )
        gmm.fit(scaled_data)

        if self.scaler:
            pred = self.scaler.inverse_transform(gmm.means_).mean(axis=0)
        else:
            pred = gmm.means_.mean(axis=0)

        return np.round(pred).astype(int)

    def predict_set(self, titles, db, indices, is_train=False):
        """预测给定数据集（训练或测试）的所有论文"""
        predictions = []
        set_type = 'Train' if is_train else 'Test'

        for i in range(len(db)):
            paper = db[i]
            if len(paper) < self.n_train + self.n_pred:
                continue

            matched_idx = self._match_papers(paper, L=self.extra_params.get("match_papers_L", 100))

            if len(matched_idx) == 0:
                continue

            try:
                gmm_pred = self._predict_with_gmm(matched_idx)
                predictions.append({
                    'Paper Title': titles[i],
                    'True Citations': paper[self.n_train:self.n_train + self.n_pred],
                    'Predicted Citations': gmm_pred,
                    'Matched Papers': len(matched_idx),
                    'Original Index': indices[i],
                    'Set Type': set_type
                })
            except Exception as e:
                continue

        print(f"\n成功处理 {len(predictions)}/{len(db)} 篇 {set_type} 论文")
        return predictions

    def predict_train_set(self):
        """预测训练集"""
        return self.predict_set(self.train_titles, self.train_db, self.train_indices, is_train=True)

    def predict_test_set(self):
        """预测测试集"""
        return self.predict_set(self.test_titles, self.test_db, self.test_indices, is_train=False)

    def save_predictions(self, predictions, output_path, original_data):
        """保存预测结果到Excel"""
        result_df = original_data.copy()

        if 'Set_Type' not in result_df.columns:
            result_df['Set_Type'] = 'Unknown'

        for year in range(self.n_pred):
            col_name = f'GMM Pred Year {self.n_train + year + 1}'
            if col_name in result_df.columns:
                result_df.drop(col_name, axis=1, inplace=True)

        for year in range(self.n_pred):
            col_name = f'GMM Pred Year {self.n_train + year + 1}'
            result_df.insert(23 + year, col_name, np.nan)

        for pred in predictions:
            idx = pred['Original Index']
            result_df.loc[idx, 'Set_Type'] = pred['Set Type']
            for year_idx in range(self.n_pred):
                col_name = f'GMM Pred Year {self.n_train + year_idx + 1}'
                result_df.at[idx, col_name] = pred['Predicted Citations'][year_idx]

        expected_columns = [
            '期刊_ref', '期刊分区', '领域', '学科', '标题', '标题长度', '作者数量',
            '作者h指数', '页码', '总引用次数', '参考文献数量', '五年影响因子', '出版日期',
            'year_1', 'year_2', 'year_3', 'year_4', 'year_5', 'year_6',
            'year_7', 'year_8', 'year_9', 'year_10',
            'GMM Pred Year 4', 'GMM Pred Year 5', 'GMM Pred Year 6',
            'GMM Pred Year 7', 'GMM Pred Year 8', 'GMM Pred Year 9',
            'GMM Pred Year 10', '标签', '起始页码', '结束页码', '篇幅', '出版社',
            '十年CNCI', 'Set_Type'
        ]

        for col in expected_columns:
            if col not in result_df.columns:
                result_df[col] = np.nan

        result_df = result_df[expected_columns]
        result_df.to_excel(output_path, index=False)
        print(f"预测结果已保存至: {output_path}")


def process_single_file(input_file, output_file, subject):
    """处理单个文件，使用9:1 train/test split"""
    try:
        print(f"\n处理文件: {input_file}")
        print(f"学科: {subject}")

        raw_data = pd.read_excel(input_file)
        for col in raw_data.columns[13:23]:
            raw_data[col] = pd.to_numeric(raw_data[col], errors='coerce').fillna(0)

        config, _ = get_predictor_config(subject)
        if config is None:
            config = subject_predictors['_default']

        train_data, test_data = train_test_split(
            raw_data,
            test_size=0.1,
            random_state=42
        )
        train_indices = train_data.index.values
        test_indices = test_data.index.values

        predictor = CitationPredictor(
            train_data=train_data,
            test_data=test_data,
            train_indices=train_indices,
            test_indices=test_indices,
            n_train_years=3,
            m_pred_years=7,
            n_clusters=config.get('n_clusters', 3),
            distance_metric=config.get('distance_metric', 'euclidean'),
            scaling_method=config.get('scaling_method', 'standard'),
            match_papers_L=config.get('match_papers_L', 100),
            covariance_type=config.get('covariance_type', 'full')
        )

        test_predictions = predictor.predict_test_set()
        train_predictions = predictor.predict_train_set()
        all_predictions = test_predictions + train_predictions

        if all_predictions:
            predictor.save_predictions(all_predictions, output_file, raw_data)
        else:
            print("未生成有效预测")

    except Exception as e:
        print(f"处理文件时出错: {str(e)}")


if __name__ == "__main__":
    # Display GPU status at the start
    check_gpu_status()

    subjects_to_process = ["物理学", "化学", "生物学", "计算机科学", "工程技术", "医学", "经济学", "心理学", "社会学",
                           "历史学", "哲学"]
    print("开始批量处理...")

    for subject in subjects_to_process:
        input_file = f"data/{chinese_to_english[subject]}.xlsx"
        output_file = f"gmm/{chinese_to_english[subject]}.xlsx"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        process_single_file(input_file, output_file, subject)

    print("\n批量处理完成!")