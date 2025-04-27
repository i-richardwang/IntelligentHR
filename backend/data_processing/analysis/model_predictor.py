import os
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

class ModelPredictor:
    def __init__(self, models_dir: str = "data/ml_models"):
        self.models_dir = models_dir
        self.model: Pipeline = None
        self.model_info: Dict[str, Any] = {}
        self.original_features: List[str] = []
        self.preprocessor: ColumnTransformer = None
        self.problem_type: str = None

    def load_model(self, model_filename: str, problem_type: str):
        model_path = os.path.join(self.models_dir, problem_type, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件 {model_path} 不存在。")

        loaded_model = joblib.load(model_path)
        if not isinstance(loaded_model, Pipeline):
            raise ValueError("加载的模型不是 Pipeline 类型。")

        self.model = loaded_model
        self.preprocessor = self.model.named_steps["preprocessor"]
        self.original_features = self.get_original_feature_names()
        self.problem_type = problem_type

        # 获取模型实例
        model_instance = self.get_model_instance()
        model_type = type(model_instance).__name__

        self.model_info = {
            "filename": model_filename,
            "type": model_type,
            "problem_type": problem_type,
            "features": self.original_features,
        }

    def get_model_instance(self) -> BaseEstimator:
        """获取模型实例，无论它被命名为什么"""
        for step_name, step_instance in self.model.named_steps.items():
            if step_name != "preprocessor":
                return step_instance
        raise ValueError("无法在管道中找到模型实例")

    def get_original_feature_names(self) -> List[str]:
        if not isinstance(self.preprocessor, ColumnTransformer):
            raise ValueError("预处理器不是 ColumnTransformer 类型。")

        original_features = []
        for name, transformer, columns in self.preprocessor.transformers_:
            if name != "remainder":
                if isinstance(columns, str):
                    original_features.append(columns)
                else:
                    original_features.extend(columns)
        return original_features

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        expected_features = set(self.original_features)
        input_features = set(data.columns)

        if not expected_features.issubset(input_features):
            missing_features = expected_features - input_features
            raise ValueError(f"输入数据缺少以下特征：{missing_features}")

        return data[self.original_features]

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model 方法。")

        preprocessed_data = self.preprocess_data(data)
        return self.model.predict(preprocessed_data)

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise ValueError("模型未加载，请先调用 load_model 方法。")

        if self.problem_type != "classification":
            raise ValueError("该方法仅适用于分类问题。")

        preprocessed_data = self.preprocess_data(data)
        return self.model.predict_proba(preprocessed_data)

    def get_model_info(self) -> Dict[str, Any]:
        return self.model_info

def list_available_models(
    models_dir: str = "data/ml_models", problem_type: str = "classification"
) -> List[str]:
    folder_path = os.path.join(models_dir, problem_type)
    return [f for f in os.listdir(folder_path) if f.endswith(".joblib")]