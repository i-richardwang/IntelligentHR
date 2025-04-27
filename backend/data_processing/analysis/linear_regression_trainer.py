import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from typing import List, Dict, Any, Tuple
import logging

from backend_demo.data_processing.analysis.model_utils import (
    BaseModel,
    create_preprocessor,
    evaluate_model,
    get_feature_importance,
)


class LinearRegressionModel(BaseModel):
    """线性回归模型类"""

    def __init__(self, problem_type: str):
        super().__init__(problem_type)
        self.logger = logging.getLogger(__name__)
        self.numeric_preprocessor = "StandardScaler"
        self.categorical_preprocessor = "OneHotEncoder"

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: List[str],
        numerical_cols: List[str],
        param_ranges: Dict[str, Any],
        n_trials: int,
    ) -> Tuple[Pipeline, Dict[str, Any], float, int]:
        """
        线性回归模型不需要参数优化，此方法仅为满足接口要求

        Returns:
            Tuple[None, None, None, None]: 占位返回值
        """
        return None, None, None, None

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: List[str],
        numerical_cols: List[str],
        param_ranges: Dict[str, Any] = None,
        n_trials: int = None,
        numeric_preprocessor: str = "StandardScaler",
        categorical_preprocessor: str = "OneHotEncoder",
    ) -> Dict[str, Any]:
        """
        训练线性回归模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            categorical_cols: 分类特征列名列表
            numerical_cols: 数值特征列名列表
            param_ranges: 未使用
            n_trials: 未使用
            numeric_preprocessor: 数值特征预处理方法
            categorical_preprocessor: 分类特征预处理方法

        Returns:
            Dict[str, Any]: 包含训练结果的字典
        """
        self.logger.info("开始线性回归模型训练")
        self.numeric_preprocessor = numeric_preprocessor
        self.categorical_preprocessor = categorical_preprocessor

        preprocessor = create_preprocessor(
            categorical_cols,
            numerical_cols,
            self.numeric_preprocessor,
            self.categorical_preprocessor,
        )
        model = LinearRegression()
        self.model = Pipeline(
            steps=[("preprocessor", preprocessor), ("regressor", model)]
        )

        # 在全部训练数据上拟合模型
        self.model.fit(X_train, y_train)

        # 计算训练集 MSE 和 R²
        y_train_pred = self.model.predict(X_train)
        train_mse = np.mean((y_train - y_train_pred) ** 2)
        train_r2 = self.model.score(X_train, y_train)

        self.logger.info(f"训练 MSE: {train_mse}, R²: {train_r2}")

        results = {
            "model": self.model,
            "feature_importance": self.get_feature_importance(),
            "train_mse": train_mse,
            "train_r2": train_r2,
        }

        # 添加系数和截距
        linear_model = self.model.named_steps["regressor"]
        results["coefficients"] = pd.Series(
            linear_model.coef_,
            index=self.model.named_steps["preprocessor"].get_feature_names_out(),
        )
        results["intercept"] = linear_model.intercept_

        # 为了保持与其他模型一致的接口，我们将 train_mse 也赋值给 cv_mean_score
        results["cv_mean_score"] = train_mse

        return results

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        评估线性回归模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            Dict[str, Any]: 包含评估指标的字典
        """
        self.logger.info("开始模型评估")
        return evaluate_model(self.model, X_test, y_test, self.problem_type)

    def get_feature_importance(self) -> pd.Series:
        """
        获取特征重要性（线性回归中使用系数的绝对值）

        Returns:
            pd.Series: 特征重要性
        """
        linear_model = self.model.named_steps["regressor"]
        feature_names = self.model.named_steps["preprocessor"].get_feature_names_out()
        importance = np.abs(linear_model.coef_)
        return pd.Series(importance, index=feature_names).sort_values(ascending=False)
