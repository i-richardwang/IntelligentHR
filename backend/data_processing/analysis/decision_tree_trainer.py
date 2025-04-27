import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from typing import List, Dict, Any, Tuple
import logging

from backend.data_processing.analysis.model_utils import (
    BaseModel,
    create_preprocessor,
    evaluate_model,
    get_feature_importance,
)


class DecisionTreeModel(BaseModel):
    """决策树模型类"""

    def __init__(self, problem_type: str):
        super().__init__(problem_type)
        self.numeric_preprocessor = "StandardScaler"
        self.categorical_preprocessor = "OneHotEncoder"
        self.logger = logging.getLogger(__name__)

    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: List[str],
        numerical_cols: List[str],
        param_grid: Dict[str, Any],
        n_trials: int,
    ) -> Tuple[Pipeline, Dict[str, Any], float, None]:
        """
        优化决策树模型参数

        Args:
            X_train: 训练特征
            y_train: 训练标签
            categorical_cols: 分类特征列名列表
            numerical_cols: 数值特征列名列表
            param_grid: 参数网格
            n_trials: 未使用，保留以兼容接口

        Returns:
            Tuple[Pipeline, Dict[str, Any], float, None]:
            最佳模型pipeline, 最佳参数, 最佳得分, None（决策树不使用trials）
        """
        self.logger.info("开始决策树模型参数优化")
        preprocessor = create_preprocessor(
            categorical_cols,
            numerical_cols,
            self.numeric_preprocessor,
            self.categorical_preprocessor,
        )

        if self.problem_type == "classification":
            dt = DecisionTreeClassifier(random_state=42)
            scoring = "roc_auc"
        else:
            dt = DecisionTreeRegressor(random_state=42)
            scoring = "neg_mean_squared_error"

        pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", dt)])

        grid_search = GridSearchCV(
            pipeline, param_grid, cv=5, scoring=scoring, n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        self.logger.info(f"决策树模型参数优化完成。最佳得分: {grid_search.best_score_}")
        return (
            grid_search.best_estimator_,
            grid_search.best_params_,
            grid_search.best_score_,
            None,  # 决策树使用网格搜索，没有 trial 的概念
        )

    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: List[str],
        numerical_cols: List[str],
        param_ranges: Dict[str, Any] = None,
        n_trials: int = 100,
        numeric_preprocessor: str = "StandardScaler",
        categorical_preprocessor: str = "OneHotEncoder",
    ) -> Dict[str, Any]:
        """
        训练决策树模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            categorical_cols: 分类特征列名列表
            numerical_cols: 数值特征列名列表
            param_ranges: 参数范围（对于决策树，这是参数网格）
            n_trials: 未使用，保留以兼容接口
            numeric_preprocessor: 数值特征预处理方法
            categorical_preprocessor: 分类特征预处理方法

        Returns:
            Dict[str, Any]: 包含训练结果的字典
        """
        self.logger.info("开始决策树模型训练")
        self.numeric_preprocessor = numeric_preprocessor
        self.categorical_preprocessor = categorical_preprocessor

        default_param_grid = {
            "classifier__max_depth": [2, 4, 6, 8, 10, None],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_leaf_nodes": [10, 20, 30, None],
        }

        # 如果提供了param_ranges，更新默认值
        if param_ranges:
            default_param_grid.update(param_ranges)

        param_grid = default_param_grid

        best_pipeline, best_params, cv_mean_score, _ = self.optimize(
            X_train, y_train, categorical_cols, numerical_cols, param_grid, n_trials
        )

        self.model = best_pipeline

        results = {
            "model": self.model,
            "feature_importance": self.get_feature_importance(),
            "cv_mean_score": cv_mean_score,
            "best_params": best_params,
        }

        # 对于回归问题，CV分数是负的MSE，我们需要取其绝对值
        if self.problem_type == "regression":
            results["cv_mean_score"] = abs(results["cv_mean_score"])

        self.logger.info("决策树模型训练完成")
        return results

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        评估决策树模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            Dict[str, Any]: 包含评估指标的字典
        """
        self.logger.info("开始决策树模型评估")
        return evaluate_model(self.model, X_test, y_test, self.problem_type)

    def get_feature_importance(self) -> pd.Series:
        """
        获取特征重要性

        Returns:
            pd.Series: 特征重要性
        """
        return get_feature_importance(
            self.model.named_steps["classifier"],
            self.model.named_steps["preprocessor"],
        )
