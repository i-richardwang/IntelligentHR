import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
import optuna
from optuna.samplers import TPESampler
from typing import List, Dict, Any, Tuple
import logging

from backend_demo.data_processing.analysis.model_utils import (
    BaseModel,
    create_preprocessor,
    evaluate_model,
    get_feature_importance,
)


class RandomForestModel(BaseModel):
    """随机森林模型类"""

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
        param_ranges: Dict[str, Any],
        n_trials: int,
    ) -> Tuple[Pipeline, Dict[str, Any], float, int]:
        """
        优化随机森林模型参数

        Args:
            X_train: 训练特征
            y_train: 训练标签
            categorical_cols: 分类特征列名列表
            numerical_cols: 数值特征列名列表
            param_ranges: 参数范围
            n_trials: 优化尝试次数

        Returns:
            Tuple[Pipeline, Dict[str, Any], float, int]:
            最佳模型pipeline, 最佳参数, 最佳得分, 最佳试验次数
        """
        self.logger.info("开始随机森林模型参数优化")
        preprocessor = create_preprocessor(
            categorical_cols,
            numerical_cols,
            self.numeric_preprocessor,
            self.categorical_preprocessor,
        )

        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int(
                    "n_estimators",
                    param_ranges["n_estimators"][0],
                    param_ranges["n_estimators"][1],
                ),
                "max_depth": trial.suggest_int(
                    "max_depth",
                    param_ranges["max_depth"][0],
                    param_ranges["max_depth"][1],
                ),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split",
                    param_ranges["min_samples_split"][0],
                    param_ranges["min_samples_split"][1],
                ),
                "min_samples_leaf": trial.suggest_int(
                    "min_samples_leaf",
                    param_ranges["min_samples_leaf"][0],
                    param_ranges["min_samples_leaf"][1],
                ),
                "max_features": trial.suggest_categorical(
                    "max_features", param_ranges["max_features"]
                ),
            }

            if self.problem_type == "classification":
                rf = RandomForestClassifier(**params, random_state=42)
                scoring = "roc_auc"
            else:
                rf = RandomForestRegressor(**params, random_state=42)
                scoring = "neg_mean_squared_error"

            pipeline = Pipeline(
                steps=[("preprocessor", preprocessor), ("classifier", rf)]
            )
            scores = cross_val_score(
                pipeline, X_train, y_train, cv=5, scoring=scoring, n_jobs=-1
            )
            return np.mean(scores)

        study = optuna.create_study(direction="maximize", sampler=TPESampler())
        study.optimize(objective, n_trials=n_trials, n_jobs=-1)

        best_params = study.best_params
        if self.problem_type == "classification":
            best_rf = RandomForestClassifier(**best_params, random_state=42)
        else:
            best_rf = RandomForestRegressor(**best_params, random_state=42)

        best_pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), ("classifier", best_rf)]
        )
        best_pipeline.fit(X_train, y_train)
        best_score = study.best_value
        best_trial = study.best_trial.number + 1

        self.logger.info(f"随机森林模型参数优化完成。最佳得分: {best_score}")
        return best_pipeline, best_params, best_score, best_trial

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
        训练随机森林模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            categorical_cols: 分类特征列名列表
            numerical_cols: 数值特征列名列表
            param_ranges: 参数范围
            n_trials: 优化尝试次数
            numeric_preprocessor: 数值特征预处理方法
            categorical_preprocessor: 分类特征预处理方法

        Returns:
            Dict[str, Any]: 包含训练结果的字典
        """
        self.logger.info("开始随机森林模型训练")
        self.numeric_preprocessor = numeric_preprocessor
        self.categorical_preprocessor = categorical_preprocessor

        default_param_ranges = {
            "n_estimators": (10, 200),
            "max_depth": (5, 30),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 20),
            "max_features": ["sqrt", "log2"],
        }
        param_ranges = param_ranges or default_param_ranges

        best_pipeline, best_params, cv_mean_score, best_trial = self.optimize(
            X_train, y_train, categorical_cols, numerical_cols, param_ranges, n_trials
        )

        self.model = best_pipeline

        results = {
            "model": self.model,
            "feature_importance": self.get_feature_importance(),
            "cv_mean_score": cv_mean_score,
            "best_params": best_params,
            "best_trial": best_trial,
        }

        if self.problem_type == "regression":
            results["cv_mean_score"] = abs(results["cv_mean_score"])

        self.logger.info("随机森林模型训练完成")
        return results

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        评估随机森林模型性能

        Args:
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            Dict[str, Any]: 包含评估指标的字典
        """
        self.logger.info("开始随机森林模型评估")
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
