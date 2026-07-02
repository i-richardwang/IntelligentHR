import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    classification_report,
    confusion_matrix,
    mean_squared_error,
    r2_score,
)
from sklearn.utils.multiclass import unique_labels
from typing import List, Dict, Any, Tuple
import joblib
from datetime import datetime
import os
from abc import ABC, abstractmethod

from backend.data_processing.analysis.model_predictor import ModelPredictor


# 建模相关常量（内联于本模块，避免引入外部配置依赖）
# 分类问题要求目标变量恰好为 2 类（仅支持二分类）
MIN_CLASSES_FOR_CLASSIFICATION = 2
MAX_CLASSES_FOR_CLASSIFICATION = 2
# 类别不平衡阈值：少数类占多数类比例低于该值时给出提示
CLASS_IMBALANCE_THRESHOLD = 0.1


def validate_binary_classification_target(y: pd.Series, target_name: str) -> None:
    """
    校验目标变量是否满足二分类要求（恰好 2 个类别）。

    Args:
        y: 目标变量序列
        target_name: 目标列名

    Raises:
        ValueError: 当目标变量的取值种类不等于 2 时抛出（中文报错）
    """
    n_classes = pd.Series(y).nunique()
    if n_classes < MIN_CLASSES_FOR_CLASSIFICATION:
        raise ValueError(
            f"目标变量“{target_name}”只有 {n_classes} 种取值，"
            f"二分类要求目标变量恰好包含 {MAX_CLASSES_FOR_CLASSIFICATION} 个类别。"
        )
    if n_classes > MAX_CLASSES_FOR_CLASSIFICATION:
        unique_values = sorted(pd.Series(y).unique())
        raise ValueError(
            f"目标变量“{target_name}”包含 {n_classes} 种取值：{unique_values}。"
            f"当前工具仅支持二分类（恰好 {MAX_CLASSES_FOR_CLASSIFICATION} 个类别）。"
            "请先对数据进行预处理，将目标变量转换为二分类变量。"
        )


class BaseModel(ABC):
    """基础模型类，为所有模型提供通用接口"""

    def __init__(self, problem_type: str):
        self.problem_type = problem_type
        self.model = None
        self.preprocessor = None

    @abstractmethod
    def optimize(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        categorical_cols: List[str],
        numerical_cols: List[str],
        param_ranges: Dict[str, Any],
        n_trials: int,
    ) -> Tuple[Pipeline, Dict[str, Any], float, int]:
        """优化模型参数"""
        pass

    @abstractmethod
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
        """训练模型"""
        pass

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """评估模型性能"""
        return evaluate_model(self.model, X_test, y_test, self.problem_type)

    def get_feature_importance(self) -> pd.Series:
        """获取特征重要性"""
        return get_feature_importance(
            self.model.named_steps["classifier"],
            self.model.named_steps["preprocessor"],
        )


def prepare_data(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    test_size: float = 0.3,
    problem_type: str = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, List[str], List[str]]:
    """
    准备训练和测试数据

    Args:
        df: 原始数据框
        target_column: 目标变量列名
        feature_columns: 特征列名列表
        test_size: 测试集比例
        problem_type: 问题类型（'classification' 或 'regression'）。
            默认 None 时保持原有行为不变；当为 'classification' 时执行二分类校验。

    Returns:
        训练特征、测试特征、训练标签、测试标签、分类特征列表、数值特征列表

    Raises:
        ValueError: 当 problem_type 为 'classification' 且目标变量不是二分类时抛出
    """
    X = df[feature_columns]
    y = df[target_column]

    # 仅在明确指定为分类问题时校验二分类要求，避免影响其它调用方
    if problem_type == "classification":
        validate_binary_classification_target(y, target_column)

    if test_size > 0:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
    else:
        X_train, y_train = X, y
        X_test, y_test = pd.DataFrame(), pd.Series()

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    return X_train, X_test, y_train, y_test, categorical_cols, numerical_cols


def create_preprocessor(
    categorical_cols: List[str],
    numerical_cols: List[str],
    numeric_preprocessor: str = "StandardScaler",
    categorical_preprocessor: str = "OneHotEncoder",
) -> ColumnTransformer:
    """
    创建数据预处理器

    Args:
        categorical_cols: 分类特征列名列表
        numerical_cols: 数值特征列名列表
        numeric_preprocessor: 数值特征预处理方法
        categorical_preprocessor: 分类特征预处理方法

    Returns:
        ColumnTransformer 预处理器
    """
    numeric_transformer = (
        StandardScaler() if numeric_preprocessor == "StandardScaler" else "passthrough"
    )

    if categorical_preprocessor == "OneHotEncoder":
        categorical_transformer = OneHotEncoder(
            handle_unknown="ignore", drop="if_binary"
        )
    elif categorical_preprocessor == "OrdinalEncoder":
        categorical_transformer = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    else:  # 'passthrough'
        categorical_transformer = "passthrough"

    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )


def evaluate_model(
    model: Any, X_test: pd.DataFrame, y_test: pd.Series, problem_type: str
) -> Dict[str, Any]:
    """
    评估模型性能

    Args:
        model: 训练好的模型
        X_test: 测试特征
        y_test: 测试标签
        problem_type: 问题类型（'classification' 或 'regression'）

    Returns:
        包含评估指标的字典
    """
    y_test_pred = model.predict(X_test)

    if problem_type == "classification":
        y_test_pred_proba = model.predict_proba(X_test)[:, 1]
        # 使用 unique_labels 获取真实类别标签，其顺序与 confusion_matrix 的行/列一致，
        # 从而保证混淆矩阵坐标轴标签与矩阵内容对应无误。
        classes = list(unique_labels(y_test, y_test_pred))
        return {
            "test_roc_auc": roc_auc_score(y_test, y_test_pred_proba),
            "test_confusion_matrix": confusion_matrix(y_test, y_test_pred),
            "test_classification_report": classification_report(y_test, y_test_pred),
            "classes": classes,
        }
    else:  # regression
        mse = mean_squared_error(y_test, y_test_pred)
        r2 = r2_score(y_test, y_test_pred)
        return {
            "test_mse": mse,
            "test_r2": r2,
            "y_test": y_test,
            "y_pred": y_test_pred,
        }


def get_feature_importance(model: Any, preprocessor: ColumnTransformer) -> pd.Series:
    """
    获取特征重要性

    Args:
        model: 训练好的模型
        preprocessor: 数据预处理器

    Returns:
        特征重要性的 Series
    """
    feature_names = preprocessor.get_feature_names_out()
    if hasattr(model, "feature_importances_"):
        feature_importance = pd.Series(
            model.feature_importances_,
            index=feature_names,
        ).sort_values(ascending=False)
    elif hasattr(model, "coef_"):
        coefficients = model.coef_
        # 系数形状可能为 (1, n_features) 等多维，先展平为一维
        if hasattr(coefficients, "ndim") and coefficients.ndim > 1:
            coefficients = coefficients.ravel()
        else:
            coefficients = np.ravel(coefficients)
        # 系数与特征名长度不一致时按较短长度对齐（截断），避免构造 Series 时崩溃
        if len(coefficients) != len(feature_names):
            min_len = min(len(coefficients), len(feature_names))
            coefficients = coefficients[:min_len]
            feature_names = feature_names[:min_len]
        feature_importance = pd.Series(
            np.abs(coefficients),
            index=feature_names,
        ).sort_values(ascending=False)
    else:
        raise ValueError("模型不支持特征重要性计算")
    return feature_importance


def train_model(
    df: pd.DataFrame,
    target_column: str,
    feature_columns: List[str],
    model_type: str,
    problem_type: str,
    test_size: float = 0.3,
    param_ranges: Dict[str, Any] = None,
    n_trials: int = 100,
    numeric_preprocessor: str = "StandardScaler",
    categorical_preprocessor: str = "OneHotEncoder",
) -> Dict[str, Any]:
    """
    训练模型的主函数

    Args:
        df: 原始数据框
        target_column: 目标变量列名
        feature_columns: 特征列名列表
        model_type: 模型类型
        problem_type: 问题类型（'classification' 或 'regression'）
        test_size: 测试集比例
        param_ranges: 参数范围
        n_trials: 优化尝试次数
        numeric_preprocessor: 数值特征预处理方法
        categorical_preprocessor: 分类特征预处理方法

    Returns:
        包含训练结果的字典
    """
    X_train, X_test, y_train, y_test, categorical_cols, numerical_cols = prepare_data(
        df, target_column, feature_columns, test_size, problem_type
    )

    model_class = get_model_class(model_type)

    if param_ranges:
        param_ranges = filter_valid_params(param_ranges, get_valid_params(model_type))

    model = model_class(problem_type)

    results = model.train(
        X_train,
        y_train,
        categorical_cols,
        numerical_cols,
        param_ranges,
        n_trials,
        numeric_preprocessor=numeric_preprocessor,
        categorical_preprocessor=categorical_preprocessor,
    )

    if test_size > 0:
        test_metrics = model.evaluate(X_test, y_test)
        results.update(test_metrics)
    else:
        results.update(
            {
                "test_roc_auc": None,
                "test_mse": None,
                "test_r2": None,
                "test_confusion_matrix": None,
                "test_classification_report": None,
            }
        )

    return results


def get_model_class(model_type: str):
    """根据模型类型获取对应的模型类"""
    from backend.data_processing.analysis.random_forest_trainer import RandomForestModel
    from backend.data_processing.analysis.decision_tree_trainer import DecisionTreeModel
    from backend.data_processing.analysis.xgboost_trainer import XGBoostModel
    from backend.data_processing.analysis.linear_regression_trainer import (
        LinearRegressionModel,
    )

    model_classes = {
        "随机森林": RandomForestModel,
        "决策树": DecisionTreeModel,
        "XGBoost": XGBoostModel,
        "线性回归": LinearRegressionModel,
    }
    return model_classes.get(model_type)


def save_model(
    model: Any,
    model_type: str,
    problem_type: str,
    timestamp: datetime,
    save_path: str = "data/ml_models",
) -> str:
    """
    保存模型

    Args:
        model: 训练好的模型
        model_type: 模型类型
        problem_type: 问题类型
        timestamp: 时间戳
        save_path: 保存路径

    Returns:
        保存的文件路径
    """
    problem_folder = (
        "classification" if problem_type == "classification" else "regression"
    )
    full_save_path = os.path.join(save_path, problem_folder)
    os.makedirs(full_save_path, exist_ok=True)

    file_name = (
        f"{model_type}_{problem_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}.joblib"
    )
    file_path = os.path.join(full_save_path, file_name)
    joblib.dump(model, file_path)
    return file_path


def add_model_record(
    model_records: pd.DataFrame,
    model_type: str,
    problem_type: str,
    model_results: Dict[str, Any],
    model_id: str,
) -> pd.DataFrame:
    """
    添加模型记录

    Args:
        model_records: 现有的模型记录
        model_type: 模型类型
        problem_type: 问题类型
        model_results: 模型结果
        model_id: 模型唯一标识（对应 st.session_state.trained_models 的键，
            用于后续按记录行精确取回并保存对应的模型对象）

    Returns:
        更新后的模型记录
    """
    new_record = {
        "模型ID": model_id,
        "模型类型": model_type,
        "问题类型": "分类" if problem_type == "classification" else "回归",
        "训练时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "交叉验证分数": model_results["cv_mean_score"],
        "测试集分数": (
            model_results["test_roc_auc"]
            if problem_type == "classification"
            else model_results["test_mse"]
        ),
        "参数": str(model_results.get("best_params", "N/A")),
        "最佳轮次": model_results.get("best_trial", "N/A"),
    }

    return pd.concat([model_records, pd.DataFrame([new_record])], ignore_index=True)


def filter_valid_params(
    params: Dict[str, Any], valid_params: List[str]
) -> Dict[str, Any]:
    """过滤有效参数"""
    return {k: v for k, v in params.items() if k in valid_params}


def get_valid_params(model_type: str) -> List[str]:
    """获取模型的有效参数列表"""
    valid_params = {
        "随机森林": [
            "n_estimators",
            "max_depth",
            "min_samples_split",
            "min_samples_leaf",
            "max_features",
        ],
        "决策树": [
            "classifier__max_depth",
            "classifier__min_samples_split",
            "classifier__min_samples_leaf",
            "classifier__max_leaf_nodes",
        ],
        "XGBoost": [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "min_child_weight",
            "reg_alpha",
            "reg_lambda",
        ],
        "线性回归": [],
    }
    return valid_params.get(model_type, [])


def initialize_session_state() -> Dict[str, Any]:
    """初始化会话状态"""
    return {
        "df": None,
        "model_results": None,
        "target_column": None,
        "feature_columns": None,
        "model_type": "随机森林",
        "problem_type": "classification",
        "rf_param_grid": {
            "n_estimators": (10, 200),
            "max_depth": (5, 30),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 20),
            "max_features": ["sqrt", "log2"],
        },
        "dt_param_grid": {
            "classifier__max_depth": [2, 4, 5, 6, 7, None],
            "classifier__min_samples_split": [2, 3, 4, 5, 8],
            "classifier__min_samples_leaf": [2, 5, 10, 15, 20, 25],
            "classifier__max_leaf_nodes": [10, 20, 25, 30, 35, 40, 45, None],
        },
        "xgb_param_ranges": {
            "n_estimators": (50, 500),
            "max_depth": (3, 10),
            "learning_rate": (0.01, 1.0),
            "subsample": (0.5, 1.0),
            "colsample_bytree": (0.5, 1.0),
            "min_child_weight": (1, 10),
            "reg_alpha": (0, 10),
            "reg_lambda": (0, 10),
        },
        "custom_param_ranges": None,
        "model_records": pd.DataFrame(
            columns=[
                "模型ID",
                "模型类型",
                "问题类型",
                "训练时间",
                "参数",
                "交叉验证分数",
                "测试集分数",
                "最佳轮次",
            ]
        ),
        # 存储每次训练得到的模型对象：{模型ID: pipeline}，用于按记录行精确保存
        "trained_models": {},
        # 目标/问题类型校验是否通过的标志，用于在校验未通过时禁用训练按钮
        "problem_valid": True,
        "rf_n_trials": 100,
        "xgb_n_trials": 200,
        "predictor": ModelPredictor(),
        "uploaded_data": None,
        "predictions": None,
        "probabilities": None,
        "data_validated": False,
        "mode": "train",
        "do_model_interpretation": False,
        "numeric_preprocessor": "StandardScaler",
        "categorical_preprocessor": "OneHotEncoder",
    }
