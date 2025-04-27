import shap
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Any, Dict, List, Tuple


def calculate_shap_values(
    model: Any,
    X: pd.DataFrame,
    preprocessor: Any,
    feature_names: List[str],
    problem_type: str,
) -> Dict[str, Any]:
    """
    计算SHAP值并生成SHAP摘要图。

    Args:
        model: 训练好的模型
        X: 原始特征数据
        preprocessor: 数据预处理器
        feature_names: 原始特征名列表
        problem_type: 问题类型 ("classification" 或 "regression")

    Returns:
        包含SHAP值和图表数据的字典
    """
    # 预处理数据
    X_processed = preprocessor.transform(X)

    # 获取预处理后的特征名称
    if hasattr(preprocessor, "get_feature_names_out"):
        processed_feature_names = preprocessor.get_feature_names_out()
    else:
        processed_feature_names = np.array(feature_names)

    # 创建SHAP解释器
    if problem_type == "classification":
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_processed)
        if isinstance(shap_values, list):
            shap_values = shap_values[
                1
            ]  # For binary classification, we use the positive class
        elif shap_values.ndim == 3:
            shap_values = shap_values[
                :, :, 1
            ]  # For binary classification with 3D output
    else:
        if hasattr(model, "coef_"):  # Linear models
            explainer = shap.LinearExplainer(model, X_processed)
            shap_values = explainer.shap_values(X_processed)
        else:  # Tree-based models
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_processed)

    # 确保shap_values是二维的
    if shap_values.ndim != 2:
        raise ValueError(f"Unexpected SHAP values shape: {shap_values.shape}")

    # 计算特征重要性
    feature_importance = np.abs(shap_values).mean(0)
    feature_importance = pd.Series(
        feature_importance, index=processed_feature_names
    ).sort_values(ascending=False)

    # 创建SHAP摘要图数据
    summary_data = []
    for feature in feature_importance.index:
        feature_shap = shap_values[:, list(processed_feature_names).index(feature)]
        summary_data.append(
            {
                "feature": feature,
                "importance": feature_importance[feature],
                "shap_values": feature_shap.tolist(),
            }
        )

    return {
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "summary_data": summary_data,
        "processed_feature_names": processed_feature_names,
        "X_processed": X_processed,
    }
