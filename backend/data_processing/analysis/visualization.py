import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def create_confusion_matrix_plot(cm: np.ndarray) -> go.Figure:
    """
    创建混淆矩阵的热力图。

    Args:
        cm (np.ndarray): 混淆矩阵

    Returns:
        go.Figure: Plotly图形对象
    """
    cm_sum = np.sum(cm)
    cm_percentages = cm / cm_sum * 100

    fig = go.Figure(
        data=go.Heatmap(
            z=cm_percentages,
            x=["预测: 0", "预测: 1"],
            y=["实际: 0", "实际: 1"],
            hoverongaps=False,
            colorscale="Blues",
            text=[
                [f"{v:.1f}%<br>({cm[i][j]})" for j, v in enumerate(row)]
                for i, row in enumerate(cm_percentages)
            ],
            texttemplate="%{text}",
            textfont={"size": 14},
        )
    )
    fig.update_layout(
        title="混淆矩阵",
        xaxis_title="预测类别",
        yaxis_title="实际类别",
        width=400,
        height=400,
        margin=dict(t=40),
    )
    return fig


def create_residual_plot(y_test: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    """
    创建残差图。

    Args:
        y_test (np.ndarray): 测试集实际值
        y_pred (np.ndarray): 预测值

    Returns:
        go.Figure: Plotly图形对象
    """
    residuals = y_test - y_pred

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers"))
    fig.update_layout(
        title="残差图", xaxis_title="预测值", yaxis_title="残差", width=600, height=400
    )
    return fig


def create_feature_importance_plot(feature_importance: pd.Series) -> go.Figure:
    """
    创建特征重要性条形图。

    Args:
        feature_importance (pd.Series): 特征重要性序列

    Returns:
        go.Figure: Plotly图形对象
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=feature_importance.values, y=feature_importance.index, orientation="h"
            )
        ]
    )
    fig.update_layout(
        title="特征重要性",
        xaxis_title="重要性得分",
        yaxis_title="特征",
        height=max(500, len(feature_importance) * 25),
        width=600,
        margin=dict(t=40),
    )
    return fig


def create_prediction_distribution_plot(
    predictions: np.ndarray, problem_type: str
) -> go.Figure:
    """
    创建预测分布直方图。

    Args:
        predictions (np.ndarray): 预测结果
        problem_type (str): 问题类型（"classification" 或 "regression"）

    Returns:
        go.Figure: Plotly图形对象
    """
    fig = go.Figure(data=[go.Histogram(x=predictions)])
    fig.update_layout(
        title="预测分布",
        xaxis_title="预测类别" if problem_type == "classification" else "预测值",
        yaxis_title="数量",
        height=400,
        margin=dict(t=40),
    )
    return fig


def create_shap_importance_plot(
    feature_importance: pd.Series, max_display: int = 20
) -> go.Figure:
    """
    创建SHAP特征重要性图。

    Args:
        feature_importance (pd.Series): 包含特征重要性的pandas Series
        max_display (int): 显示的最大特征数

    Returns:
        go.Figure: Plotly图形对象
    """
    top_features = feature_importance.nlargest(max_display)

    fig = go.Figure(
        go.Bar(x=top_features.values, y=top_features.index, orientation="h")
    )
    fig.update_layout(
        title="SHAP特征重要性",
        xaxis_title="平均|SHAP值|（特征重要性）",
        yaxis_title="特征",
        height=max(500, len(top_features) * 25),
        width=800,
        yaxis=dict(autorange="reversed"),
    )
    return fig


def create_shap_dependence_plot(
    shap_values: np.ndarray,
    features: np.ndarray,
    feature_names: np.ndarray,
    selected_feature: str,
) -> go.Figure:
    """
    为选定的特征创建SHAP依赖图。

    Args:
        shap_values (np.ndarray): SHAP值数组
        features (np.ndarray): 预处理后的特征数据
        feature_names (np.ndarray): 预处理后的特征名称数组
        selected_feature (str): 选定的特征名称

    Returns:
        go.Figure: Plotly图形对象
    """
    feature_index = np.where(feature_names == selected_feature)[0][0]
    feature_value = features[:, feature_index]
    shap_value = shap_values[:, feature_index]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=feature_value,
            y=shap_value,
            mode="markers",
            marker=dict(
                size=8,
                color=feature_value,
                colorscale="RdBu",
                showscale=True,
                colorbar=dict(title=selected_feature),
            ),
            text=[
                f"{selected_feature}: {val}<br>SHAP value: {shap:.2f}"
                for val, shap in zip(feature_value, shap_value)
            ],
            hoverinfo="text",
        )
    )

    fig.update_layout(
        title=f"SHAP Dependence Plot for {selected_feature}",
        xaxis_title=selected_feature,
        yaxis_title="SHAP value",
        width=800,
        height=600,
    )
    return fig
