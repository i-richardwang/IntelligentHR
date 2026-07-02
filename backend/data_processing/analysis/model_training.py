import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Any

from backend.data_processing.analysis.model_utils import (
    train_model,
    save_model,
    add_model_record,
    evaluate_model,
    get_feature_importance,
    MIN_CLASSES_FOR_CLASSIFICATION,
    MAX_CLASSES_FOR_CLASSIFICATION,
    CLASS_IMBALANCE_THRESHOLD,
)
from backend.data_processing.analysis.shap_analysis import calculate_shap_values
from backend.data_processing.analysis.visualization import (
    create_confusion_matrix_plot,
    create_residual_plot,
    create_feature_importance_plot,
    create_prediction_distribution_plot,
    create_shap_importance_plot,
    create_shap_dependence_plot,
)
from backend.data_processing.analysis.model_predictor import ModelPredictor
from backend.data_processing.analysis.ml_components import (
    display_info_message,
    display_data_split_settings,
    display_random_forest_settings,
    display_decision_tree_settings,
    display_xgboost_settings,
    display_linear_regression_settings,
    display_model_selection,
    display_preprocessing_settings,
)
from backend.data_processing.analysis.ml_explanations import (
    CONFUSION_MATRIX_EXPLANATION,
    CLASSIFICATION_REPORT_EXPLANATION,
    REGRESSION_METRICS_EXPLANATION,
    ML_TOOL_INFO,
    FEATURE_IMPORTANCE_EXPLANATION,
    SHAP_FEATURE_IMPORTANCE_EXPLANATION,
    SHAP_DEPENDENCE_PLOT_EXPLANATION,
    LINEAR_REGRESSION_FEATURE_IMPORTANCE_EXPLANATION,
    XGBOOST_LABEL_ENCODING_EXPLANATION,
)


def display_data_upload_and_preview(for_prediction: bool = False) -> None:
    """
    显示数据上传和预览界面

    Args:
        for_prediction: 是否用于预测
    """
    st.markdown("## 数据上传与预览")
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "上传CSV或Excel文件", type=["csv", "xlsx", "xls"]
        )

        if uploaded_file is not None:
            try:
                data = (
                    pd.read_csv(uploaded_file)
                    if uploaded_file.name.endswith(".csv")
                    else pd.read_excel(uploaded_file)
                )
                st.session_state.data_validated = False

                if for_prediction:
                    handle_prediction_data_upload(data)
                else:
                    handle_training_data_upload(data)

                st.write(f"数据集包含 {len(data)} 行和 {len(data.columns)} 列")
                st.write(data.head())

                with st.expander("查看数据类型信息", expanded=False):
                    st.write(data.dtypes)

            except Exception as e:
                st.error(f"处理文件时出错：{str(e)}")


def handle_prediction_data_upload(data: pd.DataFrame) -> None:
    """
    处理预测数据上传

    Args:
        data: 上传的数据
    """
    if st.session_state.predictor.model is not None:
        model_features = set(st.session_state.predictor.original_features)
        data_features = set(data.columns)
        missing_features = model_features - data_features
        extra_features = data_features - model_features

        if missing_features:
            st.warning(f"⚠️ 上传的数据缺少以下特征：{', '.join(missing_features)}")
            return

        st.session_state.uploaded_data = data
        st.session_state.data_validated = True
        st.success("✅ 数据上传成功！")

        if extra_features:
            st.info(f"ℹ️ 额外的特征: {', '.join(extra_features)}")
    else:
        st.warning("⚠️ 请先选择一个模型，然后再上传数据。")


def handle_training_data_upload(data: pd.DataFrame) -> None:
    """
    处理训练数据上传

    Args:
        data: 上传的数据
    """
    st.session_state.df = data
    st.session_state.data_validated = True
    st.success("文件上传成功！")


def display_column_selection() -> None:
    """显示列选择界面"""
    if st.session_state.df is not None:
        st.markdown("## 变量选择")
        with st.container(border=True):
            st.session_state.target_column = st.selectbox(
                "选择目标变量",
                options=st.session_state.df.columns,
                key="target_column_select",
            )
            with st.expander("选择特征变量", expanded=False):
                st.session_state.feature_columns = st.multiselect(
                    "选择特征变量",
                    options=[
                        col
                        for col in st.session_state.df.columns
                        if col != st.session_state.target_column
                    ],
                    default=[
                        col
                        for col in st.session_state.df.columns
                        if col != st.session_state.target_column
                    ],
                    key="feature_columns_select",
                )

            validate_problem_type()

            # 删除包含null值的行
            if st.button("确认特征和目标变量"):
                original_row_count = len(st.session_state.df)
                st.session_state.df = st.session_state.df.dropna(
                    subset=[st.session_state.target_column]
                    + st.session_state.feature_columns
                )
                new_row_count = len(st.session_state.df)
                removed_rows = original_row_count - new_row_count
                st.success(
                    f"已删除 {removed_rows} 行包含空值的数据。剩余 {new_row_count} 行数据。"
                )


def validate_problem_type() -> None:
    """
    验证问题类型与目标变量是否匹配，并将校验结果记录到
    st.session_state.problem_valid（用于禁用训练按钮）。

    分类：要求目标变量恰好为 2 类，并检测类别不平衡；
    回归：要求目标变量为数值类型，并展示取值范围/均值/标准差。
    """
    target_col = st.session_state.target_column
    # 目标列尚未选择时不做校验，保持默认通过状态
    if target_col is None or target_col not in st.session_state.df.columns:
        st.session_state.problem_valid = True
        return

    target_data = st.session_state.df[target_col]
    st.session_state.problem_valid = True

    if st.session_state.problem_type == "classification":
        n_classes = target_data.nunique()

        if n_classes < MIN_CLASSES_FOR_CLASSIFICATION:
            st.error(
                f"❌ 目标变量“{target_col}”只有 {n_classes} 种取值，"
                f"二分类要求目标变量恰好包含 {MAX_CLASSES_FOR_CLASSIFICATION} 个类别。"
            )
            st.session_state.problem_valid = False
        elif n_classes > MAX_CLASSES_FOR_CLASSIFICATION:
            unique_values = sorted(target_data.unique())[:10]  # 仅展示前 10 个取值
            st.error(
                f"❌ 目标变量“{target_col}”包含 {n_classes} 种取值"
                f"（例如：{unique_values}{'...' if n_classes > 10 else ''}）。\n\n"
                f"**当前工具仅支持二分类（恰好 {MAX_CLASSES_FOR_CLASSIFICATION} 个类别）。**\n\n"
                "💡 建议：\n"
                "- 将目标变量改造为二分类（如“是/否”“0/1”“True/False”）\n"
                "- 将多个类别合并为两组\n"
                "- 或改用支持多分类的工具"
            )
            st.session_state.problem_valid = False
        else:
            # 恰好 2 类，展示成功信息
            class_values = sorted(target_data.unique())
            class_counts = target_data.value_counts()
            st.success(
                f"✅ 已检测到二分类目标：**{class_values[0]}**（{class_counts[class_values[0]]}）"
                f" vs **{class_values[1]}**（{class_counts[class_values[1]]}）"
            )

            # 类别不平衡检测：少数类占比过低时提示
            class_ratio = class_counts.min() / class_counts.max()
            if class_ratio < CLASS_IMBALANCE_THRESHOLD:
                st.warning(
                    f"⚠️ 检测到类别不平衡（少数类占比：{class_ratio:.2%}）。"
                    "可考虑使用 SMOTE 过采样、类别权重（class weights）或分层抽样等方法。"
                )

        # 对疑似连续数值的目标变量额外提示
        if target_data.dtype in ["int64", "float64"] and n_classes > 10:
            st.warning(
                "⚠️ 目标变量看起来像是连续数值。您可能需要选择回归问题而不是分类问题。"
            )

    else:  # regression
        if target_data.dtype not in ["int64", "float64"]:
            st.error(
                f"❌ 目标变量“{target_col}”不是数值类型（当前类型：{target_data.dtype}）。"
                "回归问题需要数值类型的目标变量。"
            )
            st.session_state.problem_valid = False
        else:
            # 展示回归目标变量的统计信息
            st.success(
                f"✅ 回归目标变量校验通过："
                f"取值范围 [{target_data.min():.2f}, {target_data.max():.2f}]，"
                f"均值：{target_data.mean():.2f}，标准差：{target_data.std():.2f}"
            )


def display_model_training_and_advanced_settings() -> None:
    """显示模型训练和高级设置界面"""
    if (
        st.session_state.df is not None
        and st.session_state.target_column
        and st.session_state.feature_columns
    ):
        st.markdown("## 模型训练")
        with st.container(border=True):
            display_data_split_settings()
            display_preprocessing_settings()
            display_model_parameters_settings()

            # 目标/问题类型校验未通过时禁用训练按钮（默认 True，不影响正常流程）
            training_disabled = not st.session_state.get("problem_valid", True)
            if training_disabled:
                st.warning("请先解决目标变量/问题类型的校验问题，然后再开始训练。")

            if st.button("开始训练模型", disabled=training_disabled):
                train_and_evaluate_model()


def display_model_parameters_settings() -> None:
    """显示模型参数设置"""
    with st.expander("模型参数设置", expanded=False):
        if st.session_state.model_type == "随机森林":
            display_random_forest_settings()
        elif st.session_state.model_type == "决策树":
            display_decision_tree_settings()
        elif st.session_state.model_type == "XGBoost":
            display_xgboost_settings()
        elif st.session_state.model_type == "线性回归":
            display_linear_regression_settings()


def train_and_evaluate_model() -> None:
    """训练和评估模型"""
    with st.spinner("正在训练模型，请稍候..."):
        try:
            param_ranges, n_trials = get_model_params()

            st.session_state.model_results = train_model(
                st.session_state.df,
                st.session_state.target_column,
                st.session_state.feature_columns,
                st.session_state.model_type,
                st.session_state.problem_type,
                st.session_state.test_size,
                param_ranges=param_ranges,
                n_trials=n_trials,
                numeric_preprocessor=st.session_state.numeric_preprocessor,
                categorical_preprocessor=st.session_state.categorical_preprocessor,
            )

            # 生成本次训练的唯一模型ID（模型类型 + 精确到微秒的时间戳，保证唯一且可对应模型对象）
            model_id = (
                f"{st.session_state.model_type}_"
                f"{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
            )

            # 将本次训练得到的模型对象按 model_id 存入 trained_models，供后续按行精确保存
            if "trained_models" not in st.session_state:
                st.session_state.trained_models = {}
            st.session_state.trained_models[model_id] = (
                st.session_state.model_results["model"]
            )

            st.session_state.model_records = add_model_record(
                st.session_state.model_records,
                st.session_state.model_type,
                st.session_state.problem_type,
                st.session_state.model_results,
                model_id,
            )
            display_training_success_message()

            if st.session_state.do_model_interpretation:
                with st.spinner("正在计算模型解释..."):
                    calculate_and_store_shap_values()
                    st.success("模型解释计算完成！")

        except Exception as e:
            st.error(f"模型训练过程中出错：{str(e)}")


def display_training_success_message() -> None:
    """显示训练成功消息"""
    success_message = "模型训练完成！"
    if "best_trial" in st.session_state.model_results:
        success_message += (
            f" 最佳参数在第 {st.session_state.model_results['best_trial']} 轮获得。"
        )
    st.success(success_message)


def get_model_params() -> tuple:
    """获取模型参数"""
    if st.session_state.model_type == "随机森林":
        return st.session_state.rf_param_grid, st.session_state.rf_n_trials
    elif st.session_state.model_type == "决策树":
        return st.session_state.dt_param_grid, None
    elif st.session_state.model_type == "XGBoost":
        return st.session_state.xgb_param_ranges, st.session_state.xgb_n_trials
    elif st.session_state.model_type == "线性回归":
        return None, None
    else:
        raise ValueError(f"不支持的模型类型: {st.session_state.model_type}")


def display_model_records() -> None:
    """显示模型记录"""
    if not st.session_state.model_records.empty:
        st.markdown("## 模型记录")
        with st.container(border=True):
            columns_order = [
                "模型ID",
                "模型类型",
                "问题类型",
                "交叉验证分数",
                "测试集分数",
                "最佳模型",
                "保存",
                "训练时间",
                "参数",
            ]
            temp_df = st.session_state.model_records.reindex(columns=columns_order)
            temp_df["保存"] = False
            temp_df["最佳模型"] = False

            best_model_index = select_best_model(temp_df)
            temp_df.loc[best_model_index, "最佳模型"] = True

            edited_df = display_model_record_table(temp_df, columns_order)

            save_selected_models(edited_df)


def select_best_model(df: pd.DataFrame) -> int:
    """
    选择最佳模型

    Args:
        df: 模型记录数据框

    Returns:
        最佳模型的索引
    """
    return (
        df["交叉验证分数"].idxmax()
        if st.session_state.problem_type == "classification"
        else df["交叉验证分数"].idxmin()
    )


def display_model_record_table(df: pd.DataFrame, columns_order: list) -> pd.DataFrame:
    """
    显示模型记录表格

    Args:
        df: 模型记录数据框
        columns_order: 列顺序

    Returns:
        编辑后的数据框
    """
    return st.data_editor(
        df,
        column_config={
            "保存": st.column_config.CheckboxColumn(
                "保存",
                help="选择要保存的模型",
                default=False,
            ),
            "最佳模型": st.column_config.CheckboxColumn(
                "最佳模型",
                help="表现最好的模型",
                default=False,
            ),
            "交叉验证分数": st.column_config.NumberColumn(
                "交叉验证分数",
                format="%.4f",
                help="对于线性回归模型不使用交叉验证，此值为训练集 MSE。",
            ),
            "测试集分数": st.column_config.NumberColumn(
                "测试集分数",
                format="%.4f",
            ),
        },
        disabled=[
            "模型ID",
            "模型类型",
            "问题类型",
            "训练时间",
            "参数",
            "交叉验证分数",
            "测试集分数",
            "最佳模型",
        ],
        hide_index=True,
        column_order=columns_order,
        use_container_width=True,
    )


def save_selected_models(edited_df: pd.DataFrame) -> None:
    """
    保存选中的模型

    Args:
        edited_df: 编辑后的模型记录数据框
    """
    models_to_save = edited_df[edited_df["保存"]]
    if not models_to_save.empty:
        trained_models = st.session_state.get("trained_models", {})
        for _, row in models_to_save.iterrows():
            model_id = row["模型ID"]
            model_type = row["模型类型"]
            problem_type = (
                "classification" if row["问题类型"] == "分类" else "regression"
            )
            timestamp = datetime.strptime(row["训练时间"], "%Y-%m-%d %H:%M:%S")

            # 按每一行的模型ID从 trained_models 取回对应的模型对象再保存，
            # 避免所有勾选行都保存成最后训练的那个模型
            if model_id in trained_models:
                file_path = save_model(
                    trained_models[model_id],
                    model_type,
                    problem_type,
                    timestamp,
                )
                st.success(
                    f"模型 {model_id}（{model_type}）已成功保存到 {file_path}"
                )
            else:
                st.warning(f"无法保存模型 {model_id}，内存中不存在该模型对象。")


def display_results() -> None:
    """显示模型结果"""
    if st.session_state.model_results:
        st.markdown("## 模型结果")

        with st.container(border=True):
            tabs = get_result_tabs()
            tab_contents = st.tabs(tabs)

            with tab_contents[0]:
                display_model_performance_overview()

            if st.session_state.split_test_set:
                if st.session_state.problem_type == "classification":
                    with tab_contents[1]:
                        display_confusion_matrix()
                    with tab_contents[2]:
                        display_classification_report()
                elif st.session_state.problem_type == "regression":
                    with tab_contents[1]:
                        display_residual_plot()


def get_result_tabs() -> list:
    """获取结果标签页"""
    if st.session_state.problem_type == "classification":
        tabs = ["模型性能概览"]
        if st.session_state.split_test_set:
            tabs.extend(["混淆矩阵", "分类报告"])
    else:
        tabs = ["模型性能概览"]
        if st.session_state.split_test_set:
            tabs.append("残差图")
    return tabs


def display_model_performance_overview() -> None:
    """显示模型性能概览"""
    st.markdown("### 模型性能概览")
    col1, col2 = st.columns(2)
    with col1:
        display_cv_score()

    if st.session_state.split_test_set:
        with col2:
            display_test_score()

    display_r2_score()

    if not st.session_state.split_test_set:
        st.info("模型使用全部数据进行训练，没有单独的测试集评估。")


def display_cv_score() -> None:
    """显示交叉验证分数"""
    if st.session_state.problem_type == "classification":
        st.metric(
            label="交叉验证平均 ROC AUC",
            value=f"{st.session_state.model_results['cv_mean_score']:.4f}",
        )
    else:
        st.metric(
            label="交叉验证平均 MSE",
            value=f"{st.session_state.model_results['cv_mean_score']:.4f}",
            help="对于线性回归模型不使用交叉验证时，此值为训练集 MSE。",
        )


def display_test_score() -> None:
    """显示测试集分数"""
    if st.session_state.problem_type == "classification":
        st.metric(
            label="测试集 ROC AUC",
            value=f"{st.session_state.model_results['test_roc_auc']:.4f}",
        )
    else:
        st.metric(
            label="测试集 MSE",
            value=f"{st.session_state.model_results['test_mse']:.4f}",
        )


def display_r2_score() -> None:
    """显示 R² 分数（仅适用于线性回归）"""
    if (
        st.session_state.problem_type == "regression"
        and st.session_state.model_type == "线性回归"
    ):
        col3, col4 = st.columns(2)
        with col3:
            st.metric(
                label="训练集 R²",
                value=f"{st.session_state.model_results['train_r2']:.4f}",
            )
        if st.session_state.split_test_set:
            with col4:
                st.metric(
                    label="测试集 R²",
                    value=f"{st.session_state.model_results['test_r2']:.4f}",
                )


def display_confusion_matrix() -> None:
    """显示混淆矩阵"""
    st.markdown("### 混淆矩阵")
    cm = st.session_state.model_results["test_confusion_matrix"]
    # 传入真实类别标签以正确渲染坐标轴；取不到时函数内部回退到默认 0/1
    class_labels = st.session_state.model_results.get("classes")
    fig = create_confusion_matrix_plot(cm, class_labels)
    st.plotly_chart(fig)

    with st.expander("混淆矩阵解读", expanded=False):
        st.caption(CONFUSION_MATRIX_EXPLANATION)


def display_classification_report() -> None:
    """显示分类报告"""
    st.markdown("### 分类报告")
    st.text(st.session_state.model_results["test_classification_report"])

    with st.expander("分类报告解读", expanded=False):
        st.caption(CLASSIFICATION_REPORT_EXPLANATION)


def display_residual_plot() -> None:
    """显示残差图"""
    st.markdown("### 残差图")
    y_test = st.session_state.model_results["y_test"]
    y_pred = st.session_state.model_results["y_pred"]
    fig = create_residual_plot(y_test, y_pred)
    st.plotly_chart(fig)

    with st.expander("残差图解读", expanded=False):
        st.caption(REGRESSION_METRICS_EXPLANATION)


def display_model_interpretation() -> None:
    """显示模型解释"""
    if (
        st.session_state.model_results
        and "feature_importance" in st.session_state.model_results
    ):
        st.markdown("## 模型解释")

        with st.container(border=True):
            tab1, tab2, tab3 = st.tabs(["特征重要性", "SHAP分析", "SHAP依赖图"])

            with tab1:
                display_feature_importance()

            with tab2:
                display_shap_importance()

            with tab3:
                display_shap_dependence()


def display_feature_importance() -> None:
    """显示特征重要性"""
    st.markdown("### 模型特征重要性")
    feature_importance = st.session_state.model_results[
        "feature_importance"
    ].sort_values(ascending=True)
    fig = create_feature_importance_plot(feature_importance)
    st.plotly_chart(fig)

    with st.expander("特征重要性解释", expanded=False):
        if st.session_state.model_type == "线性回归":
            st.caption(LINEAR_REGRESSION_FEATURE_IMPORTANCE_EXPLANATION)
        else:
            st.caption(FEATURE_IMPORTANCE_EXPLANATION)


def display_shap_importance() -> None:
    """显示SHAP特征重要性分析"""
    st.markdown("### SHAP特征重要性分析")
    if "shap_results" not in st.session_state:
        calculate_and_store_shap_values()

    if "shap_results" in st.session_state:
        fig = create_shap_importance_plot(
            st.session_state.shap_results["feature_importance"]
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("SHAP特征重要性解释", expanded=False):
            st.markdown(SHAP_FEATURE_IMPORTANCE_EXPLANATION)


def display_shap_dependence() -> None:
    """显示SHAP依赖图"""
    st.markdown("### SHAP依赖图")
    if "shap_results" in st.session_state:
        processed_feature_names = st.session_state.shap_results[
            "processed_feature_names"
        ]
        selected_feature = st.selectbox("选择特征", options=processed_feature_names)

        fig = create_shap_dependence_plot(
            st.session_state.shap_results["shap_values"],
            st.session_state.shap_results["X_processed"],
            np.array(processed_feature_names),
            selected_feature,
        )
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("SHAP依赖图解释", expanded=False):
            st.markdown(SHAP_DEPENDENCE_PLOT_EXPLANATION)


def calculate_and_store_shap_values() -> None:
    """计算并存储SHAP值"""
    if "shap_results" in st.session_state:
        del st.session_state.shap_results

    with st.spinner("正在计算SHAP值，这可能需要一些时间..."):
        try:
            model_step = (
                "regressor"
                if st.session_state.model_type == "线性回归"
                else "classifier"
            )
            shap_results = calculate_shap_values(
                st.session_state.model_results["model"].named_steps[model_step],
                st.session_state.df[st.session_state.feature_columns],
                st.session_state.model_results["model"].named_steps["preprocessor"],
                st.session_state.feature_columns,
                st.session_state.problem_type,
            )
            st.session_state.shap_results = shap_results
        except Exception as e:
            st.error(f"计算SHAP值时出错：{str(e)}")
            st.error("请检查模型类型和数据是否兼容，或尝试使用其他解释方法。")


# 主函数
def main() -> None:
    """主函数，控制页面流程"""
    st.title("🤖 算法建模分析与预测")
    st.markdown("---")

    display_info_message()
    display_operation_settings()

    if st.session_state.mode == "train":
        handle_training_mode()
    else:
        handle_prediction_mode()


def display_operation_settings() -> None:
    """显示操作设置界面"""
    st.markdown("## 操作设置")
    with st.container(border=True):
        col1, col2 = st.columns(2)

        with col1:
            mode = st.radio(
                "选择操作模式",
                options=["训练新模型", "使用已保存模型"],
                index=0 if st.session_state.mode == "train" else 1,
                key="mode_radio",
            )
            st.session_state.mode = "train" if mode == "训练新模型" else "predict"

        with col2:
            problem_type = st.radio(
                "选择问题类型",
                options=["分类问题", "回归问题"],
                index=0 if st.session_state.problem_type == "classification" else 1,
                key="problem_type_radio",
            )
            st.session_state.problem_type = (
                "classification" if problem_type == "分类问题" else "regression"
            )

    display_operation_info()


def display_operation_info() -> None:
    """显示操作信息"""
    if st.session_state.mode == "train":
        st.info("您选择了训练新模型。请上传数据并设置模型参数。")
    else:
        st.info(
            f"您选择了使用已保存的{'分类' if st.session_state.problem_type == 'classification' else '回归'}模型进行预测。请选择模型并上传预测数据。"
        )


def handle_training_mode() -> None:
    """处理训练模式"""
    display_data_upload_and_preview()
    if st.session_state.df is not None:
        display_column_selection()
        display_model_selection()
        display_model_training_and_advanced_settings()
        display_model_records()

    if st.session_state.model_results:
        display_results()

        do_model_interpretation = st.checkbox(
            "进行模型解释", value=st.session_state.do_model_interpretation
        )

        if do_model_interpretation != st.session_state.do_model_interpretation:
            st.session_state.do_model_interpretation = do_model_interpretation
            if not do_model_interpretation and "shap_results" in st.session_state:
                del st.session_state.shap_results

        if st.session_state.do_model_interpretation:
            display_model_interpretation()


def handle_prediction_mode() -> None:
    """处理预测模式"""
    from backend.data_processing.analysis.model_prediction import (
        display_saved_model_selection,
        display_prediction_execution,
        display_prediction_results,
    )

    display_saved_model_selection()
    display_data_upload_and_preview(for_prediction=True)
    if st.session_state.data_validated:
        display_prediction_execution()
        display_prediction_results()


if __name__ == "__main__":
    main()
