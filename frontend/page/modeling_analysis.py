import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
from datetime import datetime

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend_demo.ui_components import show_sidebar, show_footer, apply_common_styles
from backend_demo.data_processing.analysis.ml_explanations import (
    ML_TOOL_INFO,
    CONFUSION_MATRIX_EXPLANATION,
    CLASSIFICATION_REPORT_EXPLANATION,
    FEATURE_IMPORTANCE_EXPLANATION,
    REGRESSION_METRICS_EXPLANATION,
    SHAP_FEATURE_IMPORTANCE_EXPLANATION,
    SHAP_DEPENDENCE_PLOT_EXPLANATION,
    LINEAR_REGRESSION_FEATURE_IMPORTANCE_EXPLANATION,
    XGBOOST_LABEL_ENCODING_EXPLANATION,
)
from backend_demo.data_processing.analysis.model_utils import (
    initialize_session_state,
)
from backend_demo.data_processing.analysis.ml_components import (
    display_info_message,
    display_model_selection,
)
from backend_demo.data_processing.analysis.model_training import (
    display_data_upload_and_preview,
    display_column_selection,
    display_model_training_and_advanced_settings,
    display_model_records,
    display_results,
    display_model_interpretation,
)
from backend_demo.data_processing.analysis.model_prediction import (
    display_saved_model_selection,
    display_prediction_execution,
    display_prediction_results,
)

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

if "initialized" not in st.session_state:
    st.session_state.update(initialize_session_state())
    st.session_state.initialized = True


def display_operation_settings():
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

    # 根据选择显示相应的提示信息
    if st.session_state.mode == "train":
        st.info("您选择了训练新模型。请上传数据并设置模型参数。")
    else:
        st.info(
            f"您选择了使用已保存的{'分类' if st.session_state.problem_type == 'classification' else '回归'}模型进行预测。请选择模型并上传预测数据。"
        )


def main():
    """主函数，控制页面流程"""
    initialize_session_state()

    st.title("🤖 算法建模分析与预测")
    st.markdown("---")

    display_info_message()
    display_operation_settings()

    if st.session_state.mode == "train":
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
    else:
        display_saved_model_selection()
        display_data_upload_and_preview(for_prediction=True)
        if st.session_state.data_validated:
            display_prediction_execution()
            display_prediction_results()

    show_footer()


main()
