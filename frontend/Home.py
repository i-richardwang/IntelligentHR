import streamlit as st
import sys
import os
from PIL import Image

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

set_llm_cache(SQLiteCache(database_path="data/llm_cache/langchain.db"))

from frontend.ui_components import show_sidebar, show_footer, apply_common_styles

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()


def main():
    st.title("🚀 Intelligent HR")
    st.markdown("---")

    display_project_intro()

    st.markdown("---")

    image = Image.open("frontend/assets/IntelligentHR_Intro.png")
    st.image(image, width="stretch")

    display_feature_overview()
    display_project_highlights()
    display_documentation_link()

    # 页脚
    show_footer()


def display_project_intro():
    st.info(
        """
        智能HR助手是一个实验性的人力资源管理工具集，旨在探索AI技术在HR领域的应用潜力。
        
        工具集涵盖了从数据处理、文本分析到决策支持的多个HR工作分析环节，致力于为人力资源管理提供全方位的智能化解决方案。
        """
    )


def display_feature_overview():
    st.markdown("## 功能概览")

    features = [
        ("🧮 智能数据整理", "通过自然语言交互实现复杂表格操作"),
        ("🏢 自动化数据清洗", "利用大语言模型和向量化技术进行清洗数据"),
        ("🔍 智能文档检查", "智能文档审核工具，自动识别和纠正错误"),
        ("🌐 智能语境翻译", "结合上下文理解的高质量多语言翻译"),
        ("🏷️ 情感分析与标注", "基于大语言模型的情感分析和自动标注"),
        ("🗂️ 文本聚类分析", "自动提炼大量文本中的话题模式和文本分类"),
        ("📤 简历上传系统", "支持批量全格式简历上传，自动去重和存储"),
        ("📇 智能简历解析", "利用大模型从简历中精确提取结构化信息"),
        ("🧩 智能简历推荐", "对话式需求分析和多维度评分的智能推荐"),
        ("📝 AI研究助手", "多阶段智能代理协作，生成定制的研究报告"),
        ("🤖 算法建模分析", "交互式机器学习建模平台，支持模型解释"),
        ("🧑‍💻 智能考试系统", "根据培训内容及时生成考试题目和作答系统"),
        # ("💾 向量数据库管理", "高效管理和更新用于智能检索的向量数据库"),
    ]

    cols = st.columns(2)
    for i, (icon, desc) in enumerate(features):
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"##### {icon}")
                st.markdown(desc)


def display_project_highlights():
    st.markdown("## 项目亮点")
    st.markdown(
        """
        - **实际场景应用**: 将大语言模型能力无缝集成到HR工作流程，解决实际痛点
        - **效率倍增**: 自动化处理繁琐的数据清理和分析任务，大幅提升工作效率
        - **智能洞察**: 利用先进的AI技术提供深度数据分析，支持数据驱动决策
        - **灵活定制**: 针对HR领域的多样化需求，提供可定制的AI解决方案
        - **用户友好**: 直观的界面设计和自然语言交互，降低使用门槛
        """
    )


def display_documentation_link():
    st.markdown("## 产品文档")

    st.markdown("探索完整功能、使用指南和最佳实践")
    st.link_button(
        "📚 查看完整文档",
        "https://docs.irichard.wang/intelligenthr/intelligenthr-intro",
    )


main()
