# 01_🔍_AI研究助手.py

import streamlit as st
import asyncio
from PIL import Image
import sys
import os

# 获取项目根目录的绝对路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 将项目根目录添加到 sys.path
sys.path.append(project_root)

from backend.ai_research.ai_researcher import AIResearcher
from backend.ai_research.research_enums import ReportType, Tone
from frontend.ui_components import show_sidebar, show_footer, apply_common_styles

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()


def main():
    if "generated_report" not in st.session_state:
        st.session_state.generated_report = ""
    if "verbose_output" not in st.session_state:
        st.session_state.verbose_output = ""

    st.title("📝 AI 研究助手")
    st.markdown("---")

    display_info_message()
    display_workflow()
    display_research_settings()
    display_report()
    show_footer()


def display_info_message():
    """
    显示AI研究助手的功能介绍。
    """
    st.info(
        """
    AI 研究助手是一个基于大语言模型的研究工具，旨在协助用户进行深入的主题研究。

    该工具能够根据用户提供的主题自动生成相关子查询，从多个来源收集信息，并进行分析整理。
    系统支持多种报告类型和语气，可根据用户需求生成定制化的研究报告。
    
    AI 研究助手适用于需要快速获取和整理特定主题信息的场景。
    """
    )


def display_workflow():
    """
    显示AI研究助手的工作流程。
    """
    with st.expander("📝 查看AI研究助手工作流程", expanded=False):

        with st.container(border=True):
            col1, col2 = st.columns([1, 1])

            with col2:
                st.markdown(
                    """
                    <div class="workflow-container">
                        <div class="workflow-step">
                            <strong>1. 智能代理选择</strong>: 根据研究主题，系统自动选择合适的AI代理角色和专业指令。
                        </div>
                        <div class="workflow-step">
                            <strong>2. 子查询生成</strong>: AI代理根据主题生成多个相关的子查询，以全面覆盖研究范围。
                        </div>
                        <div class="workflow-step">
                            <strong>3. 并行信息检索</strong>: 系统同时处理多个子查询，从网络搜索引擎获取相关信息。
                        </div>
                        <div class="workflow-step">
                            <strong>4. 上下文压缩</strong>: 使用嵌入技术和相似度匹配，从检索到的大量信息中提取最相关的内容。
                        </div>
                        <div class="workflow-step">
                            <strong>5. 报告生成</strong>: 根据压缩后的上下文信息，生成结构化的研究报告，并根据指定的语气和格式进行调整。
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


def display_research_settings():
    st.markdown("## 研究设置")
    with st.container(border=True):
        query = st.text_input("请输入您的研究主题：")

        col1, col2 = st.columns(2)
        with col1:
            tone = st.selectbox("请选择报告语气：", (tone.value for tone in Tone))

        with col2:
            report_type = st.selectbox(
                "请选择报告类型：",
                (type.value for type in ReportType),
                format_func=lambda x: {
                    "research_report": "综合研究报告（全面分析和总结）",
                    "resource_report": "资源汇总报告（相关资料和参考文献列表）",
                    "outline_report": "研究大纲（主要观点和结构框架）",
                    "detailed_report": "详细深度报告（全面且深入的分析）",
                    "custom_report": "自定义报告（根据特定需求定制）",
                    "subtopic_report": "子主题报告（特定子话题的深入分析）",
                }.get(x, x),
            )

        # 高级设置
        with st.expander("高级设置"):
            max_iterations = st.slider(
                "最大子查询数量",
                min_value=1,
                max_value=10,
                value=3,
                help="控制子查询的最大数量",
            )
            max_subtopics = st.slider(
                "最大子主题数",
                min_value=1,
                max_value=10,
                value=3,
                help="控制详细报告中子主题的最大数量",
            )
            max_search_results_per_query = st.slider(
                "每个查询的最大搜索结果数",
                min_value=1,
                max_value=20,
                value=5,
                help="控制每个子查询的最大搜索结果数",
            )

        if st.button("开始研究"):
            if query:
                # 每次运行前清空上一轮的进度日志，避免多次研究的输出累积
                st.session_state.verbose_output = ""
                with st.spinner("正在进行研究，请稍候..."):
                    # 创建一个 expander 来包含详细输出
                    verbose_expander = st.expander("显示研究进度", expanded=True)

                    # 在 expander 中创建一个 container 来显示详细信息
                    with verbose_expander:
                        verbose_container = st.empty()

                    # 创建一个回调函数来更新详细信息
                    def update_verbose(message):
                        st.session_state.verbose_output += message + "\n"
                        verbose_container.text(st.session_state.verbose_output)

                    try:
                        # 创建 AIResearcher 实例，传入回调函数和新的配置参数
                        researcher = AIResearcher(
                            query=query,
                            report_type=report_type,
                            tone=Tone(tone),
                            verbose=True,
                            verbose_callback=update_verbose,
                            max_iterations=max_iterations,
                            max_subtopics=max_subtopics,
                            max_search_results_per_query=max_search_results_per_query,
                        )

                        # 运行研究过程
                        report = asyncio.run(researcher.run())

                        # 存储生成的报告
                        st.session_state.generated_report = report
                    except ValueError as e:
                        st.error(f"参数错误：{str(e)}")
                        return
                    except Exception as e:
                        st.error(f"研究过程中发生错误：{str(e)}，请稍后重试或联系数据团队。")
                        return

                st.success("研究完成！")


def display_report():
    if st.session_state.generated_report:
        st.markdown("## 研究报告")
        with st.container(border=True):
            st.markdown(st.session_state.generated_report)

        # 添加下载按钮
        st.download_button(
            label="📥 下载 Markdown 格式报告",
            data=st.session_state.generated_report,
            file_name="研究报告.md",
            mime="text/markdown",
        )


main()
