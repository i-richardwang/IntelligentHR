import streamlit as st

# 版本号
VERSION = "0.0.5"


def show_sidebar():
    """
    显示应用程序的侧边栏。
    """
    with st.sidebar:
        _render_sidebar_content()
        _add_version_info()


def _render_sidebar_content():
    """
    渲染侧边栏的主要内容。
    """
    st.markdown(_get_sidebar_style(), unsafe_allow_html=True)
    # _get_sidebar_content()
    # st.markdown("---")


def _get_sidebar_style():
    """
    返回侧边栏的CSS样式。
    """
    return """
    <style>
    [data-testid="stSidebarNav"] {
        background-image: none;
        padding-top: 0;
        max-height: none;
    }
    [data-testid="stSidebarNav"] > ul {
        max-height: none;
        overflow-y: visible;
    }
    .sidebar-content {
        padding: 1rem 0;
    }
    .sidebar-title {
        color: #4F8BF9;
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    </style>
    """


def _get_sidebar_content():
    """
    返回侧边栏的HTML内容。
    """

    st.caption("[GitHub 项目仓库](https://github.com/i-Richard-me/IntelligentHR)")
    st.caption("**Richard Wang** [contact@xmail.ing](mailto:contact@xmail.ing)")


def _add_version_info():
    """
    添加版本信息到侧边栏。
    """
    st.caption(f"Version {VERSION}")


def show_footer():
    """
    显示应用程序的页脚。
    """
    st.markdown(_get_footer_style() + _get_footer_content(), unsafe_allow_html=True)


def _get_footer_style():
    """
    返回页脚的CSS样式。
    """
    return """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f0f2f6;
        color: #31333F;
        text-align: center;
        padding: 10px 0;
        font-size: 0.8rem;
    }
    .footer a {
        color: #4F8BF9;
        text-decoration: none;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    """


def _get_footer_content():
    """
    返回页脚的HTML内容。
    """
    return """
    <div class="footer">
        <span>© 2024 Richard Wang</span> &nbsp;|&nbsp; 
        <a href="https://github.com/i-Richard-me/IntelligentHR" target="_blank">GitHub</a> &nbsp;|&nbsp; 
        <a href="mailto:contact@xmail.ing">contact@xmail.ing</a>
    </div>
    """


def apply_common_styles():
    """
    应用通用的CSS样式。
    """
    st.markdown(_get_common_styles(), unsafe_allow_html=True)


def _get_common_styles():
    """
    返回通用的CSS样式。
    """
    return """
    <style>
    .stTextInput>div>div>input {
        border-color: #E0E0E0;
    }
    .stProgress > div > div > div > div {
        background-color: #4F8BF9;
    }
    h2, h3, h4 {
        border-bottom: 2px solid;
        padding-bottom: 0.5rem;
        margin-bottom: 1rem;
    }
    h2 {
        color: #1E90FF;
        border-bottom-color: #1E90FF;
        font-size: 1.8rem;
        margin-top: 1.5rem;
    }
    h3 {
        color: #16A085;
        border-bottom-color: #16A085;
        font-size: 1.5rem;
        margin-top: 1rem;
    }
    h4 {
        color: #E67E22;
        border-bottom-color: #E67E22;
        font-size: 1.2rem;
        margin-top: 0.5rem;
    }
    .workflow-container {
        background-color: rgba(248, 249, 250, 0.05);
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        border: 1px solid rgba(0, 0, 0, 0.125);
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }

    @media (prefers-color-scheme: dark) {
        .workflow-container {
            background-color: rgba(33, 37, 41, 0.05);
            border-color: rgba(255, 255, 255, 0.125);
        }
        h2 {
            color: #3498DB;
            border-bottom-color: #3498DB;
        }
        h3 {
            color: #2ECC71;
            border-bottom-color: #2ECC71;
        }
        h4 {
            color: #F39C12;
            border-bottom-color: #F39C12;
        }
    }
    .workflow-step {
        margin-bottom: 1rem;
    }
    </style>
    """

