import streamlit as st
import sys
import os
from PIL import Image
import hmac

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# 应用入口处一次性配置日志（库模块只 getLogger，不在导入期配置根 logger）
from utils.logging_config import setup_logging

setup_logging()

# 在文件开头添加这个函数
def maintain_auth_token():
    """Maintain auth_token across page navigation"""
    # 如果当前 URL 没有 auth_token 但 session 中有登录状态
    if ("auth_token" not in st.query_params and 
        st.session_state.get("password_correct", False) and 
        "current_token" in st.session_state):
        # 恢复 auth_token 到 URL
        st.query_params["auth_token"] = st.session_state.current_token

if "role" in st.query_params:
    st.session_state.role = st.query_params.role

if "role" not in st.session_state:
    st.session_state.role = None

ROLES = ["User", "Recruiter", "Admin"]


def check_password():
    """Returns `True` if the user had a correct password."""

    # 检查 URL 参数中是否有有效的登录状态
    if "auth_token" in st.query_params:
        stored_token = st.query_params["auth_token"]
        if stored_token in st.secrets["auth_tokens"].values():
            # 根据 token 找到对应的用户和角色
            for username, token in st.secrets["auth_tokens"].items():
                if token == stored_token:
                    st.session_state["password_correct"] = True
                    st.session_state.role = st.secrets.user_roles[username]
                    # 保存当前 token 到 session state
                    st.session_state.current_token = stored_token
                    return True
        
    def login_form():
        """Form with widgets to collect user information"""
        with st.form("Credentials"):
            st.text_input("Username", key="username")
            st.text_input("Password", type="password", key="password")
            st.form_submit_button("Log in", on_click=password_entered)

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["username"] in st.secrets["passwords"] and hmac.compare_digest(
            st.session_state["password"],
            st.secrets.passwords[st.session_state["username"]],
        ):
            st.session_state["password_correct"] = True
            # Set the role based on username
            st.session_state.role = st.secrets.user_roles[st.session_state["username"]]
            # 获取并保存 token
            current_token = st.secrets["auth_tokens"][st.session_state["username"]]
            st.session_state.current_token = current_token
            # 设置 auth_token 到 URL
            st.query_params["auth_token"] = current_token
            del st.session_state["password"]
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("Intelligent HR Assistant")
    st.header("Log in")
    login_form()
    if "password_correct" in st.session_state:
        st.error("😕 User not known or password incorrect")
    return False


def logout():
    st.session_state.password_correct = False
    st.session_state.role = None
    # 清除 session 中的 token
    if "current_token" in st.session_state:
        del st.session_state.current_token
    # 清除 URL 中的 auth_token
    if "auth_token" in st.query_params:
        del st.query_params["auth_token"]
    st.rerun()

# 在主程序开始时调用
maintain_auth_token()

if not check_password():
    st.stop()

role = st.session_state.role

home_page = st.Page("Home.py", title="首页",
                    icon=":material/home:", default=True)
table_operation = st.Page(
    "page/table_operation.py", title="智能数据整理", icon=":material/table_view:"
)
data_cleaning = st.Page(
    "page/data_cleaning.py",
    title="自动化数据清洗",
    icon=":material/mop:",
)
document_check = st.Page(
    "page/document_check.py",
    title="报告质量检验",
    icon=":material/verified:",
)
ai_translation = st.Page(
    "page/ai_translation.py",
    title="智能语境翻译",
    icon=":material/translate:",
)
sentiment_analysis = st.Page(
    "page/sentiment_analysis.py",
    title="情感分析与标注",
    icon=":material/family_star:",
)
text_clustering = st.Page(
    "page/text_clustering.py",
    title="文本聚类分析",
    icon=":material/folder_open:",
)
resume_upload = st.Page(
    "page/resume_upload.py",
    title="简历上传系统",
    icon=":material/upload:",
)
resume_parsing = st.Page(
    "page/resume_parsing.py",
    title="智能简历解析",
    icon=":material/newspaper:",
)
resume_recommendation = st.Page(
    "page/resume_recommendation.py",
    title="智能简历推荐",
    icon=":material/thumb_up:",
)
ai_research = st.Page(
    "page/ai_research.py",
    title="AI研究助手",
    icon=":material/quick_reference_all:",
)
modeling_analysis = st.Page(
    "page/modeling_analysis.py",
    title="建模与分析",
    icon=":material/monitoring:",
)
vector_db_management = st.Page(
    "page/vector_db_management.py",
    title="向量数据库管理",
    icon=":material/database:",
)
exam_interface = st.Page(
    "page/exam_interface.py",
    title="智能考试系统",
    icon=":material/quiz:",
)
sql_assistant = st.Page(
    "page/sql_assistant.py",
    title="SQL查询助手",
    icon=":material/conditions:",
)

account_pages = [home_page]
request_pages = [
    table_operation,
    sql_assistant,
    data_cleaning,
    document_check,
    ai_translation,
    sentiment_analysis,
    text_clustering,
    ai_research,
    modeling_analysis,
]
recruitment_pages = [resume_upload, resume_parsing, resume_recommendation]
admin_pages = [vector_db_management]


# st.logo(
#     "frontend/assets/horizontal_blue.png",
#     icon_image="frontend/assets/icon_blue.png",
# )

# 功能分类
data_processing = [
    table_operation,
    sql_assistant,
    data_cleaning,
    document_check,
    vector_db_management,
]

text_analysis = [ai_translation, sentiment_analysis, text_clustering]

talent_management = [
    resume_upload,
    resume_parsing,
    resume_recommendation,
    exam_interface,
]

decision_support = [ai_research, modeling_analysis]

# 根据角色分配权限
page_dict = {}
if st.session_state.role in ["User", "Recruiter", "Admin"]:
    page_dict["数据处理与管理"] = [
        p for p in data_processing if p != vector_db_management
    ]
    page_dict["文本分析与洞察"] = text_analysis
    page_dict["辅助决策与研究"] = decision_support

if st.session_state.role in ["Recruiter", "Admin"]:
    page_dict["人才管理工具"] = talent_management

if st.session_state.role == "Admin":
    page_dict["数据处理与管理"].append(vector_db_management)

def login():
    st.title("Intelligent HR Assistant")
    st.header("Please log in to continue")
    return

if len(page_dict) > 0:
    pg = st.navigation({"Account": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(login)])

pg.run()
