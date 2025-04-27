import streamlit as st
import os
import sys
import pandas as pd
from typing import Dict, List, Any, Optional
import uuid
import tempfile
import json

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend.ui_components import show_sidebar, show_footer, apply_common_styles
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from utils.llm_tools import init_language_model, CustomEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langfuse.callback import CallbackHandler

# st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 初始化会话状态
if "csv_file" not in st.session_state:
    st.session_state.csv_file = None
if "loaded_documents" not in st.session_state:
    st.session_state.loaded_documents = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_tab" not in st.session_state:
    st.session_state.current_tab = "知识库构建"
if "top_k" not in st.session_state:
    st.session_state.top_k = 5


def create_langfuse_handler(session_id: str, step: str) -> CallbackHandler:
    """
    创建Langfuse回调处理器

    Args:
        session_id: 会话ID
        step: 处理步骤

    Returns:
        Langfuse回调处理器
    """
    return CallbackHandler(
        tags=["knowledge_base_qa"],
        session_id=session_id,
        metadata={"step": step},
    )


def main():
    """主函数，包含应用的主要逻辑和UI结构"""
    st.title("📚 知识库内容管理与问答")
    st.markdown("---")

    # 使用标签页组织不同功能，合并前两个标签页
    tabs = st.tabs(["知识库构建", "检索查看", "问答助手"])

    # 知识库构建标签页（合并原来的上传文档和构建知识库）
    with tabs[0]:
        if st.session_state.current_tab != "知识库构建":
            st.session_state.current_tab = "知识库构建"
        display_info_message()
        display_workflow()
        
        # 上传文档部分
        st.markdown("## 数据上传")
        handle_file_upload()
        if st.session_state.csv_file:
            process_csv_file()
            preview_loaded_documents()
        
        # 构建知识库部分
        st.markdown("## 知识库构建")
        if not st.session_state.loaded_documents:
            st.warning("请先上传并处理CSV文件")
        else:
            build_knowledge_base_ui()

    # 检索查看标签页
    with tabs[1]:
        if st.session_state.current_tab != "检索查看":
            st.session_state.current_tab = "检索查看"
        retrieval_interface()

    # 问答助手标签页
    with tabs[2]:
        if st.session_state.current_tab != "问答助手":
            st.session_state.current_tab = "问答助手"
        qa_interface()

    show_footer()


def display_info_message():
    """显示知识库内容管理的信息消息"""
    st.info(
        """
        本工具支持从CSV文件中加载知识库内容。您可以上传CSV文件，系统将使用LangChain的文档加载器处理数据，
        并提供预览功能。上传的内容将被用于构建知识库，支持后续的检索和问答功能。
        """
    )


def display_workflow():
    """显示知识库内容管理的工作流程"""
    with st.expander("📋 查看知识库内容管理工作流程", expanded=False):
        st.markdown(
            """
            1. **数据上传**
               - 支持CSV文件上传
               - 自动提取文档内容
            
            2. **知识库构建**
               - 将文档分割成适当大小
               - 文档向量化并存入向量数据库
            
            3. **智能问答**
               - 基于检索增强生成(RAG)技术
               - 支持上下文感知的对话式交互
            """
        )


def handle_file_upload():
    """处理文件上传逻辑"""
    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "上传CSV文件",
            type=["csv"],
            accept_multiple_files=False,
        )
        
        if uploaded_file:
            st.session_state.csv_file = uploaded_file
            st.success(f"成功上传文件: {uploaded_file.name}")


def process_csv_file():
    """处理上传的CSV文件，使用LangChain的文档加载器"""
    if st.session_state.csv_file:
        with st.spinner("正在处理CSV文件..."):
            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
                tmp_file.write(st.session_state.csv_file.getvalue())
                tmp_path = tmp_file.name
            
            try:
                # 直接加载文档，不使用额外的按钮
                loader = CSVLoader(file_path=tmp_path)
                documents = loader.load()
                
                # 存储处理结果
                st.session_state.loaded_documents = documents
                st.success(f"成功加载 {len(documents)} 条文档记录")
            
            except Exception as e:
                st.error(f"处理文件时出错: {str(e)}")
            
            finally:
                # 删除临时文件
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)


def preview_loaded_documents():
    """预览加载的文档内容"""
    if st.session_state.loaded_documents:
        st.markdown("## 文档预览")
        with st.container(border=True):
            # 显示文档数量
            st.write(f"共加载了 {len(st.session_state.loaded_documents)} 条文档记录")
            
            # 创建文档预览
            preview_data = []
            for i, doc in enumerate(st.session_state.loaded_documents[:10]):  # 只预览前10条
                preview_data.append({
                    "序号": i + 1,
                    "内容预览": doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content,
                    "元数据": str(doc.metadata)
                })
            
            # 显示预览表格
            st.dataframe(
                pd.DataFrame(preview_data),
                use_container_width=True
            )
            
            if len(st.session_state.loaded_documents) > 10:
                st.info(f"仅显示前10条记录，共 {len(st.session_state.loaded_documents)} 条")
            
            # 提供详细查看选项
            with st.expander("查看完整文档内容", expanded=False):
                doc_index = st.number_input(
                    "选择要查看的文档序号", 
                    min_value=1, 
                    max_value=len(st.session_state.loaded_documents),
                    value=1
                )
                
                if doc_index:
                    selected_doc = st.session_state.loaded_documents[doc_index - 1]
                    st.write("### 文档内容")
                    st.write(selected_doc.page_content)
                    st.write("### 元数据")
                    st.json(selected_doc.metadata)


def build_knowledge_base_ui():
    """构建向量数据库UI组件"""
    with st.container(border=True):
        st.info("通过这个步骤，系统将处理已加载的文档，分割成适当大小，并构建向量存储以支持语义检索")
        
        col1, col2 = st.columns(2)
        with col1:
            chunk_size = st.number_input("文档分割大小", min_value=100, max_value=2000, value=500, step=100)
        with col2:
            chunk_overlap = st.number_input("分割重叠大小", min_value=0, max_value=500, value=50, step=10)
        
        embedding_model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        
        if st.button("开始构建知识库"):
            with st.spinner("正在构建向量数据库..."):
                try:
                    # 1. 文档分割
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size, 
                        chunk_overlap=chunk_overlap
                    )
                    splits = text_splitter.split_documents(st.session_state.loaded_documents)
                    
                    # 2. 初始化嵌入模型
                    embeddings = CustomEmbeddings(
                        api_key=os.getenv("EMBEDDING_API_KEY", ""),
                        api_url=os.getenv("EMBEDDING_API_BASE", ""),
                        model=embedding_model_name,
                    )
                    
                    # 3. 创建向量存储
                    from langchain_community.vectorstores import FAISS
                    vector_store = FAISS.from_documents(splits, embeddings)
                    st.session_state.vector_store = vector_store
                    
                    st.success(f"成功构建知识库！共索引了 {len(splits)} 个文档片段")
                except Exception as e:
                    st.error(f"构建知识库时出错: {str(e)}")


def qa_interface():
    """问答界面"""
    st.markdown("## 知识库问答助手")
    
    if st.session_state.vector_store is None:
        st.warning("请先构建知识库")
        return
    
    # 添加清空聊天历史的按钮
    if st.button("清空聊天历史", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # 创建一个chat容器和底部输入框容器
    chat_container = st.container()
    # 添加一个隐形的占位符，用于填充空间
    spacer = st.empty()
    input_container = st.container()
    
    # 在底部输入容器中显示输入框
    with input_container:
        prompt = st.chat_input("请输入您的问题:")
        
    # 在聊天容器中显示聊天历史
    with chat_container:
        # 显示历史消息
        if not st.session_state.chat_history:
            st.info("在下方输入问题，开始与知识库对话")
        
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # 处理用户输入
        if prompt:
            # 添加用户消息到历史记录
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # 立即显示用户消息
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # 处理用户问题并显示助手回复
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("思考中...")
                
                try:
                    # 基于RAG的问答
                    answer = process_rag_query(prompt)
                    
                    # 更新消息占位符
                    message_placeholder.markdown(answer)
                    
                    # 添加回复到历史记录
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                except Exception as e:
                    message_placeholder.error(f"处理问题时出错: {str(e)}")
            
            # 强制重新渲染聊天历史
            st.rerun()


def display_chat_history():
    """显示聊天历史"""
    if not st.session_state.chat_history:
        st.info("在下方输入问题，开始与知识库对话")
        return


def process_rag_query(query: str) -> str:
    """处理基于RAG的问答查询"""
    # 1. 获取历史对话
    chat_history = get_chat_history_context()
    
    # 2. 初始化语言模型
    llm = init_language_model(temperature=0.0)
    
    # 3. 执行检索
    retrieved_docs = st.session_state.vector_store.similarity_search(query, k=st.session_state.top_k)
    context = "\n\n".join([
        f"来源: {doc.metadata}\n内容: {doc.page_content}"
        for doc in retrieved_docs
    ])
    
    # 创建Langfuse回调处理器
    langfuse_handler = create_langfuse_handler(
        st.session_state.session_id, 
        "knowledge_base_rag_query"
    )
    
    # 4. 构建提示
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """
        你是一个智能问答助手。请使用以下检索到的内容来回答用户的问题。
        如果无法从检索内容中找到答案，请直接说明你不知道，不要编造信息。
        回答应该简洁明了，最多三句话。
        
        检索内容:
        {context}
        
        历史对话:
        {chat_history}
        """),
        ("human", "{query}")
    ])
    
    # 5. 生成回答，添加Langfuse监控
    chain = prompt_template | llm
    response = chain.invoke(
        {
            "context": context,
            "chat_history": chat_history,
            "query": query
        },
        config={"callbacks": [langfuse_handler]},
    )
    
    return response.content


def get_chat_history_context() -> str:
    """获取对话历史作为上下文"""
    if len(st.session_state.chat_history) <= 1:
        return "无历史对话"
    
    # 只取最近5轮对话
    recent_history = st.session_state.chat_history[-10:]
    formatted_history = []
    
    for message in recent_history:
        role = "用户" if message["role"] == "user" else "助手"
        formatted_history.append(f"{role}: {message['content']}")
    
    return "\n".join(formatted_history)


def retrieval_interface():
    """检索查看界面，直接显示从向量数据库中检索的文档片段"""
    st.markdown("## 知识库直接检索")
    
    if st.session_state.vector_store is None:
        st.warning("请先构建知识库")
        return
    
    with st.container(border=True):
        st.info("本功能允许您输入查询，直接查看从知识库中检索到的最相关文档片段，不经过大模型处理")
        
        # 查询输入和参数设置 - 改为垂直排列
        query = st.text_input("请输入查询内容:")
        search_button = st.button("检索", key="retrieval_search")
        
        # 更新为默认值5，并保存到会话状态中
        st.session_state.top_k = st.slider("显示前N条结果", min_value=1, max_value=10, value=5)
        
        # 检索逻辑：当按下Enter键或点击检索按钮时触发
        if query and (search_button or st.session_state.get("_last_query") != query):
            # 保存当前查询以避免重复执行
            st.session_state._last_query = query
            
            with st.spinner("正在检索相关内容..."):
                try:
                    # 执行相似度搜索
                    retrieved_docs = st.session_state.vector_store.similarity_search_with_score(query, k=st.session_state.top_k)
                    
                    if retrieved_docs:
                        st.success(f"检索完成，找到 {len(retrieved_docs)} 条相关内容")
                        
                        # 显示检索结果 - 所有结果默认展开
                        for i, (doc, score) in enumerate(retrieved_docs):
                            with st.expander(f"结果 #{i+1} (相似度: {1-score:.4f})", expanded=True):
                                st.markdown(f"### 【文档 {i+1}】")
                                st.markdown(doc.page_content)
                    else:
                        st.info("未找到相关内容")
                        
                except Exception as e:
                    st.error(f"检索过程中出错: {str(e)}")


# 添加兼容性函数
def build_knowledge_base():
    """构建向量数据库（保留为兼容性）"""
    build_knowledge_base_ui()


if __name__ == "__main__":
    main()