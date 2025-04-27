import io
import os
import sys
import json
from typing import List, Dict, Tuple, Optional

import streamlit as st
import pandas as pd
from pymilvus import Collection, utility

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from utils.llm_tools import CustomEmbeddings
from utils.vector_db_utils import (
    connect_to_milvus,
    initialize_vector_store,
    create_milvus_collection,
    insert_to_milvus,
    get_collection_stats,
    update_milvus_records,
)
from frontend.ui_components import show_sidebar, show_footer, apply_common_styles

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()

# 显示侧边栏
show_sidebar()

# 加载配置文件
with open("data/config/collections_config.json", "r", encoding="utf-8") as f:
    CONFIG = json.load(f)


def insert_examples_to_milvus(
    examples: List[Dict], collection_config: Dict, db_name: str, overwrite: bool
):
    """将示例插入到Milvus数据库"""
    connect_to_milvus(db_name)

    embeddings = CustomEmbeddings(
        api_key=os.getenv("EMBEDDING_API_KEY", ""),
        api_url=os.getenv("EMBEDDING_API_BASE", ""),
        model=os.getenv("EMBEDDING_MODEL", ""),
    )

    data = []
    vectors = {}

    for example in examples:
        row_data = {}
        for field in collection_config["fields"]:
            if field["name"] != "id":  # 排除 id 字段
                # 根据字段类型进行转换
                if field["type"] == "str":
                    value = str(example[field["name"]])
                elif field["type"] == "int":
                    value = int(example[field["name"]])
                elif field["type"] == "float":
                    value = float(example[field["name"]])
                else:
                    value = example[field["name"]]
                row_data[field["name"]] = value

        data.append(row_data)

        for field_name in collection_config["embedding_fields"]:
            if field_name not in vectors:
                vectors[field_name] = []
            embedding_text = str(example[field_name])
            vector = embeddings.embed_query(embedding_text)
            vectors[field_name].append(vector)

    if not utility.has_collection(collection_config["name"]):
        collection = create_milvus_collection(
            collection_config, len(next(iter(vectors.values()))[0])
        )
    else:
        collection = Collection(collection_config["name"])

    if overwrite:
        update_milvus_records(
            collection, data, vectors, collection_config["embedding_fields"]
        )
    else:
        insert_to_milvus(collection, data, vectors)

    return len(examples)


def process_csv_file(file, collection_config: Dict):
    """处理上传的CSV文件"""
    examples = []
    csv_file = io.StringIO(file.getvalue().decode("utf-8"))
    df = pd.read_csv(csv_file)

    required_columns = [field["name"] for field in collection_config["fields"]]

    # 检查是否包含所有必需的列
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f"CSV文件缺少以下列: {', '.join(missing_columns)}")

    for _, row in df.iterrows():
        example = {col: row[col] for col in required_columns}
        examples.append(example)

    return examples


def get_existing_records(
    collection_config: Dict, db_name: str
) -> Optional[pd.DataFrame]:
    """获取已存在的记录，如果collection不存在则返回None"""
    connect_to_milvus(db_name)
    if not utility.has_collection(collection_config["name"]):
        return None

    collection = initialize_vector_store(collection_config["name"])

    # 获取所有字段名
    field_names = [field["name"] for field in collection_config["fields"]]

    # 查询所有记录
    results = collection.query(expr="id >= 0", output_fields=field_names)

    return pd.DataFrame(results)


def dedup_examples(
    new_examples: List[Dict],
    existing_records: Optional[pd.DataFrame],
    collection_config: Dict,
) -> Tuple[List[Dict], int]:
    """对新上传的数据进行去重，基于所有用于生成向量的字段"""
    if existing_records is None:
        return new_examples, 0

    new_df = pd.DataFrame(new_examples)

    # 确保列名一致性
    new_df.columns = new_df.columns.str.strip().str.lower()
    if existing_records is not None:
        existing_records.columns = existing_records.columns.str.strip().str.lower()

    # 使用所有用于生成向量的字段进行比较
    embedding_fields = collection_config["embedding_fields"]

    # 使用这些字段进行合并
    merged = pd.merge(
        new_df,
        existing_records,
        on=embedding_fields,
        how="left",
        indicator=True,
        suffixes=("", "_existing"),
    )

    # 找出未匹配的记录（新数据）
    new_records = merged[merged["_merge"] == "left_only"]

    # 计算重复记录数量
    duplicate_count = len(new_examples) - len(new_records)

    # 只保留原始列
    original_columns = new_df.columns
    new_records = new_records[original_columns]

    # 转换回字典列表
    new_examples = new_records.to_dict("records")

    return new_examples, duplicate_count


def display_collection_info(collection_config: Dict):
    """显示Collection信息"""
    with st.container(border=True):
        st.subheader("Collection 信息")
        st.write(f"**名称:** {collection_config['name']}")
        st.write(f"**描述:** {collection_config['description']}")
        st.write("**字段:**")
        for field in collection_config["fields"]:
            st.write(f"- {field['name']}: {field['description']}")


def display_collection_stats(collection_config: Dict, db_name: str):
    """显示Collection统计信息"""
    with st.container(border=True):
        st.subheader("数据统计")
        connect_to_milvus(db_name)
        if utility.has_collection(collection_config["name"]):
            collection = initialize_vector_store(collection_config["name"])
            stats = get_collection_stats(collection)
            st.write(f"**实体数量:** {stats['实体数量']}")
            st.write(f"**字段数量:** {stats['字段数量']}")
            st.write(f"**索引类型:** {stats['索引类型']}")
        else:
            st.info("该Collection尚未创建")


def display_data_preview(
    new_examples: List[Dict], duplicate_count: int, collection_exists: bool
):
    """显示数据预览"""
    with st.container(border=True):
        st.subheader("数据预览")
        st.write(f"**上传记录总数:** {len(new_examples) + duplicate_count}")
        if collection_exists:
            st.write(f"**数据库中已存在记录:** {duplicate_count}条")
            st.write(f"**待插入新记录:** {len(new_examples)}条")
        else:
            st.write("**待插入新记录:** 所有上传记录（Collection尚未创建）")

        if len(new_examples) > 0:
            st.write("**新记录预览:**")
            new_df = pd.DataFrame(new_examples)
            st.dataframe(new_df)
        elif collection_exists:
            st.info("所有上传的记录都已存在于数据库中，没有新数据需要插入。")


def main():
    st.title("🗄️ Milvus数据库管理")
    st.markdown("---")

    # 显示功能介绍
    display_db_management_info()

    # 数据库选择
    st.header("选择数据库")
    db_names = [os.getenv("VECTOR_DB_DATABASE", "examples"), "data_cleaning"]
    selected_db = st.selectbox("选择要操作的数据库", db_names)

    # Collection 选择
    st.header("选择 Collection")
    collection_names = list(CONFIG["collections"].keys())
    selected_collection = st.selectbox("选择要操作的Collection", collection_names)
    collection_config = CONFIG["collections"][selected_collection]

    # 连接到Milvus时使用所选数据库
    connect_to_milvus(selected_db)

    # 显示Collection信息
    display_collection_info(collection_config)

    # 显示Collection统计
    display_collection_stats(collection_config, selected_db)

    st.header("上传和插入数据")

    # 文件上传
    uploaded_file = st.file_uploader("上传CSV文件", type=["csv"])

    # 添加覆盖重复数据的选项
    overwrite_option = st.checkbox(
        "覆盖重复数据", value=False, help="选中此项将更新已存在的记录，而不是忽略它们"
    )

    if uploaded_file is not None:
        try:
            examples = process_csv_file(uploaded_file, collection_config)
            st.success(f"成功读取 {len(examples)} 条记录")

            # 获取已存在的记录
            existing_records = get_existing_records(collection_config, selected_db)
            collection_exists = existing_records is not None

            # 去重
            new_examples, duplicate_count = dedup_examples(
                examples, existing_records, collection_config
            )

            # 显示数据预览
            display_data_preview(new_examples, duplicate_count, collection_exists)

            if len(new_examples) > 0 or (overwrite_option and duplicate_count > 0):
                if st.button("插入到Milvus数据库"):
                    with st.spinner("正在插入数据..."):
                        if overwrite_option:
                            inserted_count = insert_examples_to_milvus(
                                examples, collection_config, selected_db, True
                            )
                            st.success(
                                f"成功插入或更新 {inserted_count} 条记录到Milvus数据库"
                            )
                        else:
                            inserted_count = insert_examples_to_milvus(
                                new_examples, collection_config, selected_db, False
                            )
                            st.success(
                                f"成功插入 {inserted_count} 条新记录到Milvus数据库"
                            )

        except ValueError as ve:
            st.error(f"CSV文件格式错误: {str(ve)}")
        except Exception as e:
            st.error(f"处理文件时出错: {str(e)}")
            st.error("请确保CSV文件格式正确，并且包含所有必需的列。")

    show_footer()


def display_db_management_info():
    st.info(
        """
    Milvus数据库管理工具用于高效管理和更新向量数据库中的数据。
    支持CSV文件上传、数据预览和批量插入，便于维护和扩展向量数据集。
    """
    )


main()
