#!/usr/bin/env python3
"""
demo 数据一键初始化脚本（开箱即跑）。

迁移自多个 standalone 项目的初始化流程（QueryBot / LLM-DataCleaner / table-operations），
统一适配到 IntelligentHR 的纯本地零 server 栈：业务库落本地 SQLite、向量集合落本地 libSQL，
各库/集合的解析口径与对应消费方节点**逐字一致**，保证「写入的位置」正好是「功能读取的位置」。

覆盖三个域（可用 ``--domain`` 选择，默认全部）：

1. ``sql``      —— SQL 助手
   - 业务库：3 张招聘表 → ``sqlite:///{SQLBOT_DB_PATH}``（默认 data/sqlbot.db）
   - 向量库 ``VECTOR_DB_DATABASE``（默认 ""→default）：table_descriptions / term_descriptions
     / query_examples
2. ``cleaning`` —— 数据清洗（实体名标准化）
   - 向量库 ``VECTOR_DB_DATA_CLEANING``（默认 default）：company_data / school_data
3. ``table-op`` —— 智能表格操作
   - 向量库 ``VECTOR_DB_DATABASE``（默认 examples）：tools_description（由 IHR 自身工具函数
     导出）/ data_operation_examples（少样本示例）

数据来源均已迁移进本仓库，运行时不依赖任何兄弟仓库：
- SQL 业务/元数据：``tools/demo_data/generate_recruitment_data.py``（代码生成，确定性）
- 实体标准化：``tools/demo_data/entity_standardization.py``
- 表格操作示例：``tools/demo_data/data_operation_examples.csv``
- 工具描述：从 ``backend/.../table_operation/table_operations.py`` 的 @tool 函数实时导出

运行：
    python -m tools.seed_demo_data [--domain all|sql|cleaning|table-op] [--overwrite]
                                   [--activities N] [--interviewers N] [--candidates N]
                                   [--skip-business] [--skip-vectors]

说明：
- 业务表为确定性 demo（固定随机种子），每次以 ``replace`` 重建，天然幂等；
- 向量集合默认「已存在且非空则跳过」，``--overwrite`` 则按 embedding 字段原地 upsert；
- 业务库步骤不依赖 LLM / 网络；向量库步骤需已正确配置 embedding provider 环境变量。
"""

import os
import sys
import csv
import json
import logging
import argparse
from typing import Dict, List

# 项目根目录入 sys.path，支持 `python -m tools.seed_demo_data` 及直接执行
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.env_loader import load_env
from tools.demo_data.generate_recruitment_data import RecruitmentDataGenerator
from tools.demo_data.entity_standardization import company_rows, school_rows

logger = logging.getLogger("seed_demo_data")

DEMO_DATA_DIR = os.path.join(os.path.dirname(__file__), "demo_data")
DOMAIN_ORDER = ["sql", "cleaning", "table-op"]


def _load_collections_config() -> Dict:
    """加载集合配置文件。"""
    config_path = os.path.join(PROJECT_ROOT, "data", "config", "collections_config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)["collections"]


def _coerce(field_type: str, value):
    """按配置字段类型做最小化转换（与向量库管理页保持一致）。"""
    if field_type == "int":
        return int(value)
    if field_type == "float":
        return float(value)
    return str(value)


# ---------------------------------------------------------------------- #
# 业务库（SQLite）
# ---------------------------------------------------------------------- #
def seed_business_tables(tables: Dict[str, List[Dict]]) -> None:
    """将业务表写入本地 SQLite 业务库（``replace`` 重建，幂等）。"""
    import pandas as pd
    from sqlalchemy import create_engine

    db_path = os.getenv("SQLBOT_DB_PATH", "data/sqlbot.db")
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    # 与 sql_execution_node / table_structure_node 完全一致的连接串
    engine = create_engine(f"sqlite:///{db_path}")

    logger.info("业务库：%s", db_path)
    for table_name, rows in tables.items():
        df = pd.DataFrame(rows)
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        logger.info("  ✓ %-32s %d 行", table_name, len(df))
    engine.dispose()


# ---------------------------------------------------------------------- #
# 向量库（libSQL）
# ---------------------------------------------------------------------- #
def _seed_one_collection(name: str, config: Dict, rows: List[Dict], embeddings, overwrite: bool) -> int:
    """初始化单个向量集合，返回写入的记录数（跳过时返回 0）。"""
    from utils.vector_db_utils import (
        has_collection,
        get_num_entities,
        create_milvus_collection,
        insert_to_milvus,
        update_milvus_records,
    )

    exists = has_collection(name)
    if exists and get_num_entities(name) > 0 and not overwrite:
        logger.info("  ⏭  %-24s 已存在且非空，跳过（--overwrite 可强制重建）", name)
        return 0

    # 组装标量数据 + 逐 embedding 字段的向量
    data: List[Dict] = []
    vectors: Dict[str, List[List[float]]] = {}
    for row in rows:
        row_data = {}
        for field in config["fields"]:
            if field["name"] == "id":
                continue
            row_data[field["name"]] = _coerce(field["type"], row[field["name"]])
        data.append(row_data)
        for ef in config["embedding_fields"]:
            vectors.setdefault(ef, []).append(embeddings.embed_query(str(row[ef])))

    dim = len(next(iter(vectors.values()))[0])
    if not exists:
        create_milvus_collection(config, dim)

    if overwrite and exists and get_num_entities(name) > 0:
        update_milvus_records(name, data, vectors, config["embedding_fields"])
        logger.info("  ✓ %-24s upsert %d 条（overwrite）", name, len(data))
    else:
        insert_to_milvus(name, data, vectors)
        logger.info("  ✓ %-24s 写入 %d 条", name, len(data))
    return len(data)


def _seed_vector_collections(db_name: str, rows_by_collection: Dict[str, List[Dict]], overwrite: bool) -> None:
    """把「集合名 -> 行列表」灌入指定逻辑向量库。"""
    from utils.vector_db_utils import connect_to_milvus
    from utils.llm_tools import create_embeddings

    collections_config = _load_collections_config()
    connect_to_milvus(db_name)  # 与消费方逐字一致的库解析由调用方传入
    logger.info("向量库：%s", db_name or "default")

    embeddings = create_embeddings()
    for name, rows in rows_by_collection.items():
        if name not in collections_config:
            logger.warning("  ⚠ 配置中缺少集合 %s，跳过", name)
            continue
        if not rows:
            logger.warning("  ⚠ 集合 %s 无种子数据，跳过", name)
            continue
        _seed_one_collection(name, collections_config[name], rows, embeddings, overwrite)


# ---------------------------------------------------------------------- #
# 数据源（cleaning / table-op）
# ---------------------------------------------------------------------- #
def _data_operation_example_rows() -> List[Dict]:
    """读取表格操作少样本示例（user_tables / user_query / output）。"""
    path = os.path.join(DEMO_DATA_DIR, "data_operation_examples.csv")
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _tools_description_rows() -> List[Dict]:
    """从 IHR 自身的 @tool 函数实时导出工具描述（tool_name / description / full_description / args）。"""
    import backend.data_processing.table_operation.table_operations as ops
    from backend.data_processing.table_operation.generate_tools_description import (
        get_tools_description,
    )

    tools = [v for v in vars(ops).values() if hasattr(v, "invoke") and hasattr(v, "name")]
    return get_tools_description(tools)


# ---------------------------------------------------------------------- #
# 各域初始化
# ---------------------------------------------------------------------- #
def seed_sql(gen: RecruitmentDataGenerator, args) -> None:
    """SQL 助手：业务表 + table/term/query 三集合。"""
    if not args.skip_business:
        logger.info("-- 业务库（SQLite） --")
        seed_business_tables(gen.business_tables(
            activities_count=args.activities,
            interviewers_count=args.interviewers,
            candidates_count=args.candidates,
        ))
    if not args.skip_vectors:
        logger.info("-- SQL 助手向量集合 --")
        # 与 data_source_node / term_mapping_node / query_example_node 一致：getenv("VECTOR_DB_DATABASE", "")
        _seed_vector_collections(os.getenv("VECTOR_DB_DATABASE", ""), gen.vector_metadata(), args.overwrite)


def seed_cleaning(args) -> None:
    """数据清洗：company_data / school_data 实体标准化集合。"""
    if args.skip_vectors:
        return
    logger.info("-- 数据清洗向量集合 --")
    # 与 data_cleaning/data_processor 一致：getenv("VECTOR_DB_DATA_CLEANING", "default")
    _seed_vector_collections(
        os.getenv("VECTOR_DB_DATA_CLEANING", "default"),
        {"company_data": company_rows(), "school_data": school_rows()},
        args.overwrite,
    )


def seed_table_op(args) -> None:
    """智能表格操作：tools_description / data_operation_examples 集合。"""
    if args.skip_vectors:
        return
    logger.info("-- 表格操作向量集合 --")
    # 与 table_operation_core 一致：getenv("VECTOR_DB_DATABASE", "examples")
    _seed_vector_collections(
        os.getenv("VECTOR_DB_DATABASE", "examples"),
        {
            "tools_description": _tools_description_rows(),
            "data_operation_examples": _data_operation_example_rows(),
        },
        args.overwrite,
    )


# ---------------------------------------------------------------------- #
# 入口
# ---------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="demo 数据一键初始化")
    parser.add_argument("--domain", choices=["all", *DOMAIN_ORDER], default="all",
                        help="选择初始化的域（默认 all）")
    parser.add_argument("--overwrite", action="store_true", help="向量集合已存在时强制重建（upsert）")
    parser.add_argument("--activities", type=int, default=50, help="招聘活动数量（默认 50）")
    parser.add_argument("--interviewers", type=int, default=100, help="面试官数量（默认 100）")
    parser.add_argument("--candidates", type=int, default=500, help="候选人数量（默认 500）")
    parser.add_argument("--skip-business", action="store_true", help="跳过业务库初始化（仅 sql 域）")
    parser.add_argument("--skip-vectors", action="store_true", help="跳过所有向量库初始化")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env()

    selected = DOMAIN_ORDER if args.domain == "all" else [args.domain]
    gen = RecruitmentDataGenerator() if "sql" in selected else None

    failures = 0
    for domain in selected:
        logger.info("=== 初始化域：%s ===", domain)
        try:
            if domain == "sql":
                seed_sql(gen, args)
            elif domain == "cleaning":
                seed_cleaning(args)
            elif domain == "table-op":
                seed_table_op(args)
        except Exception as e:  # noqa: BLE001 - 单域失败需明确提示且不影响其他域
            failures += 1
            logger.error(
                "域 %s 初始化失败：%s\n"
                "（不影响其他域；向量步骤请确认 embedding provider 环境变量已正确配置）",
                domain, e,
            )

    if failures:
        logger.error("=== 完成（%d 个域失败）===", failures)
        return 1
    logger.info("=== 全部完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
