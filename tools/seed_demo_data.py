#!/usr/bin/env python3
"""
SQL 助手 demo 数据一键初始化脚本（开箱即跑）。

迁移自 standalone QueryBot 的 setup 流程，适配到 IntelligentHR 的纯本地零 server 栈：

- 业务库：QueryBot 的 MySQL  →  IHR 的本地 SQLite（``SQLBOT_DB_PATH``，默认
  ``data/sqlbot.db``），经 SQLAlchemy ``df.to_sql`` 建表；
- 向量库：QueryBot 的 Milvus →  IHR 的本地 libSQL（``VECTOR_DB_PATH`` 下的
  ``{VECTOR_DB_DATABASE}.db``），经 ``utils.vector_db_utils`` 建集合并写入；
- 库名解析口径与 SQL 助手节点**逐字一致**，保证「写入的位置」正好是「节点读取的位置」：
  业务库 ``sqlite:///{SQLBOT_DB_PATH}``（见 sql_execution_node / table_structure_node），
  向量库 ``connect_to_milvus(os.getenv("VECTOR_DB_DATABASE", ""))``（见 data_source_node 等）。

产出：
1. 业务库中 3 张招聘表：recruitment_activity_info / recruitment_interviewer_info /
   recruitment_candidate_info（供表结构自省 + SQL 执行）；
2. 向量库中 3 个集合：table_descriptions（选表召回）、term_descriptions（术语映射）、
   query_examples（少样本示例）。

运行：
    python -m tools.seed_demo_data [--overwrite]
                                   [--activities N] [--interviewers N] [--candidates N]
                                   [--skip-business] [--skip-vectors]

说明：
- 业务表为确定性 demo 数据（固定随机种子），每次运行都以 ``replace`` 方式重建，天然幂等；
- 向量集合默认「已存在且非空则跳过」，避免重复灌数；``--overwrite`` 则按 embedding 字段
  原地 upsert；
- 业务库步骤不依赖任何 LLM / 网络；向量库步骤需要已正确配置 embedding provider 环境变量。
"""

import os
import sys
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

logger = logging.getLogger("seed_demo_data")

# 本脚本负责初始化的 SQL 助手相关集合（其余集合属简历/数据清洗等其他功能，不在本脚本范围）
SQL_ASSISTANT_COLLECTIONS = ["table_descriptions", "term_descriptions", "query_examples"]


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
        logger.info("  ⏭  %-20s 已存在且非空，跳过（--overwrite 可强制重建）", name)
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
        logger.info("  ✓ %-20s upsert %d 条（overwrite）", name, len(data))
    else:
        insert_to_milvus(name, data, vectors)
        logger.info("  ✓ %-20s 写入 %d 条", name, len(data))
    return len(data)


def seed_vector_collections(metadata: Dict[str, List[Dict]], overwrite: bool) -> None:
    """初始化 SQL 助手所需的三个向量集合。"""
    from utils.vector_db_utils import connect_to_milvus
    from utils.llm_tools import create_embeddings

    collections_config = _load_collections_config()

    # 与 data_source_node / term_mapping_node / query_example_node 完全一致的库解析
    db_name = os.getenv("VECTOR_DB_DATABASE", "")
    connect_to_milvus(db_name)
    logger.info("向量库：%s", db_name or "default")

    embeddings = create_embeddings()
    for name in SQL_ASSISTANT_COLLECTIONS:
        if name not in collections_config:
            logger.warning("  ⚠ 配置中缺少集合 %s，跳过", name)
            continue
        rows = metadata.get(name, [])
        if not rows:
            logger.warning("  ⚠ 集合 %s 无种子数据，跳过", name)
            continue
        _seed_one_collection(name, collections_config[name], rows, embeddings, overwrite)


# ---------------------------------------------------------------------- #
# 入口
# ---------------------------------------------------------------------- #
def main() -> int:
    parser = argparse.ArgumentParser(description="SQL 助手 demo 数据一键初始化")
    parser.add_argument("--overwrite", action="store_true", help="向量集合已存在时强制重建（upsert）")
    parser.add_argument("--activities", type=int, default=50, help="招聘活动数量（默认 50）")
    parser.add_argument("--interviewers", type=int, default=100, help="面试官数量（默认 100）")
    parser.add_argument("--candidates", type=int, default=500, help="候选人数量（默认 500）")
    parser.add_argument("--skip-business", action="store_true", help="跳过业务库初始化")
    parser.add_argument("--skip-vectors", action="store_true", help="跳过向量库初始化")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    load_env()

    generator = RecruitmentDataGenerator()

    if not args.skip_business:
        logger.info("=== 初始化业务库 ===")
        tables = generator.business_tables(
            activities_count=args.activities,
            interviewers_count=args.interviewers,
            candidates_count=args.candidates,
        )
        seed_business_tables(tables)

    if not args.skip_vectors:
        logger.info("=== 初始化向量库 ===")
        try:
            seed_vector_collections(generator.vector_metadata(), args.overwrite)
        except Exception as e:  # noqa: BLE001 - 向量步骤依赖 embedding 环境，失败需明确提示
            logger.error(
                "向量库初始化失败：%s\n"
                "（业务库若已完成则不受影响；请确认 embedding provider 相关环境变量已正确配置）",
                e,
            )
            return 1

    logger.info("=== 完成 ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
