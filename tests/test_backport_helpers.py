"""回填修复的纯函数回归测试。

- ``build_resume_details_df``（wave4 R1）：简历详情全部获取失败时，必须返回带列名的
  空表；否则 ``pd.DataFrame([])`` 产生无列 DataFrame，后续按 resume_id 合并抛 KeyError。
- ``format_query_examples``（wave5 B1）：少样本示例格式化，无示例时返回占位提示。
"""

from backend.resume_management.recommendation.recommendation_output_generator import (
    build_resume_details_df,
    RESUME_DETAIL_COLUMNS,
)
from backend.sql_assistant.nodes.sql_generation_node import format_query_examples


# ---------------- R1：简历详情空表守卫 ----------------


def test_build_df_all_failed_returns_empty_with_columns():
    df = build_resume_details_df([None, None])
    assert df.empty
    assert list(df.columns) == RESUME_DETAIL_COLUMNS
    # 关键：带 resume_id 列，可安全进行 pd.merge(on="resume_id")
    assert "resume_id" in df.columns


def test_build_df_empty_input():
    df = build_resume_details_df([])
    assert df.empty
    assert list(df.columns) == RESUME_DETAIL_COLUMNS


def test_build_df_filters_none():
    detail = {
        "resume_id": "r1",
        "characteristics": "c",
        "experience": "e",
        "skills_overview": "s",
    }
    df = build_resume_details_df([detail, None])
    assert len(df) == 1
    assert df.iloc[0]["resume_id"] == "r1"


# ---------------- B1：少样本格式化 ----------------


def test_format_examples_empty():
    assert format_query_examples([]) == "暂无相关示例"


def test_format_examples_renders():
    examples = [
        {"query_text": "有多少员工", "query_sql": "SELECT COUNT(*) FROM emp"},
    ]
    out = format_query_examples(examples)
    assert "有多少员工" in out
    assert "SELECT COUNT(*) FROM emp" in out
    assert out.startswith("示例问题:")
