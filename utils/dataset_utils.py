import os
import sys
import logging
import pandas as pd
import json
import glob
from functools import reduce
from typing import List, Optional, Union, Dict

logger = logging.getLogger(__name__)

# sys.path.append(
#     os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# )


def merge_dataframes(
    *dfs: pd.DataFrame,
    dedup_cols: Optional[List[str]] = None,
    dropna_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    合并多个DataFrame，并根据指定列进行去重。在合并之前，删除指定列为空的行。

    Args:
        *dfs: 要合并的DataFrame列表
        dedup_cols: 用于去重的列名列表，如果为None，则使用所有列进行去重
        dropna_cols: 用于删除空值的列名列表，如果为None，则不删除任何空值

    Returns:
        合并后的DataFrame

    Raises:
        ValueError: 如果没有提供DataFrame或dedup_cols中的列不存在
        TypeError: 如果输入不是pandas DataFrame
    """
    if not dfs:
        raise ValueError("至少需要提供一个DataFrame")

    all_cols = reduce(set.union, (set(df.columns) for df in dfs))

    if dropna_cols and not set(dropna_cols).issubset(all_cols):
        raise ValueError("dropna_cols中的列必须存在于至少一个DataFrame中")

    processed_dfs = [
        df.dropna(subset=set(dropna_cols or []).intersection(df.columns))
        for df in dfs
        if isinstance(df, pd.DataFrame)
    ]

    if len(processed_dfs) != len(dfs):
        raise TypeError("所有输入必须是pandas DataFrame")

    merged_df = pd.concat(processed_dfs, ignore_index=True)
    total_rows_before = len(merged_df)

    if dedup_cols:
        if not set(dedup_cols).issubset(merged_df.columns):
            raise ValueError("dedup_cols中的列必须存在于DataFrame中")
        merged_df.drop_duplicates(subset=dedup_cols, inplace=True)
    else:
        merged_df.drop_duplicates(inplace=True)

    total_rows_after = len(merged_df)
    removed_duplicates = total_rows_before - total_rows_after

    summary = {
        "合并的数据集数量": len(dfs),
        "合并后数据集行数": total_rows_after,
        "删除的重复行数": removed_duplicates,
    }

    logger.info("合并结果汇总:")
    for key, value in summary.items():
        logger.info("- %s: %s", key, value)

    return merged_df


def compare_dataframes(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    id_cols: List[str],
    dropna_cols: Optional[List[str]] = None,
    output_mode: str = "combined",
) -> pd.DataFrame:
    """
    比较两个DataFrame，找出它们之间的差异。

    Args:
        df1: 第一个DataFrame
        df2: 第二个DataFrame
        id_cols: 用于唯一标识行的列名列表
        dropna_cols: 用于删除空值的列名列表，如果为None，则不删除任何空值
        output_mode: 控制输出模式，可选值：'combined'（默认），'left_only'，'right_only'

    Returns:
        包含差异的DataFrame

    Raises:
        ValueError: 如果id_cols中的列不存在于两个DataFrame中或output_mode无效
    """
    if not set(id_cols).issubset(df1.columns) or not set(id_cols).issubset(df2.columns):
        raise ValueError("id_cols中的列必须存在于两个DataFrame中")

    if dropna_cols:
        df1 = df1.dropna(subset=set(dropna_cols).intersection(df1.columns))
        df2 = df2.dropna(subset=set(dropna_cols).intersection(df2.columns))

    df1_indexed = df1.set_index(id_cols)
    df2_indexed = df2.set_index(id_cols)

    diff1 = df1_indexed.loc[~df1_indexed.index.isin(df2_indexed.index)].reset_index()
    diff2 = df2_indexed.loc[~df2_indexed.index.isin(df1_indexed.index)].reset_index()

    logger.info("DataFrame 1 中存在但 DataFrame 2 中不存在的行数: %s", len(diff1))
    logger.info("DataFrame 2 中存在但 DataFrame 1 中不存在的行数: %s", len(diff2))

    if output_mode == "combined":
        return pd.concat([diff1, diff2], axis=0, ignore_index=True)
    elif output_mode == "left_only":
        return diff1
    elif output_mode == "right_only":
        return diff2
    else:
        raise ValueError(
            "无效的output_mode。请选择'combined'、'left_only'或'right_only'。"
        )


def batch_load_datasets_with_pattern(
    file_pattern: str, directory: str = ".", file_format: str = "csv"
) -> List[Union[pd.DataFrame, Dict]]:
    """
    读取指定目录下符合给定文件模式的所有文件，并将它们加载到一个列表中。

    Args:
        file_pattern: 用于匹配文件名的模式（不包括文件扩展名）
        directory: 包含文件的目录路径，默认为当前目录('.')
        file_format: 文件格式，可以是'csv'、'xlsx'或'json'，默认为'csv'

    Returns:
        对于CSV和XLSX文件，返回包含所有读取的pandas DataFrame的列表；
        对于JSON文件，返回包含所有JSON对象的列表

    Raises:
        ValueError: 如果指定了不支持的文件格式
        FileNotFoundError: 如果没有找到匹配的文件
    """
    full_pattern = f"{directory}/{file_pattern}.{file_format}"
    matching_files = sorted(glob.glob(full_pattern))

    if not matching_files:
        raise FileNotFoundError(f"未找到匹配的文件: {full_pattern}")

    data_list = []
    total_rows = 0

    for file in matching_files:
        if file_format == "csv":
            df = pd.read_csv(file)
            data_list.append(df)
            total_rows += len(df)
        elif file_format == "xlsx":
            df = pd.read_excel(file)
            data_list.append(df)
            total_rows += len(df)
        elif file_format == "json":
            with open(file, "r", encoding="utf-8-sig") as f:
                json_data = json.load(f)
                data_list.append(json_data)
                total_rows += len(json_data) if isinstance(json_data, list) else 1
        else:
            raise ValueError(f"不支持的文件格式: {file_format}")

    summary = {
        "读取的文件数量": len(matching_files),
        "文件格式": file_format.upper(),
        "总行数/对象数": total_rows,
    }

    logger.info("数据加载汇总:")
    for key, value in summary.items():
        logger.info("- %s: %s", key, value)

    return data_list


def merge_json_datasets(json_datasets: List[Union[Dict, List[Dict]]]) -> List[Dict]:
    """
    将多个JSON数据集合并为一个大列表。

    Args:
        json_datasets: 包含多个JSON对象或JSON对象列表的列表

    Returns:
        合并后的JSON数据列表

    Raises:
        ValueError: 如果输入列表为空或包含不支持的数据类型
    """
    if not json_datasets:
        raise ValueError("输入的JSON数据集列表不能为空")

    merged_json = []
    total_objects = 0

    for dataset in json_datasets:
        if isinstance(dataset, list):
            merged_json.extend(dataset)
            total_objects += len(dataset)
        elif isinstance(dataset, dict):
            merged_json.append(dataset)
            total_objects += 1
        else:
            raise ValueError(
                f"不支持的数据类型: {type(dataset)}。应为 dict 或 list of dict。"
            )

    summary = {
        "合并的数据集数量": len(json_datasets),
        "合并后的总对象数": total_objects,
    }

    logger.info("JSON数据集合并汇总:")
    for key, value in summary.items():
        logger.info("- %s: %s", key, value)

    return merged_json


def save_df_to_csv(df: pd.DataFrame, filename: str) -> str:
    """
    将DataFrame保存为CSV文件。

    Args:
        df: 要保存的DataFrame
        filename: 文件名

    Returns:
        保存的文件名
    """
    filepath = os.path.join("data", "temp", filename)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    return filename


def load_df_from_csv(filename: str) -> pd.DataFrame:
    """
    从CSV文件加载DataFrame。

    Args:
        filename: 要加载的文件名

    Returns:
        加载的DataFrame
    """
    filepath = os.path.join("data", "temp", filename)
    return pd.read_csv(filepath)


def cleanup_temp_files():
    """
    清理临时文件夹中的所有文件。
    """
    temp_dir = os.path.join("data", "temp")
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            logger.error("删除 %s 时出错: %s", file_path, e)
