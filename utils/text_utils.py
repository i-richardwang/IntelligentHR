import random
from typing import List, Literal, Union, Optional
import pandas as pd
import regex as re


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    清洗DataFrame中所有字符串列的内容，替换特定标点符号为反引号，
    并处理连续字符（移除连续的下划线和空格，只保留一个）。

    Args:
        df (pd.DataFrame): 包含待清洗文本列的DataFrame

    Returns:
        pd.DataFrame: 包含清洗后文本列的DataFrame副本

    Raises:
        ValueError: 如果输入不是pandas DataFrame
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame类型")

    def _clean_text(text: Union[str, float]) -> str:
        if pd.isna(text):
            return ""
        text = str(text)
        text = re.sub(r"['''\"" "{}]", "`", text)
        text = re.sub(r"_{2,}", "_", text)
        text = re.sub(r"\s{2,}", " ", text)
        return text

    cleaned_df = df.copy()
    for col in cleaned_df.select_dtypes(include=["object"]):
        cleaned_df[col] = cleaned_df[col].apply(_clean_text)

    return cleaned_df


def clean_text(text: str) -> str:
    """
    清洗文本内容，将特定标点符号替换为反引号。

    Args:
        text (str): 需要清洗的文本内容

    Returns:
        str: 清洗后的文本内容
    """
    return re.sub(r"['''\"" "{}]", "`", text) if isinstance(text, str) else ""


def count_chars_and_words(text: str) -> int:
    """
    统计文本中的中文字符和英文单词数。

    Args:
        text (str): 要统计的文本内容

    Returns:
        int: 中文字符和英文单词的总数
    """
    chinese_char_count = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_word_count = len(re.findall(r"\b[a-zA-Z]+\b", text))
    return chinese_char_count + english_word_count


def add_text_count_column(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    为DataFrame添加一个新列，计算指定列的中文字符和英文单词数。

    Args:
        df (pd.DataFrame): 输入的DataFrame
        text_col (str): 要计算文本数的列名

    Returns:
        pd.DataFrame: 添加了新列的DataFrame

    Raises:
        ValueError: 如果输入不是pandas DataFrame或指定的列不存在
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame类型")
    if text_col not in df.columns:
        raise ValueError(f"指定的列 '{text_col}' 在DataFrame中不存在")

    new_column_name = f"{text_col}_text_count"
    df[new_column_name] = df[text_col].apply(count_chars_and_words)
    return df


def filter_invalid_text(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    过滤数据集中指定列为非有效文本的行。

    Args:
        df (pd.DataFrame): 输入的数据集
        text_col (str): 要检查的文本列名

    Returns:
        pd.DataFrame: 过滤后的数据集

    Raises:
        ValueError: 如果输入不是pandas DataFrame或指定的列不存在
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame类型")
    if text_col not in df.columns:
        raise ValueError(f"指定的列 '{text_col}' 在DataFrame中不存在")

    def is_valid_text(text):
        if pd.isna(text):
            return False
        text = str(text).strip()
        if (
            not text
            or re.match(r"^[\s\p{P}]+$", text, re.UNICODE)
            or text.isdigit()
            or len(set(text)) == 1
            or re.match(r"^(.)\1*(?:(.)\2*){0,2}$", text)
            or not re.search(r"[\p{L}\p{N}]", text, re.UNICODE)
        ):
            return False
        return True

    return df[df[text_col].apply(is_valid_text)]


def check_language_type(df: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """
    检查数据集中指定文本字段的内容是否为中文，并添加一个标记列。

    Args:
        df (pd.DataFrame): 包含文本字段的数据集
        text_col (str): 需要检查的文本字段名称

    Returns:
        pd.DataFrame: 返回原数据集，增加一个标记列 'is_chinese'，表示文本是否为中文

    Raises:
        ValueError: 如果输入不是pandas DataFrame或指定的列不存在
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame类型")
    if text_col not in df.columns:
        raise ValueError(f"指定的列 '{text_col}' 在DataFrame中不存在")

    def contains_chinese(text):
        if isinstance(text, str):
            chinese_chars = re.findall("[\u4e00-\u9fff]", text)
            return len(chinese_chars) / len(text) >= 0.5 if text else False
        return False

    df["is_chinese"] = df[text_col].apply(contains_chinese)
    return df


def dataframe_to_markdown_tables(
    df: pd.DataFrame,
    cols: List[str],
    rows_per_table: int = 20,
    nan_drop_method: Literal["any", "all"] = "any",
    output_format: Literal["list", "dataframe"] = "list",
) -> Union[List[str], pd.DataFrame]:
    """
    将输入的DataFrame转换为批量的Markdown表格。

    Args:
        df (pd.DataFrame): 输入的DataFrame
        cols (List[str]): 要包含在Markdown表格中的列
        rows_per_table (int, optional): 每个Markdown表格中的行数。默认为20。
        nan_drop_method (Literal["any", "all"], optional): 删除包含NaN值行的方法。默认为"any"。
        output_format (Literal["list", "dataframe"], optional): 输出格式。默认为"list"。

    Returns:
        Union[List[str], pd.DataFrame]: Markdown表格列表或包含Markdown表格的DataFrame

    Raises:
        ValueError: 如果输入参数无效
    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame类型")
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("cols必须是有效的列名列表")
    if not isinstance(rows_per_table, int) or rows_per_table <= 0:
        raise ValueError("rows_per_table必须是正整数")
    if nan_drop_method not in ["any", "all"]:
        raise ValueError("nan_drop_method必须是'any'或'all'")
    if output_format not in ["list", "dataframe"]:
        raise ValueError("output_format必须是'list'或'dataframe'")

    cleaned_df = df.dropna(subset=cols, how=nan_drop_method)
    text_cols = [col for col in cols if cleaned_df[col].dtype == "object"]
    for col in text_cols:
        cleaned_df[col] = cleaned_df[col].str.replace("\r\n|\n", " ", regex=True)

    table_data = cleaned_df[cols]
    markdown_tables = []

    for start_index in range(0, len(table_data), rows_per_table):
        table_subset = table_data.iloc[start_index : start_index + rows_per_table]
        table_header = "| " + " | ".join(cols) + " |\n"
        table_separator = "| " + " | ".join(["---"] * len(cols)) + " |\n"
        table_rows = "".join(
            f"| {' | '.join(map(str, row))} |\n" for _, row in table_subset.iterrows()
        )
        markdown_tables.append(table_header + table_separator + table_rows)

    return (
        markdown_tables
        if output_format == "list"
        else pd.DataFrame(markdown_tables, columns=["Markdown Table"])
    )


def dataframe_to_batched_texts(
    df: pd.DataFrame,
    cols: List[str],
    chars_per_batch: int = 1000,
    random_seed: Optional[int] = None,
    drop_last_batch: bool = False,
) -> List[List[str]]:
    """
    预处理文本列并将文本按字符限制分批。

    Args:
        df (pd.DataFrame): 输入的DataFrame
        cols (List[str]): 要预处理和分批的列
        chars_per_batch (int, optional): 每批次的最大字符数。默认为1000。
        random_seed (Optional[int], optional): 随机种子。默认为None。
        drop_last_batch (bool, optional): 是否删除最后一个不完整的批次。默认为False。

    Returns:
        List[List[str]]: 批次列表，每个批次包含预处理后的文本列表

    Raises:
        ValueError: 如果输入参数无效
    """
    # 参数验证
    if not isinstance(df, pd.DataFrame):
        raise ValueError("输入必须是pandas DataFrame类型")
    if not isinstance(cols, list) or not all(col in df.columns for col in cols):
        raise ValueError("cols必须是有效的列名列表")
    if not isinstance(chars_per_batch, int) or chars_per_batch <= 0:
        raise ValueError("chars_per_batch必须是正整数")
    if random_seed is not None and not isinstance(random_seed, int):
        raise ValueError("random_seed必须是整数或None")
    if not isinstance(drop_last_batch, bool):
        raise ValueError("drop_last_batch必须是布尔类型")

    # 设置随机种子
    if random_seed is not None:
        random.seed(random_seed)

    # 预处理文本（pandas 3.0 移除了 DataFrame.applymap，改用等价的 DataFrame.map）
    df[cols] = df[cols].map(clean_text)
    all_texts = df[cols].values.flatten().tolist()
    random.shuffle(all_texts)

    # 分批处理文本
    batched_texts = []
    current_batch = []
    current_batch_char_count = 0

    for text in all_texts:
        text_char_count = count_chars_and_words(text)
        if (
            current_batch_char_count + text_char_count > chars_per_batch
            and current_batch
        ):
            batched_texts.append(current_batch)
            current_batch = []
            current_batch_char_count = 0
        current_batch.append(text)
        current_batch_char_count += text_char_count

    # 处理最后一个批次
    if current_batch and not drop_last_batch:
        batched_texts.append(current_batch)

    return batched_texts
