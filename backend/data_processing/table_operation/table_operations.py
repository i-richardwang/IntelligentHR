import pandas as pd
from langchain_core.tools import tool
from typing import Annotated, List, Dict, Union, Any, Optional, Tuple

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@tool
def join_dataframes(
    left_df: Annotated[Any, "Left dataframe for the join operation"],
    right_df: Annotated[Any, "Right dataframe to be joined with the left dataframe"],
    how: Annotated[
        str,
        "Join method: 'left', or 'right'",
    ],
    left_on: Annotated[
        Union[str, List[str]],
        "Column name(s) from the left dataframe to use as join key(s)",
    ],
    right_on: Annotated[
        Union[str, List[str]],
        "Column name(s) from the right dataframe to use as join key(s)",
    ],
    left_columns: Annotated[
        Optional[List[str]],
        "List of column names from the left dataframe to include in the result, if None all columns are included",
    ] = None,
    right_columns: Annotated[
        Optional[List[str]],
        "List of column names from the right dataframe to include in the result, if None all columns are included",
    ] = None,
) -> pd.DataFrame:
    """
    Joins two dataframes based on specified columns, similar to SQL JOIN operations.

    Use cases:
    - Combining two related datasets based on common information (e.g., ID or date).
    - Enriching a primary dataset with additional information from a secondary dataset.

    Notes:
    - Ensure that the columns used for join operation exist in both dataframes.
    - For left_columns and right_columns parameters:
      * If not explicitly specified by the user, all columns will be included by default.

    Output:
    - Returns a pandas DataFrame containing the result of the join operation.
    """
    try:
        # Ensure left_on and right_on are lists
        left_on = [left_on] if isinstance(left_on, str) else left_on
        right_on = [right_on] if isinstance(right_on, str) else right_on

        # Ensure columns used for merging are included in left_columns and right_columns
        if left_columns is not None:
            left_columns = list(set(left_columns + left_on))
        if right_columns is not None:
            right_columns = list(set(right_columns + right_on))

        # Select columns to include
        if left_columns is not None:
            left_df = left_df[left_columns]
        if right_columns is not None:
            right_df = right_df[right_columns]

        # Perform merge
        merged_df = pd.merge(
            left_df, right_df, how=how, left_on=left_on, right_on=right_on
        )
        logger.info(f"Merged dataframes successfully. Result shape: {merged_df.shape}")

        return merged_df
    except Exception as e:
        logger.error(f"Error in merge_dataframes: {str(e)}")
        raise


@tool
def reshape_wide_to_long(
    df: Annotated[
        Any,
        "Input wide-format dataframe where each row represents a unique entity with data for multiple time points or categories spread across different columns",
    ],
    columns_to_compress: Annotated[
        List[str],
        "List of column names to be compressed into long format, these columns will be converted into values in a single column",
    ],
    new_column_for_old_headers: Annotated[
        str,
        "Name of the new column that will store the original column names (as identifiers)",
    ],
    new_column_for_values: Annotated[
        str,
        "Name of the new column that will store the values from the original columns",
    ],
) -> pd.DataFrame:
    """
    Reshapes a wide-format dataframe into a long format. This process is often referred to as "melting" or "unpivoting".

    Use cases:
    - Converting wide-format data to long format for time series analysis or longitudinal data analysis.
    - Preparing data for certain types of statistical analyses or visualizations that require long-format data.
    - Consolidating multiple category columns into a single category column and a value column for easier analysis.

    Notes:
    - Ensure that all columns listed in columns_to_compress exist in the dataframe.
    - For new_column_for_old_headers and new_column_for_values parameters:
      * If the user doesn't explicitly specify names for these new columns, you can generate appropriate names based on the context of the data.

    Output:
    - Returns a pandas DataFrame representing the reshaped long-format data.
    """
    try:
        # Automatically determine columns to keep
        columns_to_keep = [col for col in df.columns if col not in columns_to_compress]

        # Perform wide to long reshaping
        long_df = pd.melt(
            df,
            id_vars=columns_to_keep,
            value_vars=columns_to_compress,
            var_name=new_column_for_old_headers,
            value_name=new_column_for_values,
        )

        # Reset index
        long_df = long_df.reset_index(drop=True)
        logger.info(
            f"Reshaped dataframe from wide to long format. New shape: {long_df.shape}"
        )

        return long_df
    except Exception as e:
        logger.error(f"Error in reshape_wide_to_long: {str(e)}")
        raise


@tool
def reshape_long_to_wide(
    df: Annotated[
        Any,
        "Input long-format dataframe where one column contains values that will become new column names, and another column contains the values for these new columns",
    ],
    column_to_use_as_headers: Annotated[
        str, "Name of the column containing values that will become new column names"
    ],
    column_with_values: Annotated[
        str,
        "Name of the column containing the values that will populate the new columns",
    ],
    aggfunc: Annotated[
        str,
        "Aggregation function to use when there are duplicate values, e.g., 'first', 'last', 'mean', 'sum', etc.",
    ] = "first",
) -> pd.DataFrame:
    """
    Reshapes a long-format dataframe into a wide format. This process is often referred to as "pivoting" or "casting".

    Use cases:
    - Converting time series data from long format to wide format with each time point as a separate column.
    - Creating cross-tabulations or pivot tables for data summarization and reporting.
    - Preparing data for specific types of analyses or visualizations that require wide-format data.

    Notes:
    - Ensure that column_to_use_as_headers and column_with_values exist in the dataframe.

    Output:
    - Returns a pandas DataFrame representing the reshaped wide-format data.
    """
    try:
        # Automatically determine identifier columns
        id_columns = [
            col
            for col in df.columns
            if col not in [column_to_use_as_headers, column_with_values]
        ]

        # Perform long to wide reshaping
        wide_df = df.pivot_table(
            index=id_columns,
            columns=column_to_use_as_headers,
            values=column_with_values,
            aggfunc=aggfunc,
        )

        # Reset column names, remove MultiIndex
        wide_df.columns.name = None
        wide_df = wide_df.reset_index()
        logger.info(
            f"Reshaped dataframe from long to wide format. New shape: {wide_df.shape}"
        )

        return wide_df
    except Exception as e:
        logger.error(f"Error in reshape_long_to_wide: {str(e)}")
        raise


@tool
def compare_dataframes(
    df1: Annotated[Any, "First dataframe, used as the base for comparison"],
    df2: Annotated[Any, "Second dataframe, to be compared with the first dataframe"],
    key_column_df1: Annotated[
        str, "Name of the key column in df1 used to identify unique records"
    ],
    key_column_df2: Annotated[
        Union[str, None],
        "Name of the key column in df2, if different from df1. If None, assumes same as df1",
    ] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares two dataframes based on specified key columns and identifies the differences between them.

    Use cases:
    - Checking for changes after data updates, such as identifying new or deleted records.
    - Verifying data integrity after migration or transformation processes.
    - Comparing two similar but potentially different datasets, such as snapshots from different time points.

    Output:
    - Returns a tuple of two pandas DataFrames:
      1. The first DataFrame (only_in_df1) contains records that exist only in df1.
      2. The second DataFrame (only_in_df2) contains records that exist only in df2.
    """
    try:
        # If key column for df2 is not specified, use the same as df1
        if key_column_df2 is None:
            key_column_df2 = key_column_df1

        # Ensure both dataframes have the specified key columns
        if key_column_df1 not in df1.columns:
            raise ValueError(
                f"Key column '{key_column_df1}' does not exist in the first dataframe"
            )
        if key_column_df2 not in df2.columns:
            raise ValueError(
                f"Key column '{key_column_df2}' does not exist in the second dataframe"
            )

        # Get sets of unique values in the key columns of both dataframes
        set_df1 = set(df1[key_column_df1])
        set_df2 = set(df2[key_column_df2])

        # Find records only in df1
        only_in_df1 = df1[df1[key_column_df1].isin(set_df1 - set_df2)].reset_index(
            drop=True
        )

        # Find records only in df2
        only_in_df2 = df2[df2[key_column_df2].isin(set_df2 - set_df1)].reset_index(
            drop=True
        )

        logger.info(
            f"Comparison complete. Records only in df1: {len(only_in_df1)}, only in df2: {len(only_in_df2)}"
        )

        return only_in_df1, only_in_df2
    except Exception as e:
        logger.error(f"Error in compare_dataframes: {str(e)}")
        raise


@tool
def stack_dataframes(
    dataframes: Annotated[List[Any], "List of DataFrames to be vertically stacked"],
    equivalent_columns: Annotated[
        Optional[Dict[str, List[str]]],
        "Optional: Dictionary of equivalent column names across different DataFrames",
    ] = None,
) -> pd.DataFrame:
    """
    Vertically concatenates multiple DataFrames, similar to a SQL UNION operation.

    Use cases:
    - Combining data from multiple periods or sources with similar structures, such as employee information tables from different years.

    Note:
    - equivalent_columns:
      Keys are the standard column names, and values are lists of equivalent names.
      This allows the function to identify and standardize columns with different names but the same semantic meaning across DataFrames.
      If not provided or set to None, the function will use the original column names without any renaming.

    Output:
    Returns a pandas DataFrame containing the vertically stacked data from all input DataFrames.
    """
    if len(dataframes) < 2:
        raise ValueError("至少需要两个 DataFrame 来执行叠加操作")

    result_dfs = []

    for df_name, df in dataframes:

        temp_df = df.copy()

        if equivalent_columns:
            for standard_col, equiv_cols in equivalent_columns.items():
                for equiv_col in equiv_cols:
                    if equiv_col in temp_df.columns:
                        temp_df.rename(columns={equiv_col: standard_col}, inplace=True)

        temp_df["source_table"] = df_name

        result_dfs.append(temp_df)

    result_df = pd.concat(result_dfs, ignore_index=True)

    return result_df


@tool
def deduplicate_dataframe(
    df: Annotated[Any, "Input dataframe to be deduplicated"],
    group_by_columns: Annotated[
        Optional[Union[str, List[str]]],
        "Column(s) to group by for deduplication. If None, removes completely duplicate rows",
    ] = None,
    sort_column: Annotated[
        Optional[str],
        "Column to sort by within each group. Required if group_by_columns is specified",
    ] = None,
    ascending: Annotated[
        bool, "Sort in ascending order if True, descending if False"
    ] = False,
) -> pd.DataFrame:
    """
    Deduplicates a dataframe based on specified columns or removes completely duplicate rows.

    Use cases:
    - Removing completely duplicate rows across all columns.
    - Keeping only the most recent record for each employee (e.g., latest promotion).
    - Retaining the record with the highest value for each group (e.g., longest working day per employee per year).

    Notes:
    - If group_by_columns is None, the function removes completely duplicate rows across all columns.
    - If group_by_columns is specified, the function groups the dataframe by these columns, sorts within each group by sort_column,
      and then keeps the first record of each group after sorting.
    - Set 'ascending=False' to keep the record with the highest value in 'sort_column'.
    - Set 'ascending=True' to keep the record with the lowest value in 'sort_column'.

    Output:
    - Returns a pandas DataFrame with duplicates removed according to the specified criteria.
    """
    try:
        if group_by_columns is None:
            # Remove completely duplicate rows
            df_deduplicated = df.drop_duplicates()
            logger.info(
                f"Removed completely duplicate rows. Original shape: {df.shape}, New shape: {df_deduplicated.shape}"
            )
        else:
            # Ensure group_by_columns is a list
            if isinstance(group_by_columns, str):
                group_by_columns = [group_by_columns]

            if sort_column is None:
                raise ValueError(
                    "sort_column must be specified when group_by_columns is provided"
                )

            # Sort the dataframe and drop duplicates
            df_deduplicated = df.sort_values(
                by=group_by_columns + [sort_column], ascending=ascending
            ).drop_duplicates(subset=group_by_columns, keep="first")

            logger.info(
                f"Deduplicated dataframe based on columns {group_by_columns}. Original shape: {df.shape}, New shape: {df_deduplicated.shape}"
            )

        # Reset index
        df_deduplicated = df_deduplicated.reset_index(drop=True)

        return df_deduplicated

    except Exception as e:
        logger.error(f"Error in deduplicate_dataframe: {str(e)}")
        raise
