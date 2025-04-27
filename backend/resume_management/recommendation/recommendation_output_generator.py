import pandas as pd
from typing import Dict, List
from utils.dataset_utils import load_df_from_csv, save_df_to_csv
from backend_demo.resume_management.storage.resume_sql_storage import get_full_resume
import asyncio


class RecommendationOutputGenerator:
    """简历推荐输出生成器，负责获取简历详细信息和准备最终输出。"""

    def __init__(self):
        pass

    async def fetch_single_resume_details(self, resume_id: str) -> Dict:
        """
        异步获取单个简历的详细信息。

        Args:
            resume_id (str): 简历ID

        Returns:
            Dict: 包含简历详细信息的字典
        """
        full_resume = await asyncio.to_thread(get_full_resume, resume_id)
        if full_resume:
            return {
                "resume_id": resume_id,
                "characteristics": full_resume.get("characteristics", ""),
                "experience": full_resume.get("experience_summary", ""),
                "skills_overview": full_resume.get("skills_overview", ""),
            }
        return None

    async def fetch_resume_details(
        self, ranked_resume_scores: pd.DataFrame
    ) -> pd.DataFrame:
        """
        异步获取排名后的简历的详细信息。

        Args:
            ranked_resume_scores (pd.DataFrame): 包含排名后的简历得分的DataFrame。

        Returns:
            pd.DataFrame: 包含简历详细信息的DataFrame。

        Raises:
            ValueError: 如果输入的DataFrame为空。
        """
        if ranked_resume_scores is None or ranked_resume_scores.empty:
            raise ValueError("排名后的简历得分数据不能为空。无法获取简历详细信息。")

        top_resume_ids = ranked_resume_scores["resume_id"].tolist()

        tasks = [
            self.fetch_single_resume_details(resume_id) for resume_id in top_resume_ids
        ]
        resume_details = await asyncio.gather(*tasks)
        resume_details = [detail for detail in resume_details if detail is not None]

        resume_details_df = pd.DataFrame(resume_details)

        merged_df = pd.merge(
            ranked_resume_scores, resume_details_df, on="resume_id", how="left"
        )

        print("已获取候选简历的详细信息")
        return merged_df

    async def prepare_final_output(
        self, resume_details: pd.DataFrame, recommendation_reasons: pd.DataFrame
    ) -> pd.DataFrame:
        """
        异步准备最终的推荐输出。

        Args:
            resume_details (pd.DataFrame): 包含简历详细信息的DataFrame。
            recommendation_reasons (pd.DataFrame): 包含推荐理由的DataFrame。

        Returns:
            pd.DataFrame: 包含最终推荐结果的DataFrame。

        Raises:
            ValueError: 如果输入的DataFrame为空。
        """
        if (
            resume_details is None
            or resume_details.empty
            or recommendation_reasons is None
            or recommendation_reasons.empty
        ):
            raise ValueError("简历详细信息或推荐理由不能为空。无法准备最终输出。")

        # 使用 asyncio.to_thread 来异步执行 pandas 操作
        final_recommendations = await asyncio.to_thread(
            pd.merge, resume_details, recommendation_reasons, on="resume_id", how="left"
        )

        columns_order = [
            "resume_id",
            "total_score",
            "characteristics",
            "experience",
            "skills_overview",
            "reason",
        ]
        final_recommendations = final_recommendations[columns_order]

        final_recommendations = await asyncio.to_thread(
            final_recommendations.sort_values, "total_score", ascending=False
        )

        final_recommendations = final_recommendations.rename(
            columns={
                "resume_id": "简历ID",
                "total_score": "总分",
                "characteristics": "个人特征",
                "experience": "工作经验",
                "skills_overview": "技能概览",
                "reason": "推荐理由",
            }
        )

        print("推荐结果生成完毕，即将为您展示")
        return final_recommendations
