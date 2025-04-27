import os
from typing import Literal
from pydantic import BaseModel, Field
from utils.llm_tools import LanguageModelChain, init_language_model
from langfuse.callback import CallbackHandler

# 初始化语言模型
language_model = init_language_model(
    provider=os.getenv("SMART_LLM_PROVIDER"), model_name=os.getenv("SMART_LLM_MODEL")
)


class ResumeComparisonResult(BaseModel):
    is_same_candidate: bool = Field(..., description="判断是否为同一候选人")
    latest_version: Literal["uploaded_resume", "existing_resume"] = Field(
        ..., description="如果是同一候选人，指出哪份简历是最新版本：'uploaded_resume' 或 'existing_resume'"
    )
    explanation: str = Field(..., description="解释判断的理由")


SYSTEM_MESSAGE = """
你是一个专业的简历分析AI助手。你的任务是比较两份简历,判断它们是否属于同一个候选人,如果是,则进一步判断哪一份是最新版本。

请遵循以下步骤:
1. 仔细分析两份简历的内容,关注关键信息如姓名、联系方式、教育背景、工作经历等。
2. 判断两份简历是否属于同一个候选人。
3. 如果是同一个候选人,比较两份简历的内容,明确指出最新版本是 'uploaded_resume' 还是 'existing_resume'。
5. 提供详细的解释,说明你的判断依据。

在分析过程中,请考虑以下因素:
- 个人信息的一致性(姓名、联系方式等)
- 最后一段工作经历和项目经历的起始时间比较，起始时间越晚，简历越新

请保持客观,仅基于提供的信息做出判断。如果信息不足以做出确定的判断,请在解释中说明。
"""

HUMAN_MESSAGE_TEMPLATE = """
请比较以下两份简历,并判断它们是否属于同一个候选人。如果是同一个候选人,请进一步判断哪一份是最新版本。

**上传的简历内容:**

```
start of uploaded resume

{uploaded_resume}

end of uploaded resume
```


**数据库中已存在的简历内容:**

```
start of existing resume

{existing_resume}

end of existing resume
```


请提供你的分析结果,包括是否为同一候选人,哪一份是最新版本(如果适用),以及详细的解释。
"""

resume_comparison_chain = LanguageModelChain(
    ResumeComparisonResult, SYSTEM_MESSAGE, HUMAN_MESSAGE_TEMPLATE, language_model
)()


def create_langfuse_handler(session_id: str, step: str) -> CallbackHandler:
    return CallbackHandler(
        tags=["resume_comparison"], session_id=session_id, metadata={"step": step}
    )


async def compare_resumes(
    uploaded_resume: str, existing_resume: str, session_id: str
) -> ResumeComparisonResult:
    langfuse_handler = create_langfuse_handler(session_id, "compare_resumes")

    result = await resume_comparison_chain.ainvoke(
        {"uploaded_resume": uploaded_resume, "existing_resume": existing_resume},
        config={"callbacks": [langfuse_handler]},
    )

    return result
