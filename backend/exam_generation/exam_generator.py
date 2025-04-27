import os
from typing import List, Optional, Dict
from pydantic import BaseModel, Field
from langfuse.callback import CallbackHandler

from utils.llm_tools import init_language_model, LanguageModelChain


class MultipleChoiceQuestion(BaseModel):
    question: str = Field(..., description="考试题目")
    options: List[str] = Field(..., description="选项列表，包含4个选项")
    correct_answer: str = Field(..., description="正确答案，为选项列表中的一个")


class TrueFalseQuestion(BaseModel):
    question: str = Field(..., description="考试题目")
    correct_answer: bool = Field(..., description="正确答案，True 或 False")


class MultipleChoiceExam(BaseModel):
    questions: List[MultipleChoiceQuestion] = Field(..., description="选择题列表")


class TrueFalseExam(BaseModel):
    questions: List[TrueFalseQuestion] = Field(..., description="判断题列表")


SYSTEM_MESSAGE_MULTIPLE_CHOICE = """
你是一位专业的考试题目生成器。你的任务是根据提供的文本材料生成高质量的选择题。请遵循以下原则：

1. 仔细阅读并理解提供的文本材料。
2. 生成指定数量的选择题，每个问题应该直接基于文本材料的内容。
3. 每个问题应有4个选项，其中只有一个是正确答案。
4. 确保问题难度适中，能够考察学习者对材料的理解程度。
5. 问题应该涵盖材料的不同方面，避免重复。
6. 使用清晰、准确的语言表述问题和选项。
7. 避免生成与之前已生成的问题重复或过于相似的问题。

请严格按照指定的JSON格式输出结果。
"""

SYSTEM_MESSAGE_TRUE_FALSE = """
你是一位专业的考试题目生成器。你的任务是根据提供的文本材料生成高质量的判断题。请遵循以下原则：

1. 仔细阅读并理解提供的文本材料。
2. 生成指定数量的判断题，每个问题应该直接基于文本材料的内容。
3. 每个问题的答案只能是 True 或 False。
4. 确保问题难度适中，能够考察学习者对材料的理解程度。
5. 问题应该涵盖材料的不同方面，避免重复。
6. 使用清晰、准确的语言表述问题。
7. 避免生成与之前已生成的问题重复或过于相似的问题。
8. 确保生成的判断题有一定的挑战性，不要过于简单或明显。

请严格按照指定的JSON格式输出结果。
"""

HUMAN_MESSAGE_TEMPLATE = """
请根据以下文本材料生成{num_questions}个{question_type}：

```
{text_content}
```

请确保生成的题目直接基于这段文本，并涵盖其中的关键信息。

以下是之前已经生成的问题，请避免生成重复或过于相似的问题(如果之前没有生成过，请忽略)：

```
{previous_questions}
```

请生成与这些问题不同的新问题。
"""


class ExamGenerator:
    """考试生成器类，用于生成选择题和判断题。"""

    def __init__(self):
        """初始化考试生成器，设置语言模型和问题生成链。"""
        self.language_model = init_language_model(
            temperature=0.7,
            provider=os.getenv("SMART_LLM_PROVIDER"),
            model_name=os.getenv("SMART_LLM_MODEL"),
        )
        self.multiple_choice_chain = LanguageModelChain(
            MultipleChoiceExam,
            SYSTEM_MESSAGE_MULTIPLE_CHOICE,
            HUMAN_MESSAGE_TEMPLATE,
            self.language_model,
        )()
        self.true_false_chain = LanguageModelChain(
            TrueFalseExam,
            SYSTEM_MESSAGE_TRUE_FALSE,
            HUMAN_MESSAGE_TEMPLATE,
            self.language_model,
        )()

    async def generate_questions(
        self,
        text_content: str,
        session_id: str,
        num_questions: int = 5,
        question_type: str = "选择题",
        previous_questions: Optional[str] = None,
    ) -> List[Dict]:
        """
        生成考试问题。

        :param text_content: 用于生成问题的文本内容
        :param session_id: 会话ID，用于追踪
        :param num_questions: 要生成的问题数量
        :param question_type: 问题类型，"选择题"或"判断题"
        :param previous_questions: 之前生成的问题，用于避免重复
        :return: 生成的问题列表
        """
        langfuse_handler = CallbackHandler(
            tags=["exam_generation"],
            session_id=session_id,
            metadata={"step": "generate_questions", "question_type": question_type},
        )

        try:
            chain = (
                self.multiple_choice_chain
                if question_type == "选择题"
                else self.true_false_chain
            )
            result = await chain.ainvoke(
                {
                    "text_content": text_content,
                    "num_questions": num_questions,
                    "question_type": question_type,
                    "previous_questions": previous_questions or "",
                },
                config={"callbacks": [langfuse_handler]},
            )
            return result["questions"]
        except Exception as e:
            print(f"生成问题时出错: {str(e)}")
            return []

    def _format_previous_questions(
        self,
        previous_questions: Optional[List[MultipleChoiceQuestion | TrueFalseQuestion]],
        question_type: str,
    ) -> str:
        """
        格式化之前生成的问题，用于避免重复。

        :param previous_questions: 之前生成的问题列表
        :param question_type: 问题类型
        :return: 格式化后的问题字符串
        """
        if not previous_questions:
            return ""

        formatted_questions = []
        for question in previous_questions:
            if question_type == "选择题" and isinstance(
                question, MultipleChoiceQuestion
            ):
                formatted_questions.append(
                    f"问题: {question.question}\n选项: {', '.join(question.options)}\n正确答案: {question.correct_answer}\n"
                )
            elif question_type == "判断题" and isinstance(question, TrueFalseQuestion):
                formatted_questions.append(
                    f"问题: {question.question}\n正确答案: {'True' if question.correct_answer else 'False'}\n"
                )

        return "\n".join(formatted_questions)


def merge_questions(
    first_round: List[MultipleChoiceQuestion] | List[TrueFalseQuestion],
    second_round: List[MultipleChoiceQuestion] | List[TrueFalseQuestion],
) -> List[MultipleChoiceQuestion] | List[TrueFalseQuestion]:
    """
    合并两轮生成的问题。

    :param first_round: 第一轮生成的问题列表
    :param second_round: 第二轮生成的问题列表
    :return: 合并后的问题列表
    """
    return first_round + second_round
