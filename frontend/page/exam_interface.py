import streamlit as st
import os
import sys
import asyncio
from typing import Dict, List
import uuid

# 添加项目根目录到 Python 路径
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(project_root)

from frontend.ui_components import show_sidebar, show_footer, apply_common_styles
from backend.exam_generation.exam_generator import ExamGenerator, merge_questions

st.query_params.role = st.session_state.role

# 应用自定义样式
apply_common_styles()


# 初始化 session state
def initialize_session_state():
    """初始化会话状态变量"""
    if "exam_questions" not in st.session_state:
        st.session_state.exam_questions = {"选择题": [], "判断题": []}
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = {}
    if "score" not in st.session_state:
        st.session_state.score = None
    if "generation_complete" not in st.session_state:
        st.session_state.generation_complete = False


def display_info_message():
    """显示系统信息消息"""
    st.info(
        """
        智能考试系统可以根据输入的文本材料自动生成考试题目。
        您可以自定义选择题和判断题的数量（0-20题，5的倍数）。
        系统会避免不同题型之间的考点重复。
        完成答题后，系统将自动评分并显示结果。
        """
    )


async def generate_exam_questions(
    text_content: str, num_multiple_choice: int, num_true_false: int
) -> Dict[str, List[Dict]]:
    """
    生成考试题目

    :param text_content: 用于生成问题的文本内容
    :param num_multiple_choice: 选择题数量
    :param num_true_false: 判断题数量
    :return: 生成的考试题目字典
    """
    generator = ExamGenerator()
    session_id = str(uuid.uuid4())
    all_questions = {"选择题": [], "判断题": []}

    async def generate_questions_by_type(
        question_type: str, num_questions: int, previous_questions: str
    ):
        questions = []
        for i in range(0, num_questions, 5):
            batch_size = min(5, num_questions - i)
            batch_questions = await generator.generate_questions(
                text_content,
                session_id,
                num_questions=batch_size,
                question_type=question_type,
                previous_questions=previous_questions,
            )
            questions.extend(batch_questions)
            # 更新previous_questions，包含新生成的问题
            previous_questions = format_questions_for_prompt(
                questions + all_questions["选择题"] + all_questions["判断题"]
            )
        return questions

    # 首先生成选择题
    if num_multiple_choice > 0:
        all_questions["选择题"] = await generate_questions_by_type(
            "选择题", num_multiple_choice, ""
        )

    # 然后生成判断题，考虑到已生成的选择题
    if num_true_false > 0:
        previous_questions = format_questions_for_prompt(all_questions["选择题"])
        all_questions["判断题"] = await generate_questions_by_type(
            "判断题", num_true_false, previous_questions
        )

    return all_questions


def format_questions_for_prompt(questions: List[Dict]) -> str:
    """
    格式化问题列表为提示字符串

    :param questions: 问题列表
    :return: 格式化后的问题字符串
    """
    formatted_questions = []
    for question in questions:
        if "options" in question:  # 选择题
            formatted_questions.append(
                f"问题: {question['question']}\n选项: {', '.join(question['options'])}\n正确答案: {question['correct_answer']}\n"
            )
        else:  # 判断题
            formatted_questions.append(
                f"问题: {question['question']}\n正确答案: {'True' if question['correct_answer'] else 'False'}\n"
            )
    return "\n".join(formatted_questions)


def display_exam_questions():
    """显示考试题目并收集用户答案"""
    st.subheader("考试题目")
    with st.form("exam_form"):
        question_index = 1
        for question_type in ["选择题", "判断题"]:
            if st.session_state.exam_questions[question_type]:
                st.write(f"**{question_type}**")
                for question in st.session_state.exam_questions[question_type]:
                    st.write(f"**问题 {question_index}:** {question['question']}")
                    if question_type == "选择题":
                        options = question["options"]
                        st.session_state.user_answers[question_index] = st.radio(
                            f"选择答案 {question_index}",
                            options,
                            key=f"q_{question_index}",
                        )
                    else:  # 判断题
                        st.session_state.user_answers[question_index] = st.radio(
                            f"选择答案 {question_index}",
                            ["True", "False"],
                            key=f"q_{question_index}",
                        )
                    st.markdown("---")
                    question_index += 1

        submit_exam = st.form_submit_button("提交答案")

    if submit_exam:
        calculate_score()


def calculate_score():
    """计算考试分数"""
    correct_count = 0
    total_questions = 0
    for question_type, questions in st.session_state.exam_questions.items():
        for i, question in enumerate(questions, total_questions + 1):
            user_answer = st.session_state.user_answers[i]
            correct_answer = question["correct_answer"]

            if question_type == "判断题":
                user_answer = user_answer.lower() == "true"

            if user_answer == correct_answer:
                correct_count += 1
        total_questions += len(questions)

    st.session_state.score = (
        (correct_count / total_questions) * 100 if total_questions > 0 else 0
    )


def display_score():
    """显示考试结果"""
    st.subheader("考试结果")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("总分", f"{int(st.session_state.score)}")
    with col2:
        total_questions = sum(
            len(questions) for questions in st.session_state.exam_questions.values()
        )
        correct_count = int(st.session_state.score * total_questions / 100)
        st.metric("正确率", f"{correct_count}/{total_questions}")

    for question_type, questions in st.session_state.exam_questions.items():
        st.markdown(f"### {question_type}")
        for i, question in enumerate(questions, 1):
            with st.container(border=True):
                st.markdown(f"**问题 {i}:** {question['question']}")
                user_answer = st.session_state.user_answers[i]
                correct_answer = question["correct_answer"]

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"您的答案: **{user_answer}**")
                with col2:
                    st.markdown(f"正确答案: **{correct_answer}**")

                if (question_type == "选择题" and user_answer == correct_answer) or (
                    question_type == "判断题"
                    and user_answer.lower() == str(correct_answer).lower()
                ):
                    st.success("回答正确！")
                else:
                    st.error("回答错误。")


def main():
    """主函数，控制整个应用的流程"""
    st.title("📝 智能考试系统")
    st.markdown("---")

    # 显示侧边栏
    show_sidebar()

    # 初始化会话状态
    initialize_session_state()

    # 显示系统信息
    display_info_message()

    if not st.session_state.generation_complete:
        with st.form("exam_generation_form"):
            text_content = st.text_area("请输入文本材料", height=200)
            col1, col2 = st.columns(2)
            with col1:
                num_multiple_choice = st.number_input(
                    "选择题数量", min_value=0, max_value=20, value=5, step=5
                )
            with col2:
                num_true_false = st.number_input(
                    "判断题数量", min_value=0, max_value=20, value=5, step=5
                )
            submit_button = st.form_submit_button("生成考试题目")

        if submit_button and text_content:
            if num_multiple_choice + num_true_false == 0:
                st.error("请至少选择一种题型并设置题目数量。")
            else:
                with st.spinner("正在生成考试题目..."):
                    st.session_state.exam_questions = asyncio.run(
                        generate_exam_questions(
                            text_content, num_multiple_choice, num_true_false
                        )
                    )
                    st.session_state.user_answers = {}
                    st.session_state.score = None
                    st.session_state.generation_complete = True
                st.rerun()
    else:
        display_exam_questions()

    if st.session_state.score is not None:
        display_score()

    # 显示页脚
    show_footer()


main()
