# backend/ai_research/research_prompts.py

from datetime import date, datetime, timezone
from backend.ai_research.research_enums import ReportSource, Tone


def auto_agent_instructions():
    return """
    这个任务涉及研究给定的主题，无论其复杂程度如何，也无论是否存在明确的答案。研究由特定的服务器执行，根据其类型和角色定义，每个服务器需要不同的指令。

    智能代理
    服务器根据主题的领域和可用于研究所提供主题的特定服务器名称来确定。智能代理按其专业领域分类，每种服务器类型都与相应的表情符号相关联。

    示例：
    任务: "我应该投资苹果股票吗？"
    回复: 
    {
        "server": "💰 金融分析师",
        "agent_role_prompt": "你是一位经验丰富的AI金融分析师助手。你的主要目标是基于提供的数据和趋势，撰写全面、深刻、公正且条理清晰的金融报告。"
    }
    任务: "转售球鞋可能成为一个有利可图的生意吗？"
    回复: 
    { 
        "server": "📈 商业分析师",
        "agent_role_prompt": "你是一位经验丰富的AI商业分析师助手。你的主要目标是基于提供的商业数据、市场趋势和战略分析，制作全面、有见地、公正且系统性强的商业报告。"
    }
    任务: "特拉维夫有哪些最有趣的景点？"
    回复:
    {
        "server": "🌍 旅游顾问",
        "agent_role_prompt": "你是一位见多识广的AI旅游顾问助手。你的主要任务是为指定地点撰写引人入胜、见解深刻、客观公正且结构良好的旅游报告，包括历史、景点和文化洞察。"
    }
    """


def generate_search_queries_prompt(
    question, parent_query, report_type, max_iterations=5
):
    if report_type == "detailed_report" or report_type == "subtopic_report":
        task = f"{parent_query} - {question}"
    else:
        task = question

    return (
        f'请生成{max_iterations}个谷歌搜索查询，用于在线搜索以形成对以下任务的客观看法："{task}"\n'
        f'你必须以以下格式的字符串列表回应：["查询1", "查询2", "查询3"]。\n'
        f"回复应仅包含列表，且使用中文。"
    )


def generate_report_prompt(
    question: str,
    context,
    report_source: str,
    report_format="apa",
    total_words=1000,
    tone=None,
):
    reference_prompt = ""
    if report_source == ReportSource.Web.value:
        reference_prompt = f"""
            你必须在报告末尾以参考文献的形式列出所有使用的源URL，确保不添加重复的来源，每个来源只列一次。
            每个URL应该是超链接形式：[网站名称](URL地址)
            此外，你必须在报告中引用相关URL时包含超链接：

            例如：作者, A. A. (年份, 月 日). 网页标题. 网站名称. [网站名称](URL地址)
            """
    else:
        reference_prompt = f"""
            你必须在报告末尾以参考文献的形式列出所有使用的源文档名称，确保不添加重复的来源，每个来源只列一次。
        """

    tone_prompt = f"请以{tone.value}的语气撰写报告。" if tone else ""

    return f"""
信息: "{context}"
---
使用上述信息，以详细报告的形式回答以下查询或任务："{question}" --
报告应重点回答查询，结构清晰，信息丰富，深入全面，包含事实和数据（如果有），至少{total_words}字。
你应尽可能详尽地使用所有相关和必要的信息来撰写报告。

请在你的报告中遵循以下所有指南：
- 你必须基于给定信息形成自己的具体有效观点。不要得出笼统无意义的结论。
- 你必须使用markdown语法和{report_format}格式撰写报告。
- 使用客观公正的新闻报道语气。
- 使用{report_format}格式的文内引用，并将其作为markdown超链接放在引用它们的句子或段落末尾，如：([文内引用](url))。
- 不要忘记在报告末尾以{report_format}格式添加参考文献列表，包含完整的URL链接（无超链接）。
- {reference_prompt}
- {tone_prompt}
- 你必须用中文撰写整个报告。

请尽你所能，这对我的职业非常重要。
假设当前日期是{date.today()}。
"""


def generate_resource_report_prompt(
    question, context, report_source, report_format="apa", total_words=1000
):
    reference_prompt = ""
    if report_source == ReportSource.Web.value:
        reference_prompt = "你必须包含所有相关的源URL。"
    else:
        reference_prompt = "你必须在报告末尾以参考文献的形式列出所有使用的源文档名称，确保不添加重复的来源，每个来源只列一次。"

    return f"""
"{context}"

基于上述信息，为以下问题或主题生成一份文献推荐报告："{question}"。报告应详细分析每个推荐资源，解释每个来源如何有助于回答研究问题。
重点关注每个来源的相关性、可靠性和重要性。
确保报告结构清晰、信息丰富、深入详细，并遵循Markdown语法。
在可能的情况下，包括相关事实、数据和数字。
报告至少应有{total_words}字。
{reference_prompt}
你必须用中文撰写整个报告。
    """


def generate_outline_report_prompt(
    question, context, report_source, report_format="apa", total_words=1000
):
    return f"""
"{context}"

使用上述信息，为以下问题或主题生成一份研究报告大纲（使用Markdown语法）："{question}"。大纲应为研究报告提供一个结构清晰的框架，包括主要部分、子部分和要涵盖的关键点。
研究报告应详细、信息丰富、深入，至少{total_words}字。
使用适当的Markdown语法格式化大纲，确保可读性。
你必须用中文撰写整个大纲。
    """


def generate_subtopic_report_prompt(
    current_subtopic,
    existing_headers: list,
    main_topic: str,
    context,
    report_format: str = "apa",
    max_subsections=5,
    total_words=800,
    tone: Tone = Tone.Objective,
) -> str:
    return f"""
"背景信息":
"{context}"

"主题和子主题":
使用最新可用的信息，就主题"{main_topic}"下的子主题"{current_subtopic}"撰写一份详细报告。
子部分数量不得超过{max_subsections}个。

"内容重点":
- 报告应重点回答问题，结构清晰，信息丰富，深入全面，包括事实和数据（如果有）。
- 使用markdown语法，遵循{report_format.upper()}格式。

"结构和格式":
- 由于这份子报告将是更大报告的一部分，只需包含主体内容并分为适当的子主题，无需引言或结论部分。

- 你必须在报告中引用相关URL时包含markdown超链接，例如：

    这是一个示例文本。([网站名称](URL地址))

"现有子主题报告":
- 以下是现有子主题报告及其章节标题列表：

    {existing_headers}。

- 不要使用上述任何标题或相关细节，以避免重复。使用较小的Markdown标题（如H2或H3）来组织内容结构，避免使用最大的标题（H1），因为它将用于更大报告的标题。

"日期":
如有需要，假设当前日期为{datetime.now(timezone.utc).strftime('%Y年%m月%d日')}。

"重要提示!":
- 必须聚焦于主题！必须省略任何与之无关的信息！
- 不得包含任何引言、结论、摘要或参考文献部分。
- 你必须在必要的地方使用markdown语法的超链接（[网站名称](URL地址)）。
- 报告至少应有{total_words}字。
- 在整个报告中使用{tone.value}的语气。
- 你必须用中文撰写整个报告。
"""


def generate_report_introduction_prompt(
    question: str, research_summary: str = ""
) -> str:
    return f"""{research_summary}\n 
    使用上述最新信息，为主题 -- {question} 准备一份详细的报告引言。
    - 引言应简洁明了，结构清晰，信息丰富，使用markdown语法。
    - 由于这个引言将是更大报告的一部分，不要包含通常存在于报告中的任何其他部分。
    - 引言前应有一个适合整个报告的H1标题。
    - 你必须在必要的地方使用markdown语法的超链接（[网站名称](URL地址)）。
    - 你必须用中文撰写整个引言。
    如有需要，假设当前日期为{datetime.now(timezone.utc).strftime('%Y年%m月%d日')}。
"""


def get_report_by_type(report_type: str):
    report_type_mapping = {
        "research_report": generate_report_prompt,
        "resource_report": generate_resource_report_prompt,
        "outline_report": generate_outline_report_prompt,
        "subtopic_report": generate_subtopic_report_prompt,
    }
    return report_type_mapping.get(report_type, generate_report_prompt)


def generate_subtopics_prompt() -> str:
    return """
    基于以下主题：

    {task}

    和研究数据：

    {data}

    - 构建一个子主题列表，这些子主题将成为要生成的报告文档的标题。 
    - 以下是可能的子主题列表：{subtopics}。
    - 不应有任何重复的子主题。
    - 子主题数量不得超过{max_subtopics}个。
    - 最后，按任务对子主题进行排序，使其在详细报告中呈现时具有相关性和意义。

    "重要提示！":
    - 每个子主题必须与主题和提供的研究数据相关！
    - 你必须用中文写出所有子主题。
"""
