# backend_demo/ai_research/research_enums.py

from enum import Enum


class ReportType(Enum):
    """报告类型枚举"""

    ResearchReport = "research_report"
    ResourceReport = "resource_report"
    OutlineReport = "outline_report"
    CustomReport = "custom_report"
    DetailedReport = "detailed_report"
    SubtopicReport = "subtopic_report"


class ReportSource(Enum):
    """报告来源枚举"""

    Web = "web"
    Local = "local"
    LangChainDocuments = "langchain_documents"


class Tone(Enum):
    """报告语气枚举"""

    Objective = "客观（公正且不带偏见地呈现事实和发现）"
    Formal = "正式（遵循学术标准，使用严谨的语言和结构）"
    Analytical = "分析性（对数据和理论进行批判性评估和详细审查）"
    Persuasive = "说服性（旨在使受众接受特定观点或论点）"
    Informative = "信息丰富（提供清晰全面的主题相关信息）"
    Explanatory = "解释性（阐明复杂概念和过程）"
    Descriptive = "描述性（详细描绘现象、实验或案例研究）"
    Critical = "批判性（评判研究及其结论的有效性和相关性）"
    Comparative = "比较性（并列不同理论、数据或方法以突出差异和相似之处）"
    Speculative = "推测性（探讨假设、潜在影响或未来研究方向）"
    Reflective = "反思性（思考研究过程和个人洞见或经验）"
    Narrative = "叙事性（通过讲述故事来阐述研究发现或方法）"
    Humorous = "幽默风趣（轻松有趣，使内容更易理解和接受）"
    Optimistic = "乐观积极（强调积极发现和潜在益处）"
    Pessimistic = "谨慎保守（关注局限性、挑战或消极结果）"
