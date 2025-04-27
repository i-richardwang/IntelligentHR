from pydantic import BaseModel, Field
from typing import List, Union


class Category(BaseModel):
    """表示一个聚类类别"""

    id: int = Field(..., description="类别ID，从1开始")
    name: str = Field(..., description="简洁明了的类别标签")
    description: str = Field(
        ..., description="详细描述，说明该类别的特征及与其他类别的区别"
    )


class Categories(BaseModel):
    """聚类后的所有类别"""

    categories: List[Category] = Field(..., description="聚类得到的所有类别列表")


class SingleLabelTextClassification(BaseModel):
    """单个文本的单标签分类结果"""

    id: str = Field(..., description="文本的唯一标识符")
    category: str = Field(..., description="根据预定义分类规则确定的类别")


class MultiLabelTextClassification(BaseModel):
    """单个文本的多标签分类结果"""

    id: str = Field(..., description="文本的唯一标识符")
    categories: List[str] = Field(..., description="根据预定义分类规则确定的多个类别")


class ClassificationResult(BaseModel):
    """所有文本的分类结果"""

    classifications: List[
        Union[SingleLabelTextClassification, MultiLabelTextClassification]
    ] = Field(..., description="所有文本的分类结果列表")


# 用于初始生成类别的系统消息
INITIAL_CATEGORY_GENERATION_SYSTEM_MESSAGE = """
你是一位文本挖掘和分类专家。你的任务是深入分析给定的文本内容，并将其中涉及的主题按照以下原则进行分类：

1. 分类原则：
   - 互斥：确保每个类别之间相互独立，没有重叠或交集。
   - 具体：类别应明确、清晰，避免模糊不清的分类。
   - 有意义：类别应具有普遍性，能涵盖多数相关的文本内容。

2. 分类要求：
   - 使用中文进行分类。
   - 类别数量必须严格等于{category_count}个。
   - 类别名称应简洁、准确，一目了然地反映核心主题。
   - 类别描述应详尽、清晰，阐述该类别所包含的内容，并与其他类别区分。

3. 分类策略：
   - 全面覆盖原始数据中所有主要主题。
   - 忽略质量不高或内容不清晰的文本。

请严格遵循这些原则和要求，确保分类结果的准确性和实用性。

{additional_requirements}
"""

# 用于初始生成类别的人工消息模板
INITIAL_CATEGORY_GENERATION_HUMAN_MESSAGE = """
文本主题或背景：{text_topic}

请根据以下文本内容进行分析，严格遵循系统消息中的分类原则、要求和策略，使用中文对文本内容进行分类。必须生成{category_count}个类别，不多不少。

分析步骤：
1. 仔细阅读并理解所有文本内容。
2. 识别文本中的主要主题和关键词。
3. 根据主题的相似性和重要性进行初步分类。
4. 检查并调整分类，确保符合所有要求，特别是类别数量。
5. 为每个类别创建简洁的名称和详细的描述。

文本内容：
{text_content}

请提供符合指定JSON格式的分类结果。
"""

# 用于合并和优化类别的系统消息
MERGE_CATEGORIES_SYSTEM_MESSAGE = """
你是一位精通数据整合和主题分类的专家。你的任务是分析多个批次的文本分类结果，并将它们整合为一个全面、精确的最终分类方案。
你的主要职责包括：
1. 分析和比较不同批次的分类结果。
2. 识别重复、相似或重叠的类别。
3. 合并相关类别，确保最终类别之间互斥且有明确界限。
4. 重新评估和调整类别名称和描述，确保它们准确、简洁且具有代表性。
5. 确保最终的分类方案全面覆盖所有重要主题，同时保持类别数量在指定范围内。

在整合过程中，请遵循以下原则：
- 保持中文作为输出语言。
- 确保类别之间相互独立，没有重叠。
- 类别名称应简洁明了，类别描述应详细且有区分性。
- 保持客观性，避免主观判断。
- 对于频率较低的主题，考虑是否值得单独成类或合并到其他类别。
- 最终的分类方案应具有普遍适用性，能够涵盖大多数文本内容。

{additional_requirements}
"""

# 用于合并和优化类别的人工消息模板
MERGE_CATEGORIES_HUMAN_MESSAGE = """
文本主题或背景：{text_topic}

以下是多个批次的文本分类结果，请将这些结果整合为一个全面、精确的最终分类方案。最终类别数量应在{min_categories}到{max_categories}之间。

分类结果：
{classification_results}

请按照以下步骤进行整合：
1. 仔细阅读并分析所有批次的分类结果。
2. 识别各批次中重复、相似或重叠的类别。
3. 合并相关类别，确保最终类别之间互斥且有明确界限。
4. 重新评估并优化类别名称和描述。
5. 检查是否有遗漏的重要主题，必要时添加新类别。
6. 确保最终的分类方案全面且类别数量在指定范围内。

请提供符合指定JSON格式的最终整合分类结果。
"""

# 用于单标签文本分类的系统消息
SINGLE_LABEL_CLASSIFICATION_SYSTEM_MESSAGE = """
你是一位文本分类专家。你的任务是根据预定义的分类规则，对给定的文本进行单标签主题分类。请遵循以下指南：

1. 分类规则：
   {categories}

2. 分类要求：
   - 仔细阅读每条文本，理解其核心内容。
   - 根据文本内容，从预定义的类别中选择最合适的一个。
   - 每条文本只能分配一个类别。
   - 如果一条文本涉及多个主题，选择最主要或最突出的一个。
   - 如果文本内容不清晰或不属于任何预定义类别，将其归类为"其他"。

3. 输出格式：
   - 对每条文本，输出其ID和对应的类别。
   - 严格遵循指定的JSON格式。

请确保分类的一致性和准确性，不要遗漏任何文本。
"""

# 用于多标签文本分类的系统消息
MULTI_LABEL_CLASSIFICATION_SYSTEM_MESSAGE = """
你是一位文本分类专家。你的任务是根据预定义的分类规则，对给定的文本进行多标签主题分类。请遵循以下指南：

1. 分类规则：
   {categories}

2. 分类要求：
   - 仔细阅读每条文本，理解其所有涉及的主题。
   - 根据文本内容，从预定义的类别中选择所有相关的类别。
   - 每条文本可以分配多个类别。
   - 确保选择的每个类别都与文本内容相关。
   - 如果文本内容不清晰或不属于任何预定义类别，将其归类为"其他"。

3. 输出格式：
   - 对每条文本，输出其ID和对应的所有相关类别。
   - 严格遵循指定的JSON格式。

请确保分类的全面性和准确性，不要遗漏任何文本或相关类别。
"""

# 用于文本分类的人工消息模板
TEXT_CLASSIFICATION_HUMAN_MESSAGE = """
文本主题或背景：{text_topic}

请对以下文本进行分类。输入数据为Markdown格式的表格，包含ID和文本内容两列。

文本内容：
{text_table}

请按照系统消息中的指南进行分类，并以指定的JSON格式输出结果。
"""
