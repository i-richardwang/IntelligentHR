# backend_demo/ai_research/ai_research_agent.py

import json
import asyncio
from typing import List, Dict, Tuple, Any, Optional
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_fixed

from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI

from backend_demo.ai_research.ai_research_config import Config
from backend_demo.ai_research.research_prompts import (
    auto_agent_instructions,
    generate_search_queries_prompt,
    get_report_by_type,
    generate_report_introduction_prompt,
    generate_subtopics_prompt,
)
from backend_demo.ai_research.web_retriever import (
    get_retriever,
    get_default_retriever,
    scrape_urls,
    ContextCompressor,
)
from backend_demo.ai_research.research_enums import ReportType, ReportSource, Tone
from backend_demo.ai_research.embedding_service import Memory
from utils.llm_tools import init_language_model


class AgentResponse(BaseModel):
    """AI代理响应模型"""

    server: str = Field(
        ..., description="由主题领域确定的服务器类型，与相应的表情符号关联。"
    )
    agent_role_prompt: str = Field(
        ..., description="基于代理角色和专业知识的特定指令。"
    )


class Subtopic(BaseModel):
    """子主题模型"""

    task: str = Field(description="任务名称", min_length=1)


class Subtopics(BaseModel):
    """子主题列表模型"""

    subtopics: List[Subtopic] = []


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def choose_agent(
    query: str, cfg: Config, parent_query: Optional[str] = None
) -> Tuple[str, str]:
    """
    选择适合查询的AI代理

    :param query: 查询字符串
    :param cfg: 配置对象
    :param parent_query: 可选的父查询
    :return: 服务器类型和代理角色提示
    """
    try:
        language_model = init_language_model(temperature=cfg.temperature)
        chat = language_model

        system_message = "{auto_agent_instructions}"
        human_message = "task: {query}"

        parser = JsonOutputParser(pydantic_object=AgentResponse)

        format_instructions = """
Output your answer as a JSON object that conforms to the following schema:
```json
{schema}
```

Important instructions:
1. Wrap your entire response between ```json and ``` tags.
2. Ensure your JSON is valid and properly formatted.
3. Do not include the schema definition in your answer.
4. Only output the data instance that matches the schema.
5. Do not include any explanations or comments within the JSON output.
        """

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", system_message + format_instructions),
                ("human", human_message),
            ]
        ).partial(
            auto_agent_instructions=auto_agent_instructions(),
            schema=AgentResponse.model_json_schema(),
        )

        chain = prompt_template | chat | parser

        agent_dict = await chain.ainvoke({"query": query})

        return agent_dict["server"], agent_dict["agent_role_prompt"]
    except json.JSONDecodeError:
        print("解析JSON时出错。使用默认代理。")
        return "默认代理", (
            "你是一个AI批判性思维研究助手。你的唯一目的是就给定文本撰写结构良好、"
            "批评性强、客观公正的报告。"
        )


async def get_sub_queries(
    query: str,
    agent_role_prompt: str,
    cfg: Config,
    parent_query: Optional[str],
    report_type: str,
) -> List[str]:
    """
    生成子查询

    :param query: 主查询
    :param agent_role_prompt: 代理角色提示
    :param cfg: 配置对象
    :param parent_query: 父查询
    :param report_type: 报告类型
    :return: 子查询列表
    """
    language_model = init_language_model(temperature=cfg.temperature)
    chat = language_model

    system_message = f"{agent_role_prompt}\n\n"

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            (
                "human",
                generate_search_queries_prompt(
                    query, parent_query, report_type, max_iterations=cfg.max_iterations
                ),
            ),
        ]
    )

    messages = prompt.format_messages(question=query)
    response = await chat.ainvoke(messages)

    try:
        sub_queries = json.loads(response.content)
        return sub_queries
    except json.JSONDecodeError:
        print("解析JSON时出错。返回原始查询。")
        return [query]


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
async def construct_subtopics(
    task: str, data: str, config: Config, subtopics: List[Dict[str, str]] = []
) -> List[Dict[str, str]]:
    """
    构建子主题

    :param task: 任务
    :param data: 研究数据
    :param config: 配置对象
    :param subtopics: 现有子主题列表
    :return: 构建的子主题列表
    """
    try:
        parser = JsonOutputParser(pydantic_object=Subtopics)

        format_instructions = """
输出你的答案为符合以下模式的JSON对象：
```json
{schema}
```

重要说明：
1. 将你的整个响应包裹在 ```json 和 ``` 标签之间。
2. 确保你的JSON是有效的且格式正确。
3. 不要在你的答案中包含模式定义。
4. 只输出与模式匹配的数据实例。
5. 不要在JSON输出中包含任何解释或评论。
        """

        prompt = ChatPromptTemplate.from_template(
            generate_subtopics_prompt() + format_instructions
        ).partial(schema=Subtopics.model_json_schema())

        language_model = init_language_model(temperature=config.temperature)
        chat = language_model

        chain = prompt | chat | parser

        output = await chain.ainvoke(
            {
                "task": task,
                "data": data,
                "subtopics": subtopics,
                "max_subtopics": config.max_subtopics,
            }
        )

        return output["subtopics"]

    except Exception as e:
        print("解析子主题时出现异常：", e)
        return subtopics


async def generate_report(
    query: str,
    context: str,
    agent_role_prompt: str,
    report_type: str,
    tone: Tone,
    report_source: str,
    cfg: Config,
    websocket: Any = None,
    main_topic: str = "",
    existing_headers: List[str] = [],
    cost_callback: Optional[callable] = None,
) -> str:
    """
    生成报告

    :param query: 查询
    :param context: 上下文
    :param agent_role_prompt: 代理角色提示
    :param report_type: 报告类型
    :param tone: 语气
    :param report_source: 报告来源
    :param cfg: 配置对象
    :param websocket: WebSocket对象（可选）
    :param main_topic: 主题（可选）
    :param existing_headers: 现有标题列表（可选）
    :param cost_callback: 成本回调函数（可选）
    :return: 生成的报告
    """
    generate_prompt = get_report_by_type(report_type)

    if report_type == "subtopic_report":
        content = generate_prompt(
            query,
            existing_headers,
            main_topic,
            context,
            report_format=cfg.report_format,
            total_words=cfg.total_words,
        )
    else:
        content = generate_prompt(
            query,
            context,
            report_source,
            report_format=cfg.report_format,
            total_words=cfg.total_words,
        )

    if tone:
        content += f", tone={tone.value}"

    language_model = init_language_model(temperature=cfg.temperature)
    chat = language_model

    messages = [
        {"role": "system", "content": f"{agent_role_prompt}"},
        {"role": "user", "content": content},
    ]

    response = await chat.ainvoke(messages)

    if cost_callback:
        # 在这里实现成本计算（如果需要）
        pass

    return response.content


async def get_report_introduction(
    query: str,
    context: str,
    role: str,
    config: Config,
    websocket: Any = None,
    cost_callback: Optional[callable] = None,
) -> str:
    """
    获取报告引言

    :param query: 查询
    :param context: 上下文
    :param role: 角色
    :param config: 配置对象
    :param websocket: WebSocket对象（可选）
    :param cost_callback: 成本回调函数（可选）
    :return: 生成的报告引言
    """
    language_model = init_language_model(temperature=config.temperature)
    chat = language_model

    prompt = generate_report_introduction_prompt(query, context)

    messages = [
        {"role": "system", "content": f"{role}"},
        {"role": "user", "content": prompt},
    ]

    response = await chat.ainvoke(messages)

    return response.content


# 以下是测试函数，在实际使用时可以注释掉或删除
async def test_agent_and_queries():
    cfg = Config()
    query = "人工智能的最新进展是什么？"
    report_type = "research_report"

    agent, role_prompt = await choose_agent(query, cfg)
    print(f"选择的代理: {agent}")
    print(f"角色提示: {role_prompt}")

    sub_queries = await get_sub_queries(query, role_prompt, cfg, None, report_type)
    print(f"生成的子查询: {sub_queries}")


async def test_search_and_scrape():
    cfg = Config()
    query = "人工智能的最新进展是什么？"

    retriever_class = get_retriever(cfg.retriever) or get_default_retriever()
    retriever = retriever_class(query)
    search_results = retriever.search(max_results=cfg.max_search_results_per_query)

    print(f"搜索结果: {search_results[:2]}...")  # 只打印前两个结果

    urls = [result["href"] for result in search_results]
    scraped_content = scrape_urls(urls, cfg)

    print(f"抓取的内容: {scraped_content[:1]}...")  # 只打印第一个结果


async def test_context_compression():
    cfg = Config()
    query = "人工智能的最新进展是什么？"

    retriever_class = get_retriever(cfg.retriever) or get_default_retriever()
    retriever = retriever_class(query)
    search_results = retriever.search(max_results=cfg.max_search_results_per_query)

    urls = [result["href"] for result in search_results]
    scraped_content = scrape_urls(urls, cfg)

    memory = Memory(cfg.embedding_provider)
    embeddings = memory.get_embeddings()

    context_compressor = ContextCompressor(scraped_content, embeddings)
    compressed_context = await context_compressor.get_context(query)

    print(f"压缩后的上下文:\n{compressed_context}")


async def test_report_generation():
    cfg = Config()
    query = "人工智能的最新进展是什么？"
    context = "人工智能在自然语言处理和计算机视觉方面取得了显著进展..."
    agent_role_prompt = "你是一个专门研究人工智能最新发展的AI研究助手。"

    print("\n生成子主题:")
    subtopics = await construct_subtopics(query, context, cfg)
    print(subtopics)
    print(f"生成的子主题: {[subtopic['task'] for subtopic in subtopics]}")

    # 可以根据需要取消注释以下代码来测试不同类型的报告生成
    # report_types = [ReportType.ResearchReport.value, ReportType.ResourceReport.value, ReportType.OutlineReport.value]
    # tone = Tone.Objective
    # report_source = ReportSource.Web.value
    #
    # for report_type in report_types:
    #     print(f"\n生成 {report_type}:")
    #     report = await generate_report(query, context, agent_role_prompt, report_type, tone, report_source, cfg)
    #     print(f"{report[:1000]}...")  # 打印每个报告的前1000个字符


if __name__ == "__main__":
    # asyncio.run(test_agent_and_queries())
    # asyncio.run(test_search_and_scrape())
    asyncio.run(test_context_compression())
    # asyncio.run(test_report_generation())
