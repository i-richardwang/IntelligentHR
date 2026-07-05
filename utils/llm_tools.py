import os
import time
import logging
import traceback
from typing import Any, Dict, List, Tuple, Optional, Type, Union
import httpx

import pandas as pd
from tqdm import tqdm
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import Runnable
from langchain_core.embeddings import Embeddings

# 库模块只获取 logger，不配置根 logger（日志配置由应用入口 utils.logging_config.setup_logging 负责）
logger = logging.getLogger(__name__)


def init_language_model(
    temperature: float = 0.0,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    **kwargs: Any,
) -> ChatOpenAI:
    """
    初始化语言模型，支持OpenAI模型和其他模型供应商。

    Args:
        temperature: 模型输出的温度，控制随机性。默认为0.0。
        provider: 可选的模型供应商，优先于环境变量。
        model_name: 可选的模型名称，优先于环境变量。
        **kwargs: 其他可选参数，将传递给模型初始化。

    Returns:
        初始化后的语言模型实例。

    Raises:
        ValueError: 当提供的参数无效或缺少必要的配置时抛出。
    """
    provider = (
        provider.lower() if provider else os.getenv("LLM_PROVIDER", "openai").lower()
    )
    model_name = model_name or os.getenv("LLM_MODEL", "gpt-4")

    api_key_env_var = f"OPENAI_API_KEY_{provider.upper()}"
    api_base_env_var = f"OPENAI_API_BASE_{provider.upper()}"

    openai_api_key = os.environ.get(api_key_env_var)
    openai_api_base = os.environ.get(api_base_env_var)

    if not openai_api_key or not openai_api_base:
        raise ValueError(
            f"无法找到 {provider} 的 API 密钥或基础 URL。请检查环境变量设置。"
        )

    model_params = {
        "model": model_name,
        "api_key": openai_api_key,
        "base_url": openai_api_base,
        "temperature": temperature,
        **kwargs,
    }

    return ChatOpenAI(**model_params)


class LanguageModelChain:
    """
    语言模型链，用于处理输入并生成符合指定模式的输出。

    Attributes:
        model_cls: Pydantic 模型类，定义输出的结构。
        parser: JSON 输出解析器。
        prompt_template: 聊天提示模板。
        chain: 完整的处理链。
    """

    def __init__(
        self, model_cls: Type[BaseModel], sys_msg: str, user_msg: str, model: Any
    ):
        """
        初始化 LanguageModelChain 实例。

        Args:
            model_cls: Pydantic 模型类，定义输出的结构。
            sys_msg: 系统消息。
            user_msg: 用户消息。
            model: 语言模型实例。

        Raises:
            ValueError: 当提供的参数无效时抛出。
        """
        if not issubclass(model_cls, BaseModel):
            raise ValueError("model_cls 必须是 Pydantic BaseModel 的子类")
        if not isinstance(sys_msg, str) or not isinstance(user_msg, str):
            raise ValueError("sys_msg 和 user_msg 必须是字符串类型")
        if not isinstance(model, Runnable) and not callable(model):
            raise ValueError("model 必须是 langchain Runnable 或可调用对象")

        self.model_cls = model_cls
        self.parser = JsonOutputParser(pydantic_object=model_cls)

        format_instructions = """
Output your answer as a JSON object that conforms to the following schema:
```json
{schema}
```

Important instructions:
1. Ensure your JSON is valid and properly formatted.
2. Do not include the schema definition in your answer.
3. Only output the data instance that matches the schema.
4. Do not include any explanations or comments within the JSON output.
        """

        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", sys_msg + format_instructions),
                ("human", user_msg),
            ]
        ).partial(schema=model_cls.model_json_schema())

        self.chain = self.prompt_template | model | self.parser

    def __call__(self) -> Any:
        """
        调用处理链。

        Returns:
            处理链的输出。
        """
        return self.chain


def batch_process_data(
    llm_chain: Any,
    df: pd.DataFrame,
    field_map: Dict[str, str],
    model_cls: Type[BaseModel],
    static_params: Optional[Dict[str, Any]] = None,
    extra_fields: Optional[List[str]] = None,
    batch_size: int = 10,
    max_retries: int = 1,
    call_interval: Optional[float] = None,
    output_json: bool = False,
    config: Any = None,
) -> Union[
    Tuple[pd.DataFrame, pd.DataFrame], Tuple[List[Dict[str, Any]], pd.DataFrame]
]:
    """
    批量处理数据集，调用大模型任务链，并返回处理结果和错误信息。包含重试机制和可选的调用间隔。

    Args:
        llm_chain: 语言模型链实例。
        df: 输入数据集。
        field_map: 字段映射，将输入字段映射到模型所需字段。
        model_cls: Pydantic 模型类，定义输出的结构。
        static_params: 静态参数，应用于所有批次。
        extra_fields: 要包含在结果中的额外字段。
        batch_size: 每个批次的大小。
        max_retries: 批处理失败时的最大重试次数。
        call_interval: 每次调用后的停顿时间（秒）。如果为None，则不进行停顿。
        output_json: 是否输出原始JSON列表而不是DataFrame。
        config: 额外的配置参数。

    Returns:
        如果output_json为False，返回包含处理结果的DataFrame和错误日志的DataFrame。
        如果output_json为True，返回包含原始JSON的列表和错误日志的DataFrame。

    Raises:
        ValueError: 当提供的参数无效时抛出。
    """
    # 参数验证
    if not isinstance(df, pd.DataFrame):
        raise ValueError("df 必须是 pandas DataFrame 类型")
    if not isinstance(field_map, dict):
        raise ValueError("field_map 必须是字典类型")
    for invoke_field, data_field in field_map.items():
        if not all(isinstance(f, str) for f in (invoke_field, data_field)):
            raise ValueError("field_map 中的键和值必须都是字符串")
        if data_field not in df.columns:
            raise ValueError(f"field_map 中的数据字段 {data_field} 不存在于 df 中")
    if static_params is not None and not isinstance(static_params, dict):
        raise ValueError("static_params 必须是字典类型或 None")
    if extra_fields is not None:
        if not isinstance(extra_fields, list):
            raise ValueError("extra_fields 必须是列表类型或 None")
        missing_fields = set(extra_fields) - set(df.columns)
        if missing_fields:
            raise ValueError(
                f"extra_fields 中的字段 {', '.join(missing_fields)} 不存在于 df 中"
            )
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size 必须是一个正整数")
    if not hasattr(llm_chain, "batch"):
        raise ValueError("llm_chain 必须有一个 batch 方法")
    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError("max_retries 必须是一个非负整数")
    if call_interval is not None and (
        not isinstance(call_interval, (int, float)) or call_interval < 0
    ):
        raise ValueError("call_interval 必须是一个非负数或 None")

    processed_results = []
    error_logs = []

    def construct_params(row: pd.Series) -> Dict[str, Any]:
        return {
            **{
                invoke_field: row[data_field]
                for invoke_field, data_field in field_map.items()
            },
            **(static_params or {}),
        }

    def handle_response(response: Any, extra_data: Dict[str, Any]) -> Dict[str, Any]:
        if output_json:
            return {**response, **extra_data}
        else:
            model_field = next(iter(model_cls.__annotations__))
            if isinstance(response, dict):
                if model_field in response:
                    result = (
                        response[model_field]
                        if isinstance(response[model_field], list)
                        else [response]
                    )
                else:
                    result = [response]
            elif isinstance(response, list):
                result = response
            else:
                result = [response]
            return [{**item, **extra_data} for item in result]

    def process_batch(
        batch: pd.DataFrame, start_idx: int
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        batch_results = []
        batch_errors = []
        batch_params = [construct_params(row) for _, row in batch.iterrows()]

        for retry in range(max_retries + 1):
            try:
                responses = llm_chain.batch(batch_params, config=config)
                for i, response in enumerate(responses):
                    extra_data = {
                        field: batch.iloc[i][field] for field in (extra_fields or [])
                    }
                    processed_response = handle_response(response, extra_data)
                    batch_results.append(
                        processed_response if output_json else processed_response[0]
                    )
                return batch_results, batch_errors
            except Exception as e:
                retry_delay = 10 if call_interval is None else call_interval * 10
                if retry < max_retries:
                    logger.warning(
                        f"处理批次 {start_idx // batch_size + 1} 时发生错误，{retry_delay:.1f}秒后进行第{retry + 1}次重试:"
                    )
                    logger.warning(f"错误类型: {type(e).__name__}")
                    logger.warning(f"错误信息: {str(e)}")
                    time.sleep(retry_delay)
                else:
                    for i in range(len(batch)):
                        error_info = {
                            field: batch.iloc[i][field]
                            for field in (extra_fields or [])
                        }
                        error_info.update({"index": start_idx + i, "error": str(e)})
                        batch_errors.append(error_info)
                    logger.error(
                        f"处理批次 {start_idx // batch_size + 1} 失败，已达到最大重试次数:"
                    )
                    logger.error(f"错误类型: {type(e).__name__}")
                    logger.error(f"错误信息: {str(e)}")
                    logger.error(traceback.format_exc())

        return batch_results, batch_errors

    total_batches = (len(df) + batch_size - 1) // batch_size
    for start_idx in tqdm(
        range(0, len(df), batch_size), desc="批处理进度", total=total_batches
    ):
        end_idx = min(start_idx + batch_size, len(df))
        batch = df.iloc[start_idx:end_idx]

        batch_results, batch_errors = process_batch(batch, start_idx)
        processed_results.extend(batch_results)
        error_logs.extend(batch_errors)

        if call_interval is not None:
            time.sleep(call_interval)

    if output_json:
        logger.info(f"\n处理完成:")
        logger.info(f"成功处理的条目数: {len(processed_results)}")
        logger.info(f"处理失败的条目数: {len(error_logs)}")
        return processed_results, pd.DataFrame(error_logs)
    else:
        result_df = pd.DataFrame(processed_results)
        error_df = pd.DataFrame(error_logs)

        logger.info(f"\n处理完成:")
        logger.info(f"成功处理的行数: {len(result_df)}")
        logger.info(f"处理失败的行数: {len(error_df)}")

        return result_df, error_df


class CustomEmbeddings(Embeddings):
    """OpenAI 兼容的自定义 Embeddings 客户端。

    面向 OpenAI 兼容的 /embeddings 端点（如 SiliconFlow 上的 BAAI/bge 系列），
    逐条文本发起请求。同时提供同步与原生异步实现——异步版本覆写了 LangChain
    Embeddings 基类默认的线程池实现，以获得真正的并发能力（基类文档明确允许
    出于性能考虑覆写异步方法）。
    """

    def __init__(
        self,
        api_key: str,
        api_url: str,
        model: str,
        *,
        timeout: float = 30.0,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.model = model
        self.timeout = timeout

    def _headers(self) -> Dict[str, str]:
        return {
            "accept": "application/json",
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def _payload(self, text: str) -> Dict[str, Any]:
        return {"model": self.model, "input": text, "encoding_format": "float"}

    @staticmethod
    def _parse_embedding(data: Dict[str, Any]) -> List[float]:
        return data["data"][0]["embedding"]

    def _get_embeddings(self, texts: List[str]) -> List[List[float]]:
        with httpx.Client(timeout=self.timeout) as client:
            embeddings: List[List[float]] = []
            for text in texts:
                response = client.post(
                    self.api_url, headers=self._headers(), json=self._payload(text)
                )
                response.raise_for_status()
                embeddings.append(self._parse_embedding(response.json()))
            return embeddings

    async def _aget_embeddings(self, texts: List[str]) -> List[List[float]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            embeddings: List[List[float]] = []
            for text in texts:
                response = await client.post(
                    self.api_url, headers=self._headers(), json=self._payload(text)
                )
                response.raise_for_status()
                embeddings.append(self._parse_embedding(response.json()))
            return embeddings

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._get_embeddings(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._get_embeddings([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self._aget_embeddings(texts)

    async def aembed_query(self, text: str) -> List[float]:
        return (await self._aget_embeddings([text]))[0]


def create_embeddings(model: Optional[str] = None) -> CustomEmbeddings:
    """创建统一配置的 Embeddings 客户端（工厂）。

    从环境变量读取 embedding 服务配置，消除各调用点重复构造 CustomEmbeddings
    的样板，并确保全项目 embedding 的端点 / 密钥 / 模型来源一致（避免出现某处
    硬编码供应商 base_url、换供应商换不掉的问题）。

    Args:
        model: 可选的模型名覆盖；缺省时使用环境变量 EMBEDDING_MODEL。

    Returns:
        CustomEmbeddings: 已配置好的 embedding 客户端。

    Raises:
        ValueError: 当缺少必要的环境变量配置时抛出（快速失败，避免延迟到请求时才报错）。
    """
    api_key = os.getenv("EMBEDDING_API_KEY")
    api_url = os.getenv("EMBEDDING_API_BASE")
    model = model or os.getenv("EMBEDDING_MODEL")

    missing = [
        name
        for name, value in (
            ("EMBEDDING_API_KEY", api_key),
            ("EMBEDDING_API_BASE", api_url),
            ("EMBEDDING_MODEL", model),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"缺少 embedding 服务配置，请设置环境变量：{', '.join(missing)}"
        )

    return CustomEmbeddings(api_key=api_key, api_url=api_url, model=model)


class VectorEncoder:
    """文本向量编码器，编码失败时自动截断文本重试。

    embedding 客户端延迟初始化：避免在模块导入期就强制要求 embedding 环境变量
    （本类常被在模块级实例化），仅在首次实际编码时才创建。

    ``model`` 缺省为 ``None``，此时经工厂使用环境变量 ``EMBEDDING_MODEL``——与查询侧
    :func:`create_embeddings` 同源，确保入库与查询使用同一 embedding 模型（不同模型的
    向量不可比，口径不一致会导致相似度失真）。仅在确需覆盖时才显式传入 model。
    """

    def __init__(
        self,
        model: Optional[str] = None,
    ):
        self.model = model
        self._embeddings: Optional[CustomEmbeddings] = None

    @property
    def embeddings(self) -> CustomEmbeddings:
        if self._embeddings is None:
            self._embeddings = create_embeddings(model=self.model)
        return self._embeddings

    def get_embedding(self, text: str) -> Optional[List[float]]:
        while True:
            try:
                return self.embeddings.embed_query(text)
            except Exception:
                if len(text) <= 1:
                    logger.error("文本太短，无法进一步截断。中止操作。")
                    return None
                text = text[: int(len(text) * 0.9)]
                time.sleep(0.1)
                logger.warning(f"截断文本至 {len(text)} 个字符并重试...")
