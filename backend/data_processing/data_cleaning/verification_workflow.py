import asyncio
from typing import Dict, Any, List, Optional, Callable, Literal
from uuid import uuid4
from langfuse.callback import CallbackHandler
import traceback
import logging

logger = logging.getLogger(__name__)

from backend_demo.data_processing.data_cleaning.search_tools import SearchTools
from backend_demo.data_processing.data_cleaning.verification_models import (
    input_validator,
    search_analysis,
    name_verifier,
)
from backend_demo.data_processing.data_cleaning.data_processor import (
    initialize_vector_store,
    get_entity_retriever,
)
from enum import Enum


class ProcessingStatus(Enum):
    INVALID_INPUT = "无效输入"
    VALID_INPUT = "有效输入"
    IDENTIFIED = "已识别"
    UNIDENTIFIED = "未识别"
    VERIFIED = "已验证"
    UNVERIFIED = "未验证"
    ERROR = "处理错误"


def create_langfuse_handler(session_id: str, step: str) -> CallbackHandler:
    return CallbackHandler(
        tags=["entity_verification"], session_id=session_id, metadata={"step": step}
    )


class EntityVerificationWorkflow:
    def __init__(
        self,
        retriever: Callable,
        entity_type: str,
        original_field: str,
        standard_field: str,
        validation_instructions: str,
        analysis_instructions: str,
        verification_instructions: str,
        skip_validation: bool = False,
        skip_search: bool = False,
        skip_retrieval: bool = False,
        search_tool: Literal["duckduckgo", "tavily"] = "duckduckgo",
    ):
        self.retriever = retriever
        self.entity_type = entity_type
        self.original_field = original_field
        self.standard_field = standard_field
        self.validation_instructions = validation_instructions
        self.analysis_instructions = analysis_instructions
        self.verification_instructions = verification_instructions
        self.skip_validation = skip_validation
        self.skip_search = skip_search
        self.skip_retrieval = skip_retrieval
        self.search_tools = SearchTools()
        self.search_tool = search_tool

    async def run(
        self, user_query: str, session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        user_query = str(user_query)

        if session_id is None:
            session_id = str(uuid4())

        result = self._initialize_result()
        result["original_input"] = user_query
        result["is_identified"] = False

        try:
            # Input validation
            if not self.skip_validation:
                langfuse_handler = create_langfuse_handler(
                    session_id, "input_validation"
                )
                result["is_valid"] = await self._validate_input(
                    user_query, langfuse_handler
                )
            else:
                result["is_valid"] = True

            if not result["is_valid"]:
                result["status"] = ProcessingStatus.INVALID_INPUT
                return self._generate_output(result)

            # If skipping search and retrieval, return valid input
            if self.skip_search and self.skip_retrieval:
                result["status"] = ProcessingStatus.VALID_INPUT
                result["identified_entity_name"] = user_query
                return self._generate_output(result)

            # Web search and analysis
            if not self.skip_search:
                search_results = await self._perform_web_search(user_query)
                result["search_results"] = search_results
                langfuse_handler = create_langfuse_handler(
                    session_id, "search_analysis"
                )
                result["identified_entity_name"], is_identified = (
                    await self._analyze_search_results(
                        user_query, search_results, langfuse_handler
                    )
                )
                result["status"] = (
                    ProcessingStatus.IDENTIFIED
                    if is_identified
                    else ProcessingStatus.UNIDENTIFIED
                )
                result["is_identified"] = is_identified
            else:
                result["search_results"] = None
                result["identified_entity_name"] = user_query
                result["status"] = ProcessingStatus.IDENTIFIED

            # Vector retrieval and verification
            if (
                not self.skip_retrieval
                and result["status"] == ProcessingStatus.IDENTIFIED
            ):
                retrieval_results = await self._direct_retrieve(
                    result["identified_entity_name"]
                )
                if retrieval_results:
                    result["retrieved_entity_name"] = retrieval_results[0].get(
                        self.original_field, ""
                    )
                    result["standard_name"] = retrieval_results[0].get(
                        self.standard_field, ""
                    )
                    langfuse_handler = create_langfuse_handler(
                        session_id, "name_verification"
                    )
                    is_verified = await self._evaluate_match(
                        user_query,
                        result["retrieved_entity_name"],
                        search_results if not self.skip_search else "",
                        langfuse_handler,
                    )
                    result["status"] = (
                        ProcessingStatus.VERIFIED
                        if is_verified
                        else ProcessingStatus.UNVERIFIED
                    )
                else:
                    result["status"] = ProcessingStatus.UNVERIFIED

        except Exception as e:
            result["status"] = ProcessingStatus.ERROR
            result["error_message"] = str(e)
            result["error_traceback"] = traceback.format_exc()

        return self._generate_output(result)

    def _initialize_result(self) -> Dict[str, Any]:
        return {
            "is_valid": False,
            "identified_entity_name": None,
            "retrieved_entity_name": None,
            "status": None,
            "final_entity_name": None,
        }

    async def _validate_input(
        self, user_query: str, langfuse_handler: CallbackHandler
    ) -> bool:
        validation_result = await input_validator.ainvoke(
            {
                "user_query": user_query,
                "entity_type": self.entity_type,
                "validation_instructions": self.validation_instructions,
            },
            config={"callbacks": [langfuse_handler]},
        )
        return validation_result["is_valid"]

    async def _perform_web_search(self, user_query: str) -> str:
        if self.search_tool == "duckduckgo":
            return await self.search_tools.aduckduckgo_search(user_query)
        elif self.search_tool == "tavily":
            return await self.search_tools.atavily_search(user_query)
        else:
            raise ValueError(f"Unsupported search tool: {self.search_tool}")

    async def _analyze_search_results(
        self, user_query: str, search_results: str, langfuse_handler: CallbackHandler
    ) -> tuple[Optional[str], bool]:
        analysis_result = await search_analysis.ainvoke(
            {
                "user_query": user_query,
                "snippets": search_results,
                "entity_type": self.entity_type,
                "analysis_instructions": self.analysis_instructions,
            },
            config={"callbacks": [langfuse_handler]},
        )
        return (
            analysis_result["identified_entity"],
            analysis_result["recognition_status"] == "known",
        )

    async def _direct_retrieve(self, search_term: str) -> List[Dict]:
        results = await self.retriever(search_term)
        return [
            {
                "original_name": result.get("original_name", ""),
                "standard_name": result.get("standard_name", ""),
                "distance": result.get("distance", 0),
            }
            for result in results
        ]

    async def _evaluate_match(
        self,
        user_query: str,
        retrieved_name: str,
        search_results: str,
        langfuse_handler: CallbackHandler,
    ) -> bool:
        verified_result = await name_verifier.ainvoke(
            {
                "user_query": user_query,
                "retrieved_name": retrieved_name,
                "search_results": search_results,
                "entity_type": self.entity_type,
                "verification_instructions": self.verification_instructions,
            },
            config={"callbacks": [langfuse_handler]},
        )
        return verified_result["verification_status"] == "verified"

    def _generate_output(self, result: Dict[str, Any]) -> Dict[str, Any]:
        if result["status"] == ProcessingStatus.INVALID_INPUT:
            result["final_entity_name"] = "无效输入"
        elif result["status"] == ProcessingStatus.ERROR:
            result["final_entity_name"] = "处理错误"
        elif result["status"] == ProcessingStatus.VERIFIED:
            if "standard_name" in result:
                result["final_entity_name"] = result["standard_name"]
            else:
                result["final_entity_name"] = (
                    result.get("retrieved_entity_name")
                    or result["identified_entity_name"]
                )
        elif result["status"] == ProcessingStatus.IDENTIFIED:
            result["final_entity_name"] = result["identified_entity_name"]
        else:
            if (
                result["status"] == ProcessingStatus.UNVERIFIED
                and result["is_identified"]
            ):
                result["final_entity_name"] = result["identified_entity_name"]
            else:
                result["final_entity_name"] = result["original_input"]

        logger.info(
            f"Final entity name: {result['final_entity_name']}, status: {result['status']}"
        )

        return result
