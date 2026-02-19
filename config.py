"""
Configuration file for the Multi-Agent Medical Chatbot

This file contains all the configuration parameters for the project.

Supports both Azure OpenAI and OpenAI-compatible backends (e.g. OpenRouter)
via environment variables: OPENAI_API_KEY, OPENAI_API_BASE, OPENAI_MODEL.
"""

import logging
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

logger = logging.getLogger(__name__)


def _get_llm(temperature: float = 0.5) -> ChatOpenAI:
    """Create a ChatOpenAI instance from env-based configuration."""
    kwargs: dict = {"temperature": temperature}

    base_url = os.getenv("OPENAI_API_BASE")
    if base_url:
        kwargs["base_url"] = base_url

    model = os.getenv("OPENAI_MODEL")
    if model:
        kwargs["model"] = model

    return ChatOpenAI(**kwargs)


class AgentDecisoinConfig:
    def __init__(self):
        self.llm = _get_llm(temperature=0.1)

class ConversationConfig:
    def __init__(self):
        self.llm = _get_llm(temperature=0.7)

class WebSearchConfig:
    def __init__(self):
        self.llm = _get_llm(temperature=0.3)
        self.context_limit = 20

class RAGConfig:
    def __init__(self):
        self.vector_db_type = "qdrant"
        self.embedding_dim = 1536
        self.distance_metric = "Cosine"
        self.use_local = True
        self.vector_local_path = "./data/qdrant_db"
        self.doc_local_path = "./data/docs_db"
        self.parsed_content_dir = "./data/parsed_docs"
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection_name = "medical_assistance_rag"
        self.chunk_size = 512
        self.chunk_overlap = 50

        try:
            embed_kwargs: dict = {}
            base_url = os.getenv("EMBEDDING_API_BASE") or os.getenv("OPENAI_API_BASE")
            if base_url:
                embed_kwargs["base_url"] = base_url
            embed_model = os.getenv("EMBEDDING_MODEL")
            if embed_model:
                embed_kwargs["model"] = embed_model
            self.embedding_model = OpenAIEmbeddings(**embed_kwargs)
        except Exception as e:
            logger.warning("Embedding model init failed (%s), RAG will be unavailable", e)
            self.embedding_model = None

        self.llm = _get_llm(temperature=0.3)
        self.summarizer_model = _get_llm(temperature=0.5)
        self.chunker_model = _get_llm(temperature=0.0)
        self.response_generator_model = _get_llm(temperature=0.3)
        self.top_k = 5
        self.vector_search_type = 'similarity'

        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN")

        self.reranker_model = "cross-encoder/ms-marco-TinyBERT-L-6"
        self.reranker_top_k = 3

        self.max_context_length = 8192
        self.include_sources = True
        self.min_retrieval_confidence = 0.40
        self.context_limit = 20

class MedicalCVConfig:
    def __init__(self):
        self.brain_tumor_model_path = "./agents/image_analysis_agent/brain_tumor_agent/models/brain_tumor_segmentation.pth"
        self.chest_xray_model_path = "./agents/image_analysis_agent/chest_xray_agent/models/covid_chest_xray_model.pth"
        self.skin_lesion_model_path = "./agents/image_analysis_agent/skin_lesion_agent/models/checkpointN25_.pth.tar"
        self.skin_lesion_segmentation_output_path = "./uploads/skin_lesion_output/segmentation_plot.png"
        self.llm = _get_llm(temperature=0.1)

class SpeechConfig:
    def __init__(self):
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.eleven_labs_voice_id = "21m00Tcm4TlvDq8ikWAM"

class ValidationConfig:
    def __init__(self):
        self.require_validation = {
            "CONVERSATION_AGENT": False,
            "RAG_AGENT": False,
            "WEB_SEARCH_AGENT": False,
            "BRAIN_TUMOR_AGENT": True,
            "CHEST_XRAY_AGENT": True,
            "SKIN_LESION_AGENT": True
        }
        self.validation_timeout = 300
        self.default_action = "reject"

class APIConfig:
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = int(os.getenv("PORT", 8000))
        self.debug = True
        self.rate_limit = 10
        self.max_image_upload_size = 5

class UIConfig:
    def __init__(self):
        self.theme = "light"
        self.enable_speech = True
        self.enable_image_upload = True

class Config:
    def __init__(self):
        self.agent_decision = AgentDecisoinConfig()
        self.conversation = ConversationConfig()
        self.rag = RAGConfig()
        self.medical_cv = MedicalCVConfig()
        self.web_search = WebSearchConfig()
        self.api = APIConfig()
        self.speech = SpeechConfig()
        self.validation = ValidationConfig()
        self.ui = UIConfig()
        self.eleven_labs_api_key = os.getenv("ELEVEN_LABS_API_KEY")
        self.tavily_api_key = os.getenv("TAVILY_API_KEY")
        self.max_conversation_history = 20
