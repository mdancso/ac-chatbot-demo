import os
from typing import Literal, Dict, Callable

from langchain_openai import ChatOpenAI, AzureChatOpenAI

PossibleModels = Literal["gpt-4o-mini", "gpt-4o", "gpt-4o-azure"]

class ChatModels:
    _openai_api_initializer = lambda model_name: ChatOpenAI(model=model_name, temperature=0)
    _azure_initializer = lambda _: AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_GPT4o_DEPLOYMENT_NAME"),
        azure_endpoint=os.getenv("AZURE_DEPLOYMENT_ENDPOINT"),
        api_version=os.getenv("AZURE_GPT4o_API_VERSION")
    )

    @classmethod
    def _models(cls) -> Dict[PossibleModels, Callable]:
        return {
            "gpt-4o-mini": cls._openai_api_initializer,
            "gpt-4o": cls._openai_api_initializer,
            "gpt-4o-azure": cls._azure_initializer
        }

    @classmethod
    def get(cls, model_name: PossibleModels):
        models = cls._models()
        if model_name not in models:
            raise ValueError(f"Model '{model_name}' is not supported.")
        return models[model_name](model_name)
