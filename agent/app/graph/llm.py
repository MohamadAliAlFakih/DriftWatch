"""Chat model factory.

Production builds a ChatGroq from the GROQ_API_KEY in settings. Tests inject a
FakeChatModel via build_graph(chat_model=...) and never reach this factory.
"""

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import Settings


# build production chat model — ChatGroq via langchain-groq
def build_chat_model(settings: Settings) -> BaseChatModel:
    if settings.groq_api_key is None:
        raise RuntimeError("GROQ_API_KEY missing — set it in .env or inject a fake chat model in tests")
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=settings.groq_model,
        api_key=settings.groq_api_key.get_secret_value(),
        temperature=0,
    )
