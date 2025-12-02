from langchain_openai import ChatOpenAI
from utils.env_loader import EnvLoader
from langchain_core.messages import HumanMessage, SystemMessage

class ChatModels:
    """
        ChatModels are a central component of LangChain.

        A chat model is a language model that takes chat messages as inputs and returns chat messages as outputs (rather than plain text).

        LangChain has integrations with multiple model providers (OpenAI, Cohere, Hugging Face, etc.) and exposes a standard interface to interact with all of these models.
    """
    def __init__(self, model_name: str, api_key: str, temperature: float = 0.7):
        
        self.model_name = model_name
        self.temperature = temperature
        try:
            
            self.chat_model = ChatOpenAI(model_name=self.model_name, 
                                    temperature=self.temperature,
                                    api_key=api_key)
        except ValueError as e:
            print(f"Configuration Error: {e}")

    def get_chat_model(self):
        messages = [
            SystemMessage(content="You are a a tennis expert."),
            HumanMessage(content="Please, can you tell me about the latest tennis news?")
        ]

        for message in self.chat_model.stream(messages):
            print(message.content, end="", flush=True)
        

if __name__ == "__main__":
    config = EnvLoader()
    chat_models = ChatModels(model_name=config.get_required("OPENAI_MODEL_NAME"),
                            api_key=config.get_required("OPENAI_API_KEY"))
    
    chat_models.get_chat_model()