from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import SQLChatMessageHistory
from utils.env_loader import EnvLoader


class OpenAiMemory:
    """
    Manages interaction with an OpenAI chat model, maintaining a history of messages.

    This class initializes a ChatOpenAI model and uses SQLChatMessageHistory
    to store the conversation turns in a SQLite database, allowing for persistent interactions.
    """
    def __init__(self, model_name: str, api_key: str, session_id: str = "user_session"):
        """
        Initializes the OpenAiMemory with a specific OpenAI model and API key.

        Args:
            model_name (str): The name of the OpenAI chat model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            api_key (str): Your OpenAI API key for authentication.
            session_id (str): The session ID for the chat history. Defaults to "user_session".
        """
        self.model = ChatOpenAI(model_name=model_name, api_key=api_key)
        self.memory = SQLChatMessageHistory(
            session_id=session_id,
            connection="sqlite:///cache/chat_history.db"
        )
        
    def get_response(self, prompt: str) -> str:
        """
        Sends a user prompt to the OpenAI model and retrieves its response.

        The prompt is added to the chat history, then the full history is sent
        to the model. The model's response is also added to the history.

        Args:
            prompt (str): The user's input message.

        Returns:
            str: The content of the AI model's response.
        """
        self.memory.add_message(HumanMessage(content=prompt))
        
        messages = self.memory.messages
        response = self.model.invoke(messages)
        
        self.memory.add_message(response)
        
        return response.content

    def get_response_stream(self, prompt: str):
        """
        Streams the response from the OpenAI model.

        Args:
            prompt (str): The user's input message.

        Yields:
            str: Chunks of the AI model's response.
        """
        self.memory.add_message(HumanMessage(content=prompt))
        messages = self.memory.messages
        
        full_response = ""
        for chunk in self.model.stream(messages):
            content = chunk.content
            full_response += content
            yield content
            
        self.memory.add_message(AIMessage(content=full_response))

if __name__ == "__main__":
    config = EnvLoader()
    model_name = config.get_required("OPENAI_MODEL_NAME")
    api_key = config.get_required("OPENAI_API_KEY")
    
    # You can change the session_id to have different conversations
    chat = OpenAiMemory(model_name, api_key, session_id="user_session")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        
        for chunk in chat.get_response_stream(user_input):
            print(chunk, end="", flush=True)
        print() # Newline at the end
