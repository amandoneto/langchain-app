from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.runnables import RunnablePassthrough
from utils.env_loader import EnvLoader

"""
A router class that initializes an OpenAI chat model and creates LangChain chains
for different programming languages (e.g., JavaScript, Python).
"""

class Route(BaseModel):
    """Route the query to the most relevant area."""
    knowledge_field: Literal["JavaScript", "TypeScript", "Python", "AI", "Other"] = Field(
        ..., 
        description="The area of knowledge required to answer the user query."
    )

class Router:
    
    def __init__(self, api_key: str, model_name: str):
        """
        Initializes the Router with an OpenAI model.

        Args:
            api_key (str): The API key for OpenAI.
            model_name (str): The name of the OpenAI model to use (e.g., "gpt-4").
        """
        self.chat_model = ChatOpenAI(model=model_name, api_key=api_key)
        

    def create_javascript_chain(self):
        """
        Creates a LangChain chain configured with a system prompt for a senior JavaScript/TypeScript expert.

        Returns:
            A LangChain expression representing the prompt and the OpenAI model.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a senior developer with 20 years of experience. You are an expert in JavaScript and TypeScript. 
            You are also an expert in software architecture and design patterns.
            """),
            ("human", "{input}"),
        ])
        
        return prompt | self.chat_model

    def create_python_chain(self):
        """
        Creates a LangChain chain configured with a system prompt for a senior Python expert.

        Returns:
            A LangChain expression representing the prompt and the OpenAI model.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a senior developer with 20 years of experience. You are an expert in Python. 
            You are also an expert in software architecture and design patterns.
            """),
            ("human", "{input}"),
        ])
        
        return prompt | self.chat_model

    def create_ai_chain(self):
        """
        Creates a LangChain chain configured with a system prompt for a senior AI expert.

        Returns:
            A LangChain expression representing the prompt and the OpenAI model.
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are a senior developer with 20 years of experience. You are an expert in AI. 
            You are also an expert in best practices for AI development.
            """),
            ("human", "{input}"),
        ])
        
        return prompt | self.chat_model

    def get_category_chain(self):
        """
        Creates a classification chain to determine the query category.
        """
        system_prompt = """You are an expert at routing user queries to the appropriate specialized assistant.
        The available areas are: JavaScript, TypeScript, Python, and AI.
        If the query is not related to these areas, select 'Other'.
        Route the input to the most relevant area."""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        structured_llm = self.chat_model.with_structured_output(Route)
        return prompt | structured_llm

    def set_router(self, inputs: dict):
        category = inputs["category"]
        user_input = inputs["input"]
        
        if category.knowledge_field in ['JavaScript', 'TypeScript']:
            chain = self.create_javascript_chain()
        elif category.knowledge_field == 'Python':
            chain = self.create_python_chain()
        elif category.knowledge_field == 'AI':
            chain = self.create_ai_chain()
        else:
            return RunnablePassthrough() | (lambda x: "I'm sorry, I can only assist with JavaScript, TypeScript, Python, and AI.")
            
        return chain.invoke({"input": user_input})
    
    def run_category_chain(self, query: str):
        """
        Runs the category chain to classify the query and then routes it to the appropriate
        specialized chain based on the classification. Finally, it prints the response.

        Args:
            query (str): The user's input query.
        """
        category_chain = self.get_category_chain()
        
        chain = (
            RunnablePassthrough.assign(category=category_chain)
            | self.set_router
        )
        
        response = chain.invoke({"input": query})
        print(response.content)

    def create_chat(self):
        """
        Starts an interactive chat session with the user.
        It continuously prompts the user for input, classifies the query,
        routes it to the appropriate specialized chain, and prints the response.
        The session ends when the user types 'exit' or 'quit'.
        """
        print("Welcome to the chat! Type 'exit' or 'quit' to end the conversation.")
        print("I will be your assistant on javascript, typescript, python and AI.")

        while True:
            user_input = input("You: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            
            self.run_category_chain(user_input)


if __name__ == "__main__":
    config = EnvLoader()
    api_key = config.get_required("OPENAI_API_KEY")
    model_name = config.get_required("OPENAI_MODEL_NAME")
    router = Router(api_key, model_name)
    router.create_chat()   