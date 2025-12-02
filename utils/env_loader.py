import os
from dotenv import load_dotenv, find_dotenv
from typing import Optional

class EnvLoader:
    """
    Utility class for loading environment variables from a .env file and 
    providing controlled access via the os.getenv function.
    """
    
    # Singleton instance
    _instance: Optional['EnvLoader'] = None

    def __new__(cls, *args, **kwargs):
        """
        Implements the Singleton pattern: ensures only one instance 
        of EnvLoader exists across the application.
        """
        if cls._instance is None:
            cls._instance = super(EnvLoader, cls).__new__(cls)
            # Calls the initialization method on the first creation
            cls._instance._load_variables() 
        return cls._instance

    def _load_variables(self) -> None:
        """
        Attempts to locate and load the .env file.
        This method is called only once during the first initialization.
        """
        # find_dotenv() recursively searches for the .env file
        dotenv_path = find_dotenv(usecwd=True, raise_error_if_not_found=False)

        if not dotenv_path:
            # Does not print anything if not found, just does not load.
            return

        # Loads the variables from the found path
        load_dotenv(dotenv_path)

    def get(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """
        Retrieves an environment variable by key.
        
        :param key: The environment variable name.
        :param default: A fallback value if the key is not found.
        :return: The value of the environment variable or the default value.
        """
        return os.getenv(key, default)
        
    def get_required(self, key: str) -> str:
        """
        Retrieves a required environment variable or raises an error if not found.
        
        :param key: The environment variable name.
        :return: The value of the environment variable.
        :raises ValueError: If the required key is not set or is empty.
        """
        value = self.get(key)
        if not value: # Checks for None or empty string
            raise ValueError(f"🚨 Required environment variable '{key}' is not set or is empty.")
        return value