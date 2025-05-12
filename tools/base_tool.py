# tools/base_tool.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class BaseTool(ABC):
    """Base class for all tools in ARTIST framework."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def execute(self, query: str) -> tuple[str, bool]:
        """
        Execute the tool with the given query.
        
        Args:
            query: The query to execute
            
        Returns:
            tuple: (output, success_flag)
        """
        pass
