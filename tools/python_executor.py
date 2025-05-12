# tools/python_executor.py
import sys
import io
from contextlib import redirect_stdout, redirect_stderr
from .base_tool import BaseTool

class PythonExecutor(BaseTool):
    """Tool for executing Python code."""
    
    def __init__(self):
        super().__init__(name="python")
        self.globals = {
            "__builtins__": __builtins__,
            "print": print,
        }
        
        # Add common libraries to globals
        try:
            import numpy as np
            self.globals["np"] = np
        except ImportError:
            pass
            
        try:
            import sympy
            self.globals["sympy"] = sympy
            for name in dir(sympy):
                if not name.startswith("_"):
                    self.globals[name] = getattr(sympy, name)
        except ImportError:
            pass
    
    def execute(self, code: str) -> tuple[str, bool]:
        """
        Execute Python code and return the output and success status.
        
        Args:
            code: Python code to execute
            
        Returns:
            tuple: (output, success_flag)
        """
        buffer = io.StringIO()
        
        try:
            with redirect_stdout(buffer), redirect_stderr(buffer):
                exec(code, self.globals)
            output = buffer.getvalue()
            return output, True
        except Exception as e:
            error_msg = f"Error executing code: {str(e)}"
            return error_msg, False
