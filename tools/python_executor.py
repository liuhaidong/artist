# tools/python_executor.py
import sys
import io
import traceback
from typing import Tuple

class PythonExecutor:
    """
    Tool for executing Python code safely.
    """
    
    name = "python"
    
    def __init__(self, timeout=5):
        """
        Initialize the Python executor.
        
        Args:
            timeout: Maximum execution time in seconds
        """
        self.timeout = timeout
        
    def execute(self, code: str) -> Tuple[str, bool]:
        """
        Execute Python code and return the output.
        
        Args:
            code: Python code to execute
            
        Returns:
            Tuple of (output, success)
        """
        # Capture stdout and stderr
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        redirected_output = io.StringIO()
        sys.stdout = redirected_output
        sys.stderr = redirected_output
        
        success = True
        
        try:
            # Execute the code with timeout
            exec_globals = {}
            exec(code, exec_globals)
            output = redirected_output.getvalue()
            
            # If no output, try to find the last variable
            if not output.strip():
                # Find last assignment
                last_var = None
                for line in code.strip().split('\n'):
                    line = line.strip()
                    if '=' in line and not line.startswith('#'):
                        var_name = line.split('=')[0].strip()
                        if var_name in exec_globals:
                            last_var = var_name
                
                # Get the value of the last variable
                if last_var and last_var in exec_globals:
                    output = str(exec_globals[last_var])
            
        except Exception as e:
            output = f"Error: {str(e)}\n{traceback.format_exc()}"
            success = False
        finally:
            # Restore stdout and stderr
            sys.stdout = old_stdout
            sys.stderr = old_stderr
        
        # Limit output length
        if len(output) > 1000:
            output = output[:997] + "..."
            
        return output, success
