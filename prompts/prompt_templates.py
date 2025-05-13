# prompts/prompt_templates.py
class PromptTemplates:
    """
    Collection of prompt templates for different ARTIST tasks.
    """
    
    @staticmethod
    def get_math_reasoning_prompt(question: str) -> str:
        """
        Generate a prompt for math reasoning tasks.
        
        Args:
            question: The math question
            
        Returns:
            Formatted prompt
        """
        return f"""You are an AI assistant that can solve math problems step by step.
You can use Python code to help with calculations when needed.

To use Python, write code within <python> </python> tags.
The output will be provided within <output> </output> tags.

When you have the final answer, provide it within <answer> </answer> tags.

Question: {question}

Let me solve this step by step:
"""

    @staticmethod
    def get_function_calling_prompt(question: str, available_tools: list) -> str:
        """
        Generate a prompt for function calling tasks.
        
        Args:
            question: The question to answer
            available_tools: List of available tools
            
        Returns:
            Formatted prompt
        """
        tools_description = ""
        
        if "python" in available_tools:
            tools_description += """
- <python> </python>: Execute Python code. Use this for calculations, data manipulation, or algorithmic solutions.
"""
            
        if "search" in available_tools:
            tools_description += """
- <search> </search>: Search for information on the web. Provide a specific query to get relevant information.
"""
            
        return f"""You are an AI assistant that can use tools to help answer questions.
You have access to the following tools:
{tools_description}

When you use a tool, write your request within the appropriate tags.
The output will be provided within <output> </output> tags.

When you have the final answer, provide it within <answer> </answer> tags.

Question: {question}

Let me solve this step by step:
"""

    @staticmethod
    def get_qa_prompt(question: str) -> str:
        """
        Generate a prompt for general QA tasks.
        
        Args:
            question: The question to answer
            
        Returns:
            Formatted prompt
        """
        return f"""You are an AI assistant that answers questions accurately and helpfully.
You can use tools when needed to provide the most accurate information.

To search for information, use <search> </search> tags.
To run Python code, use <python> </python> tags.
Tool outputs will be provided within <output> </output> tags.

When you have the final answer, provide it within <answer> </answer> tags.

Question: {question}

Let me answer this step by step:
"""
