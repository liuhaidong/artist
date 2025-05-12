# prompts/prompt_templates.py

class PromptTemplates:
    @staticmethod
    def get_math_reasoning_prompt(question):
        return f"""You are a helpful assistant that can solve complex math problems step by step with the help of a python executor tool. Given a question, you need to first think about the reasoning process in the mind and then provide the answer. During thinking, you can write python code, and invoke python tool to execute the code and get back the output of the code. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags respectively, and the python code and the output are enclosed within <python></python> and <output></output> tags respectively. You can utilize the Sympy library to write the python code and make sure to print the result at the end of the python code. You can utilize the python tool as many times as required, however each python code will be executed separately.

Question: {question}

"""

    @staticmethod
    def get_function_calling_prompt(question, available_functions):
        functions_str = ", ".join(available_functions)
        return f"""You are an expert in composing functions. You are given a question from a user and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose. If none of the functions can be used, point it out. If the given question lacks the parameters required by the function, also point it out.

Available functions: {functions_str}

For each step:
1. Start with a step-by-step thinking process inside <reasoning></reasoning> tags to think through the problem.
2. If needed, use tools by writing one or more JSON commands as a list inside <tool></tool> tags. Each item in the list should have a name and args key, with args being a dictionary.
example: <tool>[func_name1(params_name1=params_value1, params_name2=params_value2...), func_name2(params)]</tool>
Tools expect specific JSON input formats. Do not make up tools or arguments that aren't listed.
3. After you have used the tools, you will see the tool outputs inside <tool_result></tool_result> tags in the same order from the system.

Question: {question}

"""
