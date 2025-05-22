import json
import re

def load_alignment_results(filepath: str, formatted = True) -> dict:
    with open(filepath, 'r') as f:
        alignment_results = json.load(f)

    if formatted:
        return format_alignment_results(alignment_results)

    return alignment_results

def format_alignment_results(alignment_results: dict) -> dict:
    """
        Remove the 'parameters' key from the alignment_results
        and return a dictionary with the problem_id as the key
        the value is 
        {
            'response': ['choices'][0]['message']['content'],
            'finish_reason': ['choices'][0]['finish_reason']
        }
        if there is an error, the value is
        {
            'error': str(e)
        }
    """
    formatted_results = {}
    for problem_id, result in alignment_results.items():
        if problem_id == 'parameters':
            continue
        if 'error' in result:
            formatted_results[problem_id] = {
                'error': result['error']
            }
            continue

        formatted_results[problem_id] = {
            'response': result['choices'][0]['message']['content'],
            'finish_reason': result['choices'][0]['finish_reason']
        }

    return formatted_results


def convert_to_plaintext(text : str) -> str:
    """
        Remove latex and markdown syntax and command from text
    """
    # Convert \\text{...} to ...
    text = re.sub(r"\\text\{(.*?)\}", r"\1", text)

    # Convert
    # \begin{...}
    # \end{...}
    # to 
    # ...
    # ...
    text = re.sub(r"\\begin\{.*?\}", "", text)
    text = re.sub(r"\\end\{.*?\}", "", text)

    # Remove \left, \right
    text = re.sub(r"\\left", "", text)
    text = re.sub(r"\\right", "", text)
    
    # Replace \( with ( and \) with )
    text = re.sub(r"\\\(", "(", text)
    text = re.sub(r"\\\)", ")", text)
    # Remove \\[, \\], \\{, \\}
    text = re.sub(r"\\[\[\{\]\}]", "", text)
    # Remove \\
    text = re.sub(r"\\\\", "", text)

    # Remove &
    text = re.sub(r"&", "", text)

    return text



def convert_possible_logic_forms_in_line(line : str):
    """
        Convert some obvious logic forms in wrong format to correct format in a line
    """    
    # 1. Convert xxx = yyy to Equals(xxx, yyy)
    # xxx and yyy can be logic form, and an expression with spaces in between
    line = re.sub(r"^(.+)\s*=\s*(.+)$", r"Equals(\1, \2)", line)

    # 2, Convert Equals(AB, CD) to Equals(LengthOf(Line(A, B)), LengthOf(Line(C, D)))
    line = re.sub(r"Equals\(\s*([A-Z])([A-Z])\s*,\s*([A-Z])([A-Z])\s*\)", r"Equals(LengthOf(Line(\1, \2)), LengthOf(Line(\3, \4)))", line)

    # 3. Convert Equals(AB, number/var) to Equals(LengthOf(Line(A, B)), number/var)
    # a variable is of form [a-z\_]+
    line = re.sub(r"Equals\(\s*([A-Z])([A-Z])\s*,\s*([a-z\_]+|\-?\d+\.?\d*)\s*\)", r"Equals(LengthOf(Line(\1, \2)), \3)", line)

    # 4. Convert Equals(number/var, AB) to Equals(number/var, LengthOf(Line(A, B)))
    line = re.sub(r"Equals\(\s*([a-z\_]+|\-?\d+\.?\d*)\s*,\s*([A-Z])([A-Z])\s*\)", r"Equals(\1, LengthOf(Line(\2, \3)))", line)

    return line


def convert_possible_logic_forms(text : str):
    """
        Convert some obvious logic forms in wrong format to correct format
    """
    lines = text.split('\n')
    for i, line in enumerate(lines):
        lines[i] = convert_possible_logic_forms_in_line(line)

    return '\n'.join(lines)

def find_logic_forms(text):
    """
        Find possible logic forms with loose constraints.
    """
    results = []
    stack = []
    start_idx = None

    for i, char in enumerate(text):
        if char == '*':
            # A line started with * is considered as a comment
            while i < len(text) and text[i] != '\n':
                i += 1
            continue
        if char.isalpha() and char.isupper() and (i == 0 or not text[i-1].isalnum()) and i+1 < len(text) and text[i+1].isalnum():
            if not stack:
                start_idx = i
        elif char == '(':
            if start_idx is not None:
                stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
                if not stack and start_idx is not None:
                    expr = text[start_idx:i+1]
                    results.append(expr)
                    start_idx = None
            else:
                # If there is no left parentheses, reset the start_idx
                start_idx = None
        elif char == '\n' or char == '\\':
            # A logic form is not allowed to break lines
            # If we have a start_idx, try to match the parentheses and give a possible logic form
            if start_idx is not None and stack:
                # Add right parentheses to the end of the text
                expr = text[start_idx:i] + ')' * len(stack)
                results.append(expr)

            # Reset the start_idx and stack
            start_idx = None
            stack = []

    
    # Remove duplicates
    results = list(set(results))
    return results


def get_logic_forms_from_response(s : str) -> list[str]:
    """
        Get possible logic forms from response
        This function use simple test to format possible logic forms from the string 's'
        The format of logic forms is not guaranteed to be correct
    """
    # Convert to plaintext
    s = convert_to_plaintext(s)
    # Convert possible logic forms
    s = convert_possible_logic_forms(s)
    # Find logic forms
    logic_forms = find_logic_forms(s)
    # Remove the space
    logic_forms = [expr.replace(' ', '') for expr in logic_forms]
    # Remove duplicates
    logic_forms = list(set(logic_forms))
    return logic_forms


def get_logic_forms_from_code_block_only(s : str, num = 1) -> list[str]:
    """
        Get possible logic forms from code block only
        This is used when the response is well formatted in a code block
    """
    # The code block is in the format ```plaintext...```
    if num == 0:
        # Match the content in the all code blocks
        code_block = re.compile(r"```plaintext(.*?)```", re.DOTALL)
        logic_forms = []
        for match in code_block.finditer(s):
            code_block = match.group(1)
            code_block = convert_possible_logic_forms(code_block)
            logic_forms.extend(find_logic_forms(code_block))
    else:
        # Match the content in the last num code blocks
        code_block = re.compile(r"```plaintext(.*?)```", re.DOTALL)
        logic_forms = []
        matches = list(code_block.finditer(s))
        if len(matches) < num:
            num = len(matches)

        for match in matches[-num:]:
            code_block = match.group(1)
            code_block = convert_possible_logic_forms(code_block)
            logic_forms.extend(find_logic_forms(code_block))

    return logic_forms


def get_code_block_content(s : str) -> list[str]:
    """
        Get all content from code blocks
    """
    # The code block is in the format ```alphabet...```
    # Match the content in the code block
    code_block = re.compile(r"```([a-z]*)(.*?)```", re.DOTALL)
    code_blocks = []
    for match in code_block.finditer(s):
        code_blocks.append(match.group(2))

    return code_blocks



def clean(response):
    """
        Remove all markdown syntax and command from the response
    """
    response = re.sub(r"\*\*(.*?)\*\*", r"\1", response)
    response = re.sub(r"\*(.*?)\*", r"\1", response)
    response = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", response)
    response = re.sub(r"\n\n", r"\n", response)

    # replace ``` with """
    response = re.sub(r"```", r'"""', response)
    # Remove all latex syntax
    # replace \text{...} with ...
    response = re.sub(r"\\text\{(.*?)\}", r"\1", response, flags=re.DOTALL)
    # replace \[...\] with ...
    response = re.sub(r"\\\[(.*?)\\\]", r"\1", response, flags=re.DOTALL)
    # replace \(...\) with ...
    response = re.sub(r"\\\((.*?)\\\)", r"\1", response, flags=re.DOTALL)
    # remove - 
    response = re.sub(r"-\ ", r"", response)
    
    return response


def match_problem_answer(response):
    response = clean(response)
    response = response.replace('\n','').replace(' ', '')
    response = response[::-1]
    # 1. The answer is in the format ``answer``
    answer = re.search(r"``(.*?)``", response, re.DOTALL)
    if not answer:
        # 2. The answer is in the format \boxed{...}
        # The answer maybe in the format \boxed{...} - this is preferrable to qwen model
        # The reverse string is in the format \}...{dexob\
        answer = re.search(r"\}(.*?)\{dexob\\", response, re.DOTALL)

    if not answer:
        # 3. answer: xxxx
        # The answer is in the format Answer: xxxx or answer: xxxx
        # The reverse string is in the format :xxxx:rewsna
        answer = re.search(r"\b(\w|\d*\.\d+|\d+):rewsna", response, re.DOTALL | re.IGNORECASE)
    
    if not answer:
        # 4. answer is: xxxx
        # The answer is in the format Answer is: xxxx or answer is: xxxx
        # The reverse string is in the format :xxxx\s:si rewsna
        answer = re.search(r"\b(\w|\d*\.\d+|\d+):sirewsna", response, re.DOTALL | re.IGNORECASE)

    if not answer:
        # 5. choice: xxxx
        # The reverse string is in the format :xxxx:eciohc
        answer = re.search(r"\b(\w|\d*\.\d+|\d+):eciohc", response, re.DOTALL | re.IGNORECASE)
        
    if not answer:
        # 6. choice is: xxxx
        # The reverse string is in the format :xxxx:si eciohc
        answer = re.search(r"\b(\w|\d*\.\d+|\d+):sieciohc", response, re.DOTALL | re.IGNORECASE)

    if not answer:
        # 7. answer is xxxx 
        # The reverse string is in the format xxxxsirewsna
        answer = re.search(r"\b(\w|\d*\.\d+|\d+)sirewsna", response, re.DOTALL | re.IGNORECASE)
    
    if not answer:
        # 8. choice is xxxx
        # The reverse string is in the format xxxxsieciohc
        answer = re.search(r"\b(\w|\d*\.\d+|\d+)sieciohc", response, re.DOTALL | re.IGNORECASE)

    if not answer:
        # 9. answer = xxxx
        # The reverse string is in the format xxxx=rewsna
        answer = re.search(r"\b(\w|\d*\.\d+|\d+)=rewsna", response, re.DOTALL | re.IGNORECASE)

    if not answer:
        # 10. choice = xxxx
        # The reverse string is in the format xxxx=eciohc
        answer = re.search(r"\b(\w|\d*\.\d+|\d+)=eciohc", response, re.DOTALL | re.IGNORECASE)

    if not answer:
        #  11 """..."""
        answer = re.search(r'"""(.*?)"""', response, re.DOTALL)

    if answer is not None:
        # Remove the space and reverse it back
        return answer.group(1).strip()[::-1]

    return ""