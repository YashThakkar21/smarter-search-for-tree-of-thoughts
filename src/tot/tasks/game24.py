import re
import json
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import *


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    if 'left: ' not in last_line:
        return last_line  # y is just the raw input numbers
    return last_line.split('left: ')[-1].split(')')[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example: 
        1 2 3 4
    Output Example: 
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """
    def __init__(self, file='24.csv'):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, '24', file)
        self.data = list(pd.read_csv(path)['Puzzles'])
        self.value_cache = {}
        self.steps = 4
        self.stops = ['\n'] * 4

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output(self, idx: int, output: str):
        expression = output.strip().split('\n')[-1].lower().replace('answer: ', '').split('=')[0]
        numbers = re.findall(r'\d+', expression)
        problem_numbers = re.findall(r'\d+', self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {'r': 0}
        try:
            # print(sympy.simplify(expression))
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {'r': 0}

    def get_ensemble_prompts(self, x: str, y: str) -> list[str]:
        # 1. Extract the current remaining numbers
        last_line = y.strip().split('\n')[-1] if y.strip() else ""
        if 'left: ' in last_line:
            current_numbers = last_line.split('left: ')[1].split(')')[0].strip()
        else:
            current_numbers = x.strip()

        # 2. Check for hardcoded terminal states (NOW RETURNS JSON!)
        if current_numbers == '24':
            return ['{"reasoning": "Already solved", "score": 10}'] * len(value_prompts_ensemble)
        if current_numbers == 'impossible':
            return ['{"reasoning": "Dead end", "score": 1}'] * len(value_prompts_ensemble)

        # 3. Format and return the ensemble
        return [prompt.format(input=current_numbers) for prompt in value_prompts_ensemble]


    def extract_numerical_score(self, output: str) -> float:
        import json
        import re

        # 1. Try to parse as strict JSON
        try:
            clean_output = re.sub(r'```json|```', '', output).strip()
            data = json.loads(clean_output)
            if "score" in data:
                return float(data["score"])
        except Exception:
            pass

        # 2. Fallback: Regex hunt for the JSON score key if parsing fails
        match = re.search(r'"score"\s*:\s*(10|[1-9])', output, re.IGNORECASE)
        if match:
            return float(match.group(1))

        # 3. Ultimate Fallback
        return 1.0

    @staticmethod
    def standard_prompt_wrap(x: str, y: str='') -> str:
        raise NotImplementedError("standard_prompt not used in this configuration")

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        raise NotImplementedError("cot_prompt not used in this configuration")

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == '24':
            # Return a valid prompt - the propose shortcut will call _propose_steps('24')
            # which returns empty steps (no combinations from a single number)
            return propose_prompt.format(input='24')
        return propose_prompt.format(input=current_numbers)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:  # last step
            ans = last_line.lower().replace('answer: ', '')
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value