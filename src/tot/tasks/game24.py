import re
import os
import sympy
import pandas as pd
from fractions import Fraction
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import *

def get_current_numbers(y: str) -> str:
    last_line = y.strip().split('\n')[-1]
    if 'left: ' not in last_line:
        return last_line
    return last_line.split('left: ')[-1].split(')')[0]

class Game24Task(Task):
    def __init__(self, file='24.csv'):
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
            return {'r': int(sympy.simplify(expression) == 24)}
        except Exception:
            return {'r': 0}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str='') -> str:
        raise NotImplementedError("standard_prompt not used in this configuration")

    @staticmethod
    def cot_prompt_wrap(x: str, y: str='') -> str:
        raise NotImplementedError("cot_prompt not used in this configuration")

    @staticmethod
    def propose_prompt_wrap(x: str, y: str='') -> str:
        current_numbers = get_current_numbers(y if y else x)
        return propose_prompt.format(input=current_numbers)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split('\n')[-1]
        if 'left: ' not in last_line:
            ans = last_line.lower().replace('answer: ', '')
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        if len(y.strip().split('\n')) == 4 and 'answer' not in y.lower():
            return 0
        value_names = [_.split('\n')[-1] for _ in value_outputs]
        value_map = {'impossible': 0.001, 'likely': 1, 'sure': 20}
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value

    @staticmethod
    def _parse_fraction(num_str: str) -> Fraction:
        return Fraction(num_str)

    @staticmethod
    def finalize_output(x: str, y: str) -> str:
        lines = [line.strip() for line in y.strip().split('\n') if line.strip()]
        if not lines:
            return y

        if any(line.lower().startswith('answer:') for line in lines):
            return y

        num_token = r'-?\d+(?:/\d+|\.\d+)?'
        step_pattern = re.compile(
            rf'^\s*({num_token})\s*([+\-*/])\s*({num_token})\s*=\s*({num_token})\s*\(left:[^)]*\)\s*$'
        )

        try:
            pool = [{"value": Game24Task._parse_fraction(tok), "expr": tok} for tok in x.strip().split()]
            valid_lines = []

            def pop_value(target):
                for i, item in enumerate(pool):
                    if item["value"] == target:
                        return pool.pop(i)
                raise ValueError(f"Could not find {target} in pool {pool}")

            for line in lines:
                match = step_pattern.match(line)
                if not match:
                    continue

                a_str, op, b_str, c_str = match.groups()
                a = Game24Task._parse_fraction(a_str)
                b = Game24Task._parse_fraction(b_str)
                c = Game24Task._parse_fraction(c_str)

                left_item = pop_value(a)
                right_item = pop_value(b)

                if op == '+':
                    calc = left_item["value"] + right_item["value"]
                elif op == '-':
                    calc = left_item["value"] - right_item["value"]
                elif op == '*':
                    calc = left_item["value"] * right_item["value"]
                elif op == '/':
                    calc = left_item["value"] / right_item["value"]
                else:
                    return y

                if calc != c:
                    return y

                expr = f'({left_item["expr"]} {op} {right_item["expr"]})'
                pool.append({"value": c, "expr": expr})
                valid_lines.append(line)

            if len(pool) == 1 and pool[0]["value"] == 24:
                answer_line = f'Answer: {pool[0]["expr"]} = 24'
                return '\n'.join(valid_lines + [answer_line])

            return y

        except Exception:
            return y