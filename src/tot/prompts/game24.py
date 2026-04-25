# 1-shot completion
propose_prompt = '''Input: 2 8 8 14
Possible next steps:
2 + 8 = 10 (left: 8 10 14)
8 / 2 = 4 (left: 4 8 14)
14 + 2 = 16 (left: 8 8 16)
2 * 8 = 16 (left: 8 14 16)
8 - 2 = 6 (left: 6 8 14)
14 - 8 = 6 (left: 2 6 8)
14 /  2 = 7 (left: 7 8 8)
14 - 2 = 12 (left: 8 8 12)
Input: {input}
Possible next steps:
'''

value_prompt = '''Evaluate if given numbers can reach 24 (sure/likely/impossible)
10 14
10 + 14 = 24
sure
11 12
11 + 12 = 23
12 - 11 = 1
11 * 12 = 132
11 / 12 = 0.91
impossible
4 4 10
4 + 4 + 10 = 8 + 10 = 18
4 * 10 - 4 = 40 - 4 = 36
(10 - 4) * 4 = 6 * 4 = 24
sure
4 9 11
9 + 11 + 4 = 20 + 4 = 24
sure
5 7 8
5 + 7 + 8 = 12 + 8 = 20
(8 - 5) * 7 = 3 * 7 = 21
I cannot obtain 24 now, but numbers are within a reasonable range
likely
5 6 6
5 + 6 + 6 = 17
(6 - 5) * 6 = 1 * 6 = 6
I cannot obtain 24 now, but numbers are within a reasonable range
likely
10 10 11
10 + 10 + 11 = 31
(11 - 10) * 10 = 10
10 10 10 are all too big
impossible
1 3 3
1 * 3 * 3 = 9
(1 + 3) * 3 = 12
1 3 3 are all too small
impossible
{input}
'''

value_last_step_prompt = '''Use numbers and basic arithmetic operations (+ - * /) to obtain 24. Given an input and an answer, give a judgement (sure/impossible) if the answer is correct, i.e. it uses each input exactly once and no other numbers, and reach 24.
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) = 24
Judge:
sure
Input: 2 9 10 12
Answer: 2 * 12 * (10 - 9) = 24
Judge:
sure
Input: 4 9 10 13
Answer: (13 - 9) * (10 - 4) = 24
Judge:
sure
Input: 4 4 6 8
Answer: (4 + 8) * (6 - 4) + 1 = 25
Judge:
impossible
Input: 2 9 10 12
Answer: 2 * (12 - 10) = 24
Judge:
impossible
Input: 4 9 10 13
Answer: (13 - 4) * (10 - 9) = 24
Judge:
impossible
Input: {input}
Answer: {answer}
Judge:'''

# --- NEW ENSEMBLE PROMPTS ---
# value_prompt_v1 = """Evaluate if the following numbers can be combined using basic arithmetic (+, -, *, /) to reach the target number 24.
# You may think step-by-step. When you are done, you MUST provide a single integer score from 1 to 10 inside XML tags. 10 is absolutely sure, 1 is impossible.
# Example: <score>8</score>

# Input: {input}
# """

# value_prompt_v2 = """Look at the given set of numbers. How mathematically likely is it that they can evaluate to 24 using addition, subtraction, multiplication, and division?
# Rate the potential of these numbers on a scale of 1 to 10 (1 = dead end, 10 = guaranteed solution).
# Write out your reasoning, then place your final integer score inside <score> tags.

# Input: {input}
# """

# value_prompt_v3 = """Act as a mathematical evaluator. Analyze the remaining numbers below. Are there clear algebraic paths to make 24?
# Assign a probability score from 1 to 10 representing the likelihood of success. You can show your work, but your final answer MUST be an integer wrapped in <score> tags.

# Input: {input}
# Score:"""

value_prompt_v1 = """Evaluate if the following numbers can be combined using basic arithmetic (+, -, *, /) to reach 24.

You MUST respond using this EXACT JSON format:
{{
  "reasoning": "your brief step-by-step work here",
  "score": <integer from 1 to 10>
}}

CRITICAL SCORING RULE: 10 means a valid expression to 24 is guaranteed. 1 means it is mathematically impossible.

Input: {input}
Output:"""

value_prompt_v2 = """Look at the given set of numbers. How mathematically likely is it that they can evaluate to 24 using addition, subtraction, multiplication, and division?

You MUST respond using this EXACT JSON format:
{{
  "reasoning": "briefly evaluate the potential in 3 sentences",
  "score": <integer from 1 to 10>
}}
(1 = dead end, 10 = guaranteed solution)

Input: {input}
Output:"""

value_prompt_v3 = """Act as a mathematical evaluator. Analyze the remaining numbers below. Are there clear algebraic paths to make 24? 

You MUST respond using this EXACT JSON format:
{{
  "reasoning": "your evaluation here",
  "score": <integer from 1 to 10>
}}
(10 is certain success, 1 is absolute failure)

Input: {input}
Output:"""

value_prompts_ensemble = [value_prompt_v1, value_prompt_v2, value_prompt_v3]