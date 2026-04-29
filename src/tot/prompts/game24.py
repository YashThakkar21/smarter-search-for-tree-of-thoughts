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

_SCORING_RULE = """SCORING RULE (this is about the PUZZLE, not your confidence):
  1-2  = clearly impossible (numbers can't combine to anywhere near 24)
  3-4  = unlikely; no obvious path but you can't fully rule it out
  5-6  = genuinely uncertain; some plausible directions, can't verify
  7-8  = looks promising; near-misses or partial paths exist
  9-10 = a valid expression to 24 definitely exists
Use the full 1-10 range. Do not default to only 1 or 10."""

_JSON_FORMAT = """You MUST respond using this EXACT JSON format and nothing else:
{{
  "reasoning": "your brief work here",
  "score": <integer from 1 to 10>
}}"""

value_prompt_v1 = f"""Evaluate whether the following numbers can be combined using basic arithmetic (+, -, *, /) and parentheses to reach 24.
 
{_SCORING_RULE}
 
{_JSON_FORMAT}
 
Input: {{input}}
Output:"""

value_prompt_v2 = f"""Estimate how likely it is that the following numbers can be combined with +, -, *, / and parentheses to evaluate to 24.
 
{_SCORING_RULE}
 
{_JSON_FORMAT}
 
Input: {{input}}
Output:"""

value_prompt_v3 = f"""Act as a mathematical evaluator. Analyze the remaining numbers below and decide whether there is a clear algebraic path to 24.
 
{_SCORING_RULE}
 
{_JSON_FORMAT}
 
Input: {{input}}
Output:"""

value_prompts_ensemble = [value_prompt_v1, value_prompt_v2, value_prompt_v3]