import re
import os
import json
from typing import Optional
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.crosswords import standard_prompt, cot_prompt, output_prompt, propose_prompt, value_prompt, value_prompts_ensemble
from tot.models import gpt as _default_gpt

class MiniCrosswordsEnv:
    def __init__(self, file='mini0505.json'):
        self.file = os.path.join(DATA_PATH, 'crosswords', file)

        self.file = json.load(open(self.file))
        self.n = len(self.file)
        self.cache = {}
        self.idx = None
        self.times = 0
        self.prompt_status_cache = {}

    def __len__(self):
        return self.n
    
    def reset(self, idx, board=None, status=None, steps=None):
        self.idx = idx
        self.data, self.board_gt = self.file[idx]
        self.board = ['_'] * 25
        self.ans = ['_____'] * 10
        self.ans_gt = self.get_ans(self.board_gt)
        self.steps = 0
        self.status = [0] * 10  # 0: unfilled; 1: filled; 2: filled then changed
        if board is not None:
            self.board = board
            self.ans = self.get_ans(self.board)
        if status is not None:
            self.status = status
        if steps is not None:
            self.steps = steps
        return self.render()
    

    def prompt_status(self):
        count = {'sure': 0, 'maybe': 0, 'impossible': 0}
        for ans, data, status in zip(self.ans, self.data, self.status):
            # if status != 0: continue
            if ans.count('_') >= 4: continue
            ans = ' '.join(ans.lower())
            line = f'{data}: {ans}'
            prompt = value_prompt.format(input=line)
            if prompt in self.prompt_status_cache:
                res = self.prompt_status_cache[prompt]
            else:
                res = _default_gpt(prompt)[0]
                self.prompt_status_cache[prompt] = res
            # print(line)
            # print(res)
            # print()
            res = res.split('\n')[-1].strip()
            if res in count: count[res] += 1
        # print(count)
        return count
    
    def render_gt_board(self):
        s = "GT Board:\n"
        for i in range(5):
            s += ' '.join(self.board_gt[i*5:(i+1)*5]) + '\n'
        return s
    
    def render_board(self):
        s = "Current Board:\n"
        for i in range(5):
            s += ''.join(self.board[i*5:(i+1)*5]) + '\n'
        return s

    def render_clues(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + '\n'
        return s
    
    def render_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans[i] + '\n'
        return s
    
    def render_gt_ans(self, status=None):
        s = ""
        # s += "Horizontal:\n"
        for i in range(5):
            if status is None or self.status[i] == status:
                s += 'h' + str(i+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        # s += "Vertical:\n"
        for i in range(5, 10):
            if status is None or self.status[i] == status:
                s += 'v' + str(i-5+1) + '. ' + self.data[i] + ': ' + self.ans_gt[i] + '\n'
        return s

    def render(self, status=True):
        if status:
            return self.render_board() + '\nUnfilled:\n' + self.render_ans(status=0) + '\nFilled:\n' + self.render_ans(status=1) + '\nChanged:\n' + self.render_ans(status=2)
        else:
            return self.render_board() + '\n' + self.render_ans()
    
    def get_ans(self, board):
        ans = [''] * 10
        for i in range(5):
            ans[i] = ''.join(board[i*5:(i+1)*5])
        for i in range(5):
            ans[i+5] = ''.join(board[i::5])
        return ans
    
    def step(self, action):
        self.steps += 1
        action = action.split('\n')[-1]
        action = action.split('. ')
        if len(action) != 2:
            return 'Invalid! Format should be like "h1. apple"', 0, False, {}
        pos, word = action

        if len(word) != 5:
            return 'Invalid! Word should have 5 letters.', 0, False, {}
        if pos.startswith('h'):
            idx = int(pos[1:]) - 1
            self.board[idx*5:(idx+1)*5] = list(word.upper())
        elif pos.startswith('v'):
            idx = int(pos[1:]) - 1
            self.board[idx::5] = list(word.upper())
            idx += 5  # for later status update
        else:
            return 'Invalid! Position should be h1-h5 or v1-v5', 0, False, {}
        
        self.new_ans = self.get_ans(self.board)
        # self.status = [2 if (status == 1 and ans != new_ans) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status = [2 if any(letter != new_letter and letter != '_' for letter, new_letter in zip(ans, new_ans)) else status for status, ans, new_ans in zip(self.status, self.ans, self.new_ans)]
        self.status[idx] = 1
        self.ans = self.new_ans
        r_all = (self.board == self.board_gt)
        r_letter = sum(a == b for a, b in zip(self.board, self.board_gt)) / 25
        r_word = sum(a == b for a, b in zip(self.ans, self.ans_gt)) / 10
        return self.render(), r_all, (r_all or self.steps >= 20), {'r_letter': r_letter, 'r_word': r_word, 'r_game': r_all}


class MiniCrosswordsTask(Task):
    """
    Input (x)   : Description of a 5x5 mini crossword
    Output (y)  : List of 10 words to fill in the crossword
    Reward (r)  : word level and game level
    """
    def __init__(self, file='mini0505.json'):
        super().__init__()
        self.env = MiniCrosswordsEnv(file)
        self.xs = []
        for idx in range(len(self.env)):
            self.env.reset(idx)
            self.xs.append(self.env.render_clues())
        self.steps = 10
        self.cache_proposals = {}
        self.value_cache = {}
        self._gpt_fn = _default_gpt

    def set_gpt_fn(self, fn):
        self._gpt_fn = fn

    def __len__(self) -> int:
        return len(self.env)

    def get_input(self, idx: int) -> str:
        self.env.reset(idx)
        return self.env.render_clues()

    def test_output(self, idx: int, output: str):
        self.env.reset(idx)
        grid_rows = self._extract_grid_rows(output)
        info = {'r_word': 0, 'r_letter': 0, 'r_game': 0}
        for i, line in enumerate(grid_rows, 1):
            line = line.strip()
            if re.match(r'^[a-zA-Z]{5}$', line):
                word = line
            else:
                word = ''.join(line.split(' ')[:5])
            word = (word + '_____')[:5]
            _, _, _, info = self.env.step(f'h{i}. {word}')
        info['r'] = info.get('r_word', 0)
        return info

    @staticmethod
    def _parse_json_board(output: str) -> list:
        """Extract h1-h5 answers from JSON output, return list of 5 uppercase rows."""
        # Try full JSON objects first (normal case)
        candidates = re.findall(r'\{[^{}]*\}', output, flags=re.DOTALL)
        for candidate in reversed(candidates):
            try:
                data = json.loads(candidate)
                if not all(f'h{i}' in data for i in range(1, 6)):
                    continue
                rows = []
                for i in range(1, 6):
                    word = re.sub(r'[^a-zA-Z]', '', str(data[f'h{i}']))[:5].upper()
                    rows.append((word + '_____')[:5])
                return rows
            except (json.JSONDecodeError, TypeError, ValueError):
                continue

        # Fallback: extract key-value pairs individually.
        # Handles prefill-started output like: MOTOR", "h2": "ITEMS", ...}
        rows = []
        for i in range(1, 6):
            m = re.search(rf'"?h{i}"?\s*:\s*"([A-Za-z]{{5}})"', output, re.IGNORECASE)
            rows.append(m.group(1).upper() if m else '_____')
        if any(r != '_____' for r in rows):
            return rows
        return []

    @staticmethod
    def _clean_board(output: str) -> list:
        """Extract up to 5 valid grid rows (exactly 5 alpha chars) from model output."""
        board = []
        for line in output.strip().splitlines():
            line = line.strip().lower()
            if len(line) == 5 and line.isalpha():
                board.append(line)
        return board[:5]

    @staticmethod
    def _extract_grid_rows(output: str) -> list:
        """Find 5 consecutive grid rows: compact (abcde) or spaced (A B C D E)."""
        normalized = output.replace('\xa0', ' ').replace('\u2009', ' ').replace('\u200b', '')
        lines = normalized.splitlines()

        compact = [(i, ln.strip()) for i, ln in enumerate(lines) if re.match(r'^[a-zA-Z]{5}$', ln.strip())]
        for k in range(len(compact) - 4):
            idxs = [compact[k + j][0] for j in range(5)]
            if idxs == list(range(idxs[0], idxs[0] + 5)):
                return [compact[k + j][1] for j in range(5)]

        spaced = [(i, ln.strip()) for i, ln in enumerate(lines) if re.match(r'^[A-Z] [A-Z] [A-Z] [A-Z] [A-Z]$', ln.strip())]
        for k in range(len(spaced) - 4):
            idxs = [spaced[k + j][0] for j in range(5)]
            if idxs == list(range(idxs[0], idxs[0] + 5)):
                return [spaced[k + j][1] for j in range(5)]

        return MiniCrosswordsTask._clean_board(output) or normalized.split('Output:')[-1].strip().split('\n')[-5:]
    def _apply_history(self, idx: int, y: str):
        """Replay BFS word-placement history (lines like 'h1. motor') into self.env."""
        self.env.reset(idx)
        for line in y.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('h') or line.startswith('v')):
                self.env.step(line)

    def set_status(self, x: str, y: str):
        idx = self.xs.index(x)
        self._apply_history(idx, y)

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def output_prompt_wrap(x: str, y: str = '') -> str:
        return output_prompt.format(input=x)

    def propose_prompt_wrap(self, x: str, y: str = '') -> str:
        idx = self.xs.index(x)
        self._apply_history(idx, y)
        return propose_prompt.format(input=self.env.render_ans())

    def propose_outputs_unwrap(self, x: str, y: str, outputs: list, n_max_propose: int) -> list:
        confidence_to_value = {'certain': 1, 'high': 0.5, 'medium': 0.2, 'low': 0.1}
        proposals_to_scores = {}
        # Use re.search so proposals are found anywhere in the text (incl. reasoning output)
        pattern = re.compile(
            r'\b([hv][1-5])[\.\:]\s*([a-zA-Z]{5})\b(?:[^\S\n]*\((certain|high|medium|low)\))?',
            re.IGNORECASE
        )
        for output in outputs:
            for m in pattern.finditer(output):
                proposal = m.group(1).lower() + '. ' + m.group(2).lower()
                conf = m.group(3).lower() if m.group(3) else 'medium'
                score = confidence_to_value.get(conf, 0.2)
                proposals_to_scores[proposal] = proposals_to_scores.get(proposal, 0) + score

        proposals = sorted(proposals_to_scores.items(), key=lambda p: p[1], reverse=True)
        if n_max_propose != -1:
            proposals = proposals[:n_max_propose]
        return [y + proposal[0] + '\n' for proposal in proposals]

    def evaluate(self, x: str, y: str, n_evaluate_sample: int) -> float:
        cache_key = (x, y)
        if cache_key in self.value_cache:
            return self.value_cache[cache_key]

        prompts = self.get_ensemble_prompts(x, y)
        scores = []
        for prompt in prompts:
            res = self._gpt_fn(prompt)[0]
            score = self.extract_numerical_score(res)
            if score is not None:
                scores.append(score)

        result = float(sum(scores) / len(scores)) if scores else 1.0
        self.value_cache[cache_key] = result
        return result

    def finalize_output(self, x: str, y: str) -> str:
        idx = self.xs.index(x)
        self._apply_history(idx, y)

        prompt = self.output_prompt_wrap(x, y)
        raw = self._gpt_fn(prompt)[0]
        rows = self._parse_json_board(raw)
        if len(rows) == 5:
            return '\n'.join(rows)

        # Fallback: use whatever BFS placed on the board
        board = self.env.board
        return '\n'.join(''.join(board[i * 5:(i + 1) * 5]).upper() for i in range(5))

    def get_ensemble_prompts(self, x: str, y: str) -> list:
        idx = self.xs.index(x)
        self._apply_history(idx, y)

        if self.env.board == self.env.board_gt:
            return ['{"reasoning": "Board solved", "score": 10}'] * len(value_prompts_ensemble)

        state = self.env.render_ans()
        return [prompt.format(input=state) for prompt in value_prompts_ensemble]

    def extract_numerical_score(self, output: str) -> Optional[float]:
        if not output:
            return None

        cleaned = re.sub(r'```json|```', '', output).strip()
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict) and "score" in data:
                score = float(data["score"])
                if 1 <= score <= 10:
                    return score
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        candidates = re.findall(r'\{[^{}]*\}', output, flags=re.DOTALL)
        for candidate in reversed(candidates):
            try:
                data = json.loads(candidate)
                if isinstance(data, dict) and "score" in data:
                    score = float(data["score"])
                    if 1 <= score <= 10:
                        return score
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

        match = re.search(r'"score"\s*:\s*"?(\d+)"?\b', output)
        if match:
            try:
                score = float(match.group(1))
                if 1 <= score <= 10:
                    return score
            except ValueError:
                pass

        return None
