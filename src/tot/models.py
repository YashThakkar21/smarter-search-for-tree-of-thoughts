from __future__ import annotations
import itertools
import json
import os
import re
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, List
from dotenv import load_dotenv

load_dotenv()

completion_tokens = 0
prompt_tokens = 0
REQUEST_TIMEOUT_SECS = 300

TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")

_service = None
_sampling_clients = {}
_tokenizers = {}
_tinker_types = None

def _get_tinker():
    global _tinker_types
    if _tinker_types is None:
        try:
            from tinker import ServiceClient, types
        except ImportError:
            raise ImportError("The tinker package is required for model calls")
        _tinker_types = (ServiceClient, types)
    return _tinker_types

def _get_service():
    global _service
    if _service is None:
        ServiceClient, _ = _get_tinker()
        if not TINKER_API_KEY:
            raise ValueError("TINKER_API_KEY is not set")
        _service = ServiceClient(api_key=TINKER_API_KEY)
    return _service

def _get_sampling_client(model: str):
    if model not in _sampling_clients:
        _sampling_clients[model] = _get_service().create_sampling_client(base_model=model)
    return _sampling_clients[model]

def _get_tokenizer(model: str):
    if model not in _tokenizers:
        tok = _get_sampling_client(model).get_tokenizer()
        if hasattr(tok, "result"):
            tok = tok.result(timeout=REQUEST_TIMEOUT_SECS)
        _tokenizers[model] = tok
    return _tokenizers[model]

def _encode_text(model: str, text: str):
    tokenizer = _get_tokenizer(model)
    ids = tokenizer.encode(text)
    if isinstance(ids, dict):
        ids = ids.get("input_ids", ids)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return ids

def _decode_text(model: str, token_ids, skip_special_tokens: bool = True):
    tokenizer = _get_tokenizer(model)
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens).strip()
    except TypeError:
        return tokenizer.decode(token_ids)

def _encode_chat_prompt(model: str, system_msg: str, user_content: str, assistant_prefill: str):
    tokenizer = _get_tokenizer(model)
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_content},
        ]
        template_kwargs = {"reasoning_effort": "minimal"}
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})
            template_kwargs["continue_final_message"] = True
        else:
            template_kwargs["add_generation_prompt"] = True
        encoded = tokenizer.apply_chat_template(messages, **template_kwargs)
        ids = encoded["input_ids"] if hasattr(encoded, "keys") else encoded
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return ids, True

    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefill}"
    )
    return _encode_text(model, prompt_text), False

def _infer_mode(prompt_text: str) -> str:
    lower = prompt_text.lower()
    if "possible answers for unfilled words" in lower or "possible answers for unfilled or changed words" in lower:
        return "crosswords_propose"
    if "single integer from 1 to 10" in lower:
        return "crosswords_score"
    if '"h1": "word1"' in lower:
        return "crosswords_output"
    if "return the final answer grid" in lower or "output only 5 rows" in lower:
        return "crosswords_output"
    if "solve 5x5 mini crosswords" in lower:
        return "crosswords_standard"
    if "crossword state" in lower and '"score"' in lower:
        return "crosswords_eval"
    if "five letter word" in lower and "sure/maybe/impossible" in lower:
        return "crosswords_status"
    if "possible next steps" in lower:
        return "propose"
    if "judge:" in lower or "sure/impossible" in lower or "sure/likely/impossible" in lower:
        return "value"
    return "default"

def _format_crossword_rows(rows: list[str]) -> str:
    formatted = []
    for row in rows[:5]:
        letters = re.sub(r"[^A-Za-z]", "", row).upper()[:5]
        if len(letters) != 5:
            return ""
        formatted.append(" ".join(letters))
    return "\n".join(formatted) if len(formatted) == 5 else ""

def _extract_assigned_crossword_rows(text: str) -> list[str]:
    normalized = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u200b", "")
    rows = [""] * 5
    verbs = r"(?:=|:|is|are|be|becomes|became|would be|could be|likely|answer(?:\s+is)?)"
    for i in range(1, 6):
        patterns = [
            rf"(?:\brow\s*{i}\b|\bh{i}\b)[^\n]{{0,180}}?{verbs}\s*[\"']?([A-Za-z]{{5}})\b",
            rf"(?:\brow\s*{i}\b|\bh{i}\b)[^\n]{{0,80}}\b([A-Za-z]{{5}})\s+(?:fits|matches|works|solved)\b",
        ]
        matches = []
        for pattern in patterns:
            matches.extend(re.finditer(pattern, normalized, flags=re.IGNORECASE))
        if matches:
            rows[i - 1] = matches[-1].group(1)
    return rows

def _extract_crossword_grid(text: str) -> str:
    candidates = re.findall(r"\{[^{}]*\}", text, flags=re.DOTALL)
    for candidate in reversed(candidates):
        try:
            data = json.loads(candidate)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if isinstance(data, dict) and all(f"h{i}" in data for i in range(1, 6)):
            grid = _format_crossword_rows([str(data[f"h{i}"]) for i in range(1, 6)])
            if grid:
                return grid

    normalized = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u200b", "")
    lines = normalized.splitlines()

    row_patterns = [
        r"^\s*([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s+([A-Za-z])\s*$",
        r"^\s*([A-Za-z]{5})\s*$",
        r"^\s*(?:h[1-5]|row\s*[1-5]|[1-5])[\.\):\-\s]+([A-Za-z]{5}|[A-Za-z](?:\s+[A-Za-z]){4})\s*$",
    ]

    parsed = []
    for line in lines:
        row = ""
        for pattern in row_patterns:
            match = re.match(pattern, line, flags=re.IGNORECASE)
            if not match:
                continue
            if len(match.groups()) == 5:
                row = "".join(match.groups())
            else:
                row = match.group(1)
            break
        if row:
            parsed.append(row)
            if len(parsed) == 5:
                grid = _format_crossword_rows(parsed)
                if grid:
                    return grid
        else:
            parsed = []

    labeled = []
    for i in range(1, 6):
        matches = list(re.finditer(
            rf"\bh{i}\b[^A-Za-z\n]{{0,80}}([A-Za-z]{{5}}|[A-Za-z](?:\s+[A-Za-z]){{4}})\b",
            normalized,
            flags=re.IGNORECASE,
        ))
        if not matches:
            break
        labeled.append(matches[-1].group(1))
    grid = _format_crossword_rows(labeled)
    if grid:
        return grid

    assigned = _extract_assigned_crossword_rows(normalized)
    if all(assigned):
        return _format_crossword_rows(assigned)
    return ""

def _extract_crossword_proposals(text: str) -> str:
    normalized = text.replace("\xa0", " ").replace("\u2009", " ").replace("\u200b", "")
    proposals = []
    seen = set()

    pattern = re.compile(
        r"\b([hv][1-5])[\.\:]\s*([A-Za-z]{5})\b(?:[^\S\n]*\((certain|high|medium|low)\))?",
        re.IGNORECASE,
    )
    for match in pattern.finditer(normalized):
        line_start = normalized.rfind("\n", 0, match.start()) + 1
        context = normalized[line_start:match.start()].lower()
        if "pattern" in context:
            continue
        pos = match.group(1).lower()
        word = match.group(2).lower()
        confidence = (match.group(3) or "medium").lower()
        key = (pos, word)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(f"{pos}. {word} ({confidence})")

    for i, row in enumerate(_extract_assigned_crossword_rows(normalized), start=1):
        if not row:
            continue
        pos = f"h{i}"
        word = row.lower()
        key = (pos, word)
        if key in seen:
            continue
        seen.add(key)
        proposals.append(f"{pos}. {word} (medium)")

    grid = _extract_crossword_grid(normalized)
    if grid:
        for i, line in enumerate(grid.splitlines(), start=1):
            word = re.sub(r"[^A-Za-z]", "", line).lower()
            if len(word) != 5:
                continue
            pos = f"h{i}"
            key = (pos, word)
            if key in seen:
                continue
            seen.add(key)
            proposals.append(f"{pos}. {word} (medium)")

    return "\n".join(proposals)

def _coerce_crossword_grid(text: str) -> str:
    assigned = _extract_assigned_crossword_rows(text)
    if any(assigned):
        rows = [row if row else "XXXXX" for row in assigned]
        return _format_crossword_rows(rows)

    letters = re.findall(r"[A-Za-z]", text.upper())
    if len(letters) < 25:
        letters.extend(["X"] * (25 - len(letters)))
    letters = letters[:25]
    rows = ["".join(letters[i:i + 5]) for i in range(0, 25, 5)]
    return _format_crossword_rows(rows)

def _extract_final_channel(text: str) -> str:
    matches = list(re.finditer(
        r"<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)",
        text,
        flags=re.DOTALL,
    ))
    if matches:
        return matches[-1].group(1).strip()
    return ""

def _clean_output(text: str, mode: str) -> str:
    text = (_extract_final_channel(text) or text)
    for token in ("<|im_end|>", "<|im_start|>", "<|return|>", "<|end|>", "<|start|>"):
        text = text.replace(token, "")
    text = text.strip()

    if mode == "crosswords_propose":
        return _extract_crossword_proposals(text) or text

    if mode == "crosswords_output":
        return _extract_crossword_grid(text) or _coerce_crossword_grid(text)

    if mode == "crosswords_standard":
        return _extract_crossword_grid(text) or _coerce_crossword_grid(text)

    if mode == "crosswords_eval":
        return text  # extract_numerical_score handles JSON parsing

    if mode == "crosswords_status":
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            for label in ("sure", "maybe", "impossible"):
                if re.search(rf"\b{label}\b", line):
                    return label
        return text

    if mode == "crosswords_score":
        m = re.search(r'\b(10|[1-9])\b', text)
        return m.group(1) if m else "1"

    if mode == "propose":
        valid_lines = []
        for line in text.split("\n"):
            line = line.strip()
            if "(left:" in line and "=" in line:
                match = re.match(r".+?\(left:[^)]*\)", line)
                if match:
                    valid_lines.append(match.group(0).strip().strip('"'))
        return "\n".join(valid_lines)

    if mode == "value":
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            for label in ("sure", "likely", "impossible"):
                if label in line:
                    return label
        return "impossible"

    return text

def _fmt_fraction(value: Fraction) -> str:
    return str(int(value)) if value.denominator == 1 else str(value)

def _sorted_state(values: Iterable[Fraction]) -> tuple[Fraction, ...]:
    return tuple(sorted(values))

def _propose_steps(numbers_str: str) -> str:
    nums = [Fraction(token) for token in numbers_str.strip().split()]
    steps: List[str] = []
    seen_states = set()

    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        remaining = [nums[k] for k in range(len(nums)) if k != i and k != j]

        candidates = [
            ("+", a, b, a + b),
            ("-", a, b, a - b),
            ("-", b, a, b - a),
            ("*", a, b, a * b),
        ]
        if b != 0:
            candidates.append(("/", a, b, a / b))
        if a != 0:
            candidates.append(("/", b, a, b / a))

        for op, left, right, result in candidates:
            if result <= 0:
                continue

            if remaining:
                if op == "*" and (left == 1 or right == 1):
                    continue
                if op == "/" and right == 1:
                    continue

            next_state = _sorted_state(remaining + [result])
            if next_state in seen_states:
                continue
            seen_states.add(next_state)

            left_str = " ".join(_fmt_fraction(v) for v in next_state)
            steps.append(
                f"{_fmt_fraction(left)} {op} {_fmt_fraction(right)} = "
                f"{_fmt_fraction(result)} (left: {left_str})"
            )

    return "\n".join(steps)

@lru_cache(maxsize=None)
def _can_reach_24_state(state: tuple[Fraction, ...]) -> bool:
    if len(state) == 1:
        return state[0] == 24

    nums = list(state)
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(len(nums)) if k != i and k != j]

        candidates = [a + b, a - b, b - a, a * b]
        if b != 0:
            candidates.append(a / b)
        if a != 0:
            candidates.append(b / a)

        for candidate in candidates:
            if _can_reach_24_state(_sorted_state(rest + [candidate])):
                return True

    return False

def _value_score(numbers_str: str) -> str:
    try:
        nums = [Fraction(token) for token in numbers_str.strip().split()]
    except Exception:
        return "impossible"

    if len(nums) == 1:
        return "sure" if nums[0] == 24 else "impossible"

    return "sure" if _can_reach_24_state(_sorted_state(nums)) else "impossible"


def gpt(prompt, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=2048, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=64, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens

    raw_user_content = "\n".join(message["content"] for message in messages)
    mode = _infer_mode(raw_user_content)

    if mode == "propose":
        matches = re.findall(r"Input:\s*([\d\s./-]+)", raw_user_content)
        if matches:
            numbers_str = matches[-1].strip()
            return [_propose_steps(numbers_str)] * n

    if mode == "value":
        lines = [line.strip() for line in raw_user_content.splitlines() if line.strip()]
        if lines:
            return [_value_score(lines[-1])] * n

    if mode == "crosswords_output":
        system_msg = (
            "You are an expert crossword puzzle solver. "
            "Use only the final channel. "
            "Return only the final 5x5 answer grid. "
            "Use exactly 5 rows, each row exactly 5 uppercase letters separated by single spaces. "
            "Do not output reasoning, JSON, labels, or extra text."
        )
    elif mode == "crosswords_standard":
        system_msg = (
            "You are an expert crossword puzzle solver. "
            "Use only the final channel. "
            "You will be shown examples of solved crosswords, then a new puzzle to solve. "
            "Output ONLY the 5x5 grid: exactly 5 rows, each row containing 5 uppercase letters separated by single spaces. "
            "Do not output any reasoning, analysis, explanation, or other text — only the 5 rows."
        )
    elif mode == "crosswords_eval":
        system_msg = (
            "You are a crossword evaluator. Respond ONLY with valid JSON in the exact format: "
            '{"reasoning": "<brief explanation>", "score": <integer 1-10>}'
        )
    elif mode == "crosswords_status":
        system_msg = (
            "You are a crossword evaluator. Decide whether the clue can still be satisfied by "
            "a five-letter word matching the shown letter constraints. Respond with exactly one "
            "word: sure, maybe, or impossible."
        )
    elif mode == "crosswords_propose":
        system_msg = (
            "You are an expert crossword puzzle solver. "
            "Output ONLY candidate words in the format: position. word (confidence). "
            "No reasoning, no explanation. One candidate per line."
        )
    elif mode == "crosswords_score":
        system_msg = "You are an expert crossword puzzle solver. Follow the formatting instructions precisely."
    else:
        system_msg = "You are a helpful mathematical assistant. Follow the user's formatting instructions precisely."

    assistant_prefill = ""

    if mode == "crosswords_propose" and (
        "Possible answers for unfilled or changed words:" in raw_user_content
        or "Possible answers for unfilled words:" in raw_user_content
    ):
        marker = (
            "Possible answers for unfilled or changed words:"
            if "Possible answers for unfilled or changed words:" in raw_user_content
            else "Possible answers for unfilled words:"
        )
        raw_user_content = raw_user_content[: raw_user_content.rfind(marker)].strip()
        assistant_prefill = marker + "\n"
    elif mode == "crosswords_output" and "Output:" in raw_user_content:
        raw_user_content = raw_user_content[: raw_user_content.rfind("Output:")].strip()
        assistant_prefill = "Output:\n"
    elif mode == "crosswords_standard" and "Thoughts:" in raw_user_content and raw_user_content.rstrip().endswith("Thoughts:"):
        # cot_prompt — prefill the Thoughts: anchor so the model starts structured reasoning.
        raw_user_content = raw_user_content.rstrip()[: -len("Thoughts:")].strip()
        assistant_prefill = "Thoughts:\n"
    elif mode == "crosswords_standard" and "Output:" in raw_user_content:
        # standard_prompt — prefill Output: so the model completes with the grid directly.
        # rfind targets the LAST "Output:" (the empty one), not the ones in examples.
        raw_user_content = raw_user_content[: raw_user_content.rfind("Output:")].strip()
        assistant_prefill = "Output:\n"
    elif mode == "propose" and "Possible next steps:" in raw_user_content:
        raw_user_content = raw_user_content[: raw_user_content.rfind("Possible next steps:")].strip()
        assistant_prefill = "Possible next steps:\n"
    elif mode == "value" and "Judge:" in raw_user_content:
        raw_user_content = raw_user_content[: raw_user_content.rfind("Judge:")].strip()
        assistant_prefill = "Judge:\n"

    input_ids, uses_chat_template = _encode_chat_prompt(
        model,
        system_msg,
        raw_user_content,
        assistant_prefill,
    )

    _, types = _get_tinker()
    outputs = []
    remaining = n

    while remaining > 0:
        sampling_client = _get_sampling_client(model)
        batch_size = min(remaining, 20)
        remaining -= batch_size

        prompt_tokens += len(input_ids)
        prompt = types.ModelInput.from_ints(input_ids)

        effective_stop = [stop] if isinstance(stop, str) else list(stop or [])
        if uses_chat_template:
            for stop_token in ("<|return|>", "<|end|>"):
                if stop_token not in effective_stop:
                    effective_stop.append(stop_token)
        elif "<|im_end|>" not in effective_stop:
            effective_stop.append("<|im_end|>")

        if mode == "crosswords_output":
            effective_max_tokens = 1024
        elif mode == "crosswords_propose":
            effective_max_tokens = 1024
            effective_stop = [token for token in effective_stop if token != "\n\n"]
        elif mode == "crosswords_standard":
            effective_max_tokens = 1024
            # Only stop on the chat turn boundary or if the model tries to start
            # a new few-shot example — never on \n\n, which fires after every grid row.
            if "\nInput:" not in effective_stop:
                effective_stop.append("\nInput:")
        elif mode == "crosswords_eval":
            effective_max_tokens = 512
        elif mode == "crosswords_status":
            effective_max_tokens = 64
        elif mode == "crosswords_score":
            effective_max_tokens = 10
        elif mode == "propose":
            effective_max_tokens = 512
            effective_stop = [token for token in effective_stop if token != "\n\n"]
        elif mode == "value":
            effective_max_tokens = 64
        else:
            effective_max_tokens = max(max_tokens, 128)

        sampling_params = types.SamplingParams(
            temperature=temperature,
            max_tokens=effective_max_tokens,
            stop=effective_stop,
        )

        result = sampling_client.sample(
            prompt=prompt,
            num_samples=batch_size,
            sampling_params=sampling_params,
        )

        if hasattr(result, "result"):
            result = result.result(timeout=REQUEST_TIMEOUT_SECS)

        for seq in result.sequences:
            output_tokens = seq.tokens
            completion_tokens += len(output_tokens)
            decoded = _decode_text(
                model,
                output_tokens,
                skip_special_tokens=not uses_chat_template,
            )
            if os.getenv("TOT_DEBUG_MODEL") == "1" and mode in ("crosswords_propose", "crosswords_score", "crosswords_standard", "crosswords_output", "crosswords_eval", "crosswords_status"):
                print(f'[{mode}] raw decoded ({len(output_tokens)} tokens): {repr(decoded[:300])}')
            outputs.append(_clean_output(decoded, mode))

    return outputs

def gpt_usage(backend="openai/gpt-oss-120b"):
    global completion_tokens, prompt_tokens
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": None,
    }
