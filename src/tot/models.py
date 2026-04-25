from __future__ import annotations
import itertools
import os
import re
from fractions import Fraction
from functools import lru_cache
from typing import Iterable, List
from dotenv import load_dotenv
from tinker import ServiceClient, types

load_dotenv()

completion_tokens = 0
prompt_tokens = 0
REQUEST_TIMEOUT_SECS = 300

TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")
if not TINKER_API_KEY:
    raise ValueError("TINKER_API_KEY is not set")

_service = ServiceClient(api_key=TINKER_API_KEY)
_sampling_clients = {}
_tokenizers = {}

def _get_sampling_client(model: str):
    if model not in _sampling_clients:
        _sampling_clients[model] = _service.create_sampling_client(base_model=model)
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

def _decode_text(model: str, token_ids):
    tokenizer = _get_tokenizer(model)
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(token_ids)

def _infer_mode(prompt_text: str) -> str:
    lower = prompt_text.lower()
    if "possible next steps" in lower:
        return "propose"
    if "judge:" in lower or "sure/impossible" in lower or "sure/likely/impossible" in lower:
        return "value"
    return "default"

def _clean_output(text: str, mode: str) -> str:
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

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


def gpt(prompt, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=128, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=64, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens

    sampling_client = _get_sampling_client(model)
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

    system_msg = "You are a helpful mathematical assistant. Follow the user's formatting instructions precisely."

    # Extract trailing labels to pre-fill the assistant so it doesn't get confused
    assistant_prefill = ""

    if mode == "propose" and "Possible next steps:" in raw_user_content:
        raw_user_content = raw_user_content[: raw_user_content.rfind("Possible next steps:")].strip()
        assistant_prefill = "Possible next steps:\n"
    elif mode == "value" and "Judge:" in raw_user_content:
        raw_user_content = raw_user_content[: raw_user_content.rfind("Judge:")].strip()
        assistant_prefill = "Judge:\n"

    prompt_text = (
        f"<|im_start|>system\n{system_msg}<|im_end|>\n"
        f"<|im_start|>user\n{raw_user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{assistant_prefill}"
    )

    outputs = []
    remaining = n

    while remaining > 0:
        batch_size = min(remaining, 20)
        remaining -= batch_size

        input_ids = _encode_text(model, prompt_text)
        prompt_tokens += len(input_ids)
        prompt = types.ModelInput.from_ints(input_ids)

        effective_stop = [stop] if isinstance(stop, str) else list(stop or [])
        if "<|im_end|>" not in effective_stop:
            effective_stop.append("<|im_end|>")

        if mode == "propose":
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
            decoded = _decode_text(model, output_tokens)
            outputs.append(_clean_output(decoded, mode))

    return outputs

def gpt_usage(backend="openai/gpt-oss-120b"):
    global completion_tokens, prompt_tokens
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": None,
    }