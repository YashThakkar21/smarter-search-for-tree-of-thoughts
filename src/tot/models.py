# from __future__ import annotations
# import itertools
# import os
# import re
# from fractions import Fraction
# from functools import lru_cache
# from typing import Iterable, List
# from dotenv import load_dotenv
# from tinker import ServiceClient, types

# load_dotenv()

# completion_tokens = 0
# prompt_tokens = 0
# REQUEST_TIMEOUT_SECS = 300

# TINKER_API_KEY = os.getenv("TINKER_API_KEY", "")
# if not TINKER_API_KEY:
#     raise ValueError("TINKER_API_KEY is not set")

# _service = ServiceClient(api_key=TINKER_API_KEY)
# _sampling_clients = {}
# _tokenizers = {}

# def _get_sampling_client(model: str):
#     if model not in _sampling_clients:
#         _sampling_clients[model] = _service.create_sampling_client(base_model=model)
#     return _sampling_clients[model]

# def _get_tokenizer(model: str):
#     if model not in _tokenizers:
#         tok = _get_sampling_client(model).get_tokenizer()
#         if hasattr(tok, "result"):
#             tok = tok.result(timeout=REQUEST_TIMEOUT_SECS)
#         _tokenizers[model] = tok
#     return _tokenizers[model]

# def _encode_text(model: str, text: str):
#     tokenizer = _get_tokenizer(model)
#     ids = tokenizer.encode(text)
#     if isinstance(ids, dict):
#         ids = ids.get("input_ids", ids)
#     if hasattr(ids, "tolist"):
#         ids = ids.tolist()
#     return ids

# def _decode_text(model: str, token_ids):
#     tokenizer = _get_tokenizer(model)
#     try:
#         return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
#     except TypeError:
#         return tokenizer.decode(token_ids)

# def _infer_mode(prompt_text: str) -> str:
#     lower = prompt_text.lower()
#     if "possible next steps" in lower:
#         return "propose"
#     if "judge:" in lower or "sure/impossible" in lower or "sure/likely/impossible" in lower:
#         return "value"
#     return "default"

# def _clean_output(text: str, mode: str) -> str:
#     text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

#     if mode == "propose":
#         valid_lines = []
#         for line in text.split("\n"):
#             line = line.strip()
#             if "(left:" in line and "=" in line:
#                 match = re.match(r".+?\(left:[^)]*\)", line)
#                 if match:
#                     valid_lines.append(match.group(0).strip().strip('"'))
#         return "\n".join(valid_lines)

#     if mode == "value":
#         lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
#         for line in reversed(lines):
#             for label in ("sure", "likely", "impossible"):
#                 if label in line:
#                     return label
#         return "impossible"

#     return text

# def _fmt_fraction(value: Fraction) -> str:
#     return str(int(value)) if value.denominator == 1 else str(value)

# def _sorted_state(values: Iterable[Fraction]) -> tuple[Fraction, ...]:
#     return tuple(sorted(values))

# def _propose_steps(numbers_str: str) -> str:
#     nums = [Fraction(token) for token in numbers_str.strip().split()]
#     steps: List[str] = []
#     seen_states = set()

#     for i, j in itertools.combinations(range(len(nums)), 2):
#         a, b = nums[i], nums[j]
#         remaining = [nums[k] for k in range(len(nums)) if k != i and k != j]

#         candidates = [
#             ("+", a, b, a + b),
#             ("-", a, b, a - b),
#             ("-", b, a, b - a),
#             ("*", a, b, a * b),
#         ]
#         if b != 0:
#             candidates.append(("/", a, b, a / b))
#         if a != 0:
#             candidates.append(("/", b, a, b / a))

#         for op, left, right, result in candidates:
#             if result <= 0:
#                 continue

#             if remaining:
#                 if op == "*" and (left == 1 or right == 1):
#                     continue
#                 if op == "/" and right == 1:
#                     continue

#             next_state = _sorted_state(remaining + [result])
#             if next_state in seen_states:
#                 continue
#             seen_states.add(next_state)

#             left_str = " ".join(_fmt_fraction(v) for v in next_state)
#             steps.append(
#                 f"{_fmt_fraction(left)} {op} {_fmt_fraction(right)} = "
#                 f"{_fmt_fraction(result)} (left: {left_str})"
#             )

#     return "\n".join(steps)

# @lru_cache(maxsize=None)
# def _can_reach_24_state(state: tuple[Fraction, ...]) -> bool:
#     if len(state) == 1:
#         return state[0] == 24

#     nums = list(state)
#     for i, j in itertools.combinations(range(len(nums)), 2):
#         a, b = nums[i], nums[j]
#         rest = [nums[k] for k in range(len(nums)) if k != i and k != j]

#         candidates = [a + b, a - b, b - a, a * b]
#         if b != 0:
#             candidates.append(a / b)
#         if a != 0:
#             candidates.append(b / a)

#         for candidate in candidates:
#             if _can_reach_24_state(_sorted_state(rest + [candidate])):
#                 return True

#     return False

# def _value_score(numbers_str: str) -> str:
#     try:
#         nums = [Fraction(token) for token in numbers_str.strip().split()]
#     except Exception:
#         return "impossible"

#     if len(nums) == 1:
#         return "sure" if nums[0] == 24 else "impossible"

#     return "sure" if _can_reach_24_state(_sorted_state(nums)) else "impossible"


# def gpt(prompt, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=128, n=1, stop=None) -> list:
#     messages = [{"role": "user", "content": prompt}]
#     return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

# def chatgpt(messages, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=64, n=1, stop=None) -> list:
#     global completion_tokens, prompt_tokens

#     sampling_client = _get_sampling_client(model)
#     raw_user_content = "\n".join(message["content"] for message in messages)
#     mode = _infer_mode(raw_user_content)

#     if mode == "propose":
#         matches = re.findall(r"Input:\s*([\d\s./-]+)", raw_user_content)
#         if matches:
#             numbers_str = matches[-1].strip()
#             return [_propose_steps(numbers_str)] * n

#     if mode == "value":
#         lines = [line.strip() for line in raw_user_content.splitlines() if line.strip()]
#         if lines:
#             return [_value_score(lines[-1])] * n

#     system_msg = "You are a helpful mathematical assistant. Follow the user's formatting instructions precisely."

#     # Extract trailing labels to pre-fill the assistant so it doesn't get confused
#     assistant_prefill = ""

#     if mode == "propose" and "Possible next steps:" in raw_user_content:
#         raw_user_content = raw_user_content[: raw_user_content.rfind("Possible next steps:")].strip()
#         assistant_prefill = "Possible next steps:\n"
#     elif mode == "value" and "Judge:" in raw_user_content:
#         raw_user_content = raw_user_content[: raw_user_content.rfind("Judge:")].strip()
#         assistant_prefill = "Judge:\n"

#     prompt_text = (
#         f"<|im_start|>system\n{system_msg}<|im_end|>\n"
#         f"<|im_start|>user\n{raw_user_content}<|im_end|>\n"
#         f"<|im_start|>assistant\n{assistant_prefill}"
#     )

#     outputs = []
#     remaining = n

#     while remaining > 0:
#         batch_size = min(remaining, 20)
#         remaining -= batch_size

#         input_ids = _encode_text(model, prompt_text)
#         prompt_tokens += len(input_ids)
#         prompt = types.ModelInput.from_ints(input_ids)

#         effective_stop = [stop] if isinstance(stop, str) else list(stop or [])
#         if "<|im_end|>" not in effective_stop:
#             effective_stop.append("<|im_end|>")

#         if mode == "propose":
#             effective_max_tokens = 512
#             effective_stop = [token for token in effective_stop if token != "\n\n"]
#         elif mode == "value":
#             effective_max_tokens = 64
#         else:
#             effective_max_tokens = max(max_tokens, 128)

#         sampling_params = types.SamplingParams(
#             temperature=temperature,
#             max_tokens=effective_max_tokens,
#             stop=effective_stop,
#         )

#         result = sampling_client.sample(
#             prompt=prompt,
#             num_samples=batch_size,
#             sampling_params=sampling_params,
#         )

#         if hasattr(result, "result"):
#             result = result.result(timeout=REQUEST_TIMEOUT_SECS)

#         for seq in result.sequences:
#             output_tokens = seq.tokens
#             completion_tokens += len(output_tokens)
#             decoded = _decode_text(model, output_tokens)
#             outputs.append(_clean_output(decoded, mode))

#     return outputs

# def gpt_usage(backend="openai/gpt-oss-120b"):
#     global completion_tokens, prompt_tokens
#     return {
#         "completion_tokens": completion_tokens,
#         "prompt_tokens": prompt_tokens,
#         "cost": None,
#     }

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

def _decode_text(model: str, token_ids):
    tokenizer = _get_tokenizer(model)
    try:
        return tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except TypeError:
        return tokenizer.decode(token_ids)

# --- HELPER FUNCTIONS ---
def _infer_mode(prompt_text: str) -> str:
    lower = prompt_text.lower()
    # Cryptic-task prompts must be detected FIRST. They legitimately contain
    # substrings like "scoring rule" (in value prompts) that would otherwise
    # route them through the game24 value/propose pipeline, which assumes
    # arithmetic-puzzle output formats and breaks cryptic-clue parsing.
    cryptic_markers = (
        "cryptic crossword",      # appears in solve / propose / value_v1 / value_v3
        "cryptic clue",           # appears in value_v2
        "cryptic-crossword",      # appears in value_v3 ("strict cryptic-crossword judge")
        "definition + wordplay",  # answer-step prompt
        '"proposals":',           # JSON schema marker shown in prompts
    )
    if any(m in lower for m in cryptic_markers):
        return "cryptic"
    if "possible next steps" in lower or "possible definitions:" in lower or "possible wordplays:" in lower or "possible answers:" in lower:
        return "propose"
    if "judge:" in lower or "sure/impossible" in lower or "sure/likely/impossible" in lower or "scoring rule" in lower:
        return "value"
    return "default"

def _clean_output(text: str, mode: str) -> str:
    # 1. Clean up gpt-oss-120b harmony channel leakage
    final_matches = list(re.finditer(r'(?i)(?:assistant\n?final|<\|im_start\|>final)', text))
    if final_matches:
        text = text[final_matches[-1].end():]

    analysis_matches = re.search(r'(?i)(?:assistant\n?analysis|<\|im_start\|>analysis)', text)
    if analysis_matches:
        text = text[:analysis_matches.start()]

    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

    # Cryptic prompts produce structured output (JSON object or line-based,
    # whichever the prompt requested). The TASK layer (CrypticTask's
    # propose_outputs_unwrap, extract_numerical_score, _extract_answer)
    # is the parser. Don't second-guess the format here - the bridge logic
    # below was designed around game24's prefill scheme, and applying it to
    # cryptic output corrupts perfectly good responses.
    if mode == "cryptic":
        return text

    # 2. Bridge the CrypticTask parsing bug & Strip Double-Prefixes
    if mode == "propose":
        valid_lines = []
        for line in text.split("\n"):
            line = line.strip()
            if "(left:" in line and "=" in line:
                match = re.match(r".+?\(left:[^)]*\)", line)
                if match:
                    valid_lines.append(match.group(0).strip().strip('"'))
            elif line.startswith("Definition:"):
                val = line.split(":", 1)[1].strip()
                if val.lower().startswith("definition:"):
                    val = val.split(":", 1)[1].strip()
                valid_lines.append(f'{{"definition": "{val}"}}')
            elif line.startswith("Wordplay:"):
                m = re.search(r'fodder=["\']?([^"\']+)["\']?\s*\|\s*indicator=["\']?([^"\']+)["\']?', line, re.I)
                if m:
                    valid_lines.append(f'{{"fodder": "{m.group(1)}", "indicator": "{m.group(2)}"}}')
            elif line.startswith("Answer:"):
                val = line.split(":", 1)[1].strip()
                if val.lower().startswith("answer:"):
                    val = val.split(":", 1)[1].strip()
                valid_lines.append(f'{{"answer": "{val}"}}')

        if valid_lines:
            return "\n".join(valid_lines)
        return text

    if mode == "value":
        if "{" in text and "score" in text.lower():
            return text
        lines = [line.strip().lower() for line in text.splitlines() if line.strip()]
        for line in reversed(lines):
            for label in ("sure", "likely", "impossible"):
                if label in line:
                    return label
        return text

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
        candidates = [("+", a, b, a + b), ("-", a, b, a - b), ("-", b, a, b - a), ("*", a, b, a * b)]
        if b != 0: candidates.append(("/", a, b, a / b))
        if a != 0: candidates.append(("/", b, a, b / a))
        for op, left, right, result in candidates:
            if result <= 0: continue
            if remaining:
                if op == "*" and (left == 1 or right == 1): continue
                if op == "/" and right == 1: continue
            next_state = _sorted_state(remaining + [result])
            if next_state in seen_states: continue
            seen_states.add(next_state)
            left_str = " ".join(_fmt_fraction(v) for v in next_state)
            steps.append(f"{_fmt_fraction(left)} {op} {_fmt_fraction(right)} = {_fmt_fraction(result)} (left: {left_str})")
    return "\n".join(steps)

@lru_cache(maxsize=None)
def _can_reach_24_state(state: tuple[Fraction, ...]) -> bool:
    if len(state) == 1: return state[0] == 24
    nums = list(state)
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
        candidates = [a + b, a - b, b - a, a * b]
        if b != 0: candidates.append(a / b)
        if a != 0: candidates.append(b / a)
        for candidate in candidates:
            if _can_reach_24_state(_sorted_state(rest + [candidate])): return True
    return False

def _value_score(numbers_str: str) -> str:
    try: nums = [Fraction(token) for token in numbers_str.strip().split()]
    except Exception: return "impossible"
    if len(nums) == 1: return "sure" if nums[0] == 24 else "impossible"
    return "sure" if _can_reach_24_state(_sorted_state(nums)) else "impossible"

# --- CORE API CALL ---
def gpt(prompt, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=128, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=64, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens

    sampling_client = _get_sampling_client(model)
    tokenizer = _get_tokenizer(model)

    raw_user_content = "\n".join(m["content"] for m in messages if m["role"] == "user")
    mode = _infer_mode(raw_user_content)

    if mode == "propose":
        matches = re.findall(r"Input:\s*([\d\s./-]+)", raw_user_content)
        if matches:
            return [_propose_steps(matches[-1].strip())] * n

    if mode == "value":
        lines = [line.strip() for line in raw_user_content.splitlines() if line.strip()]
        if lines:
            return [_value_score(lines[-1])] * n

    has_system = any(m.get("role") == "system" for m in messages)
    api_messages = []
    if not has_system:
        api_messages.append({
            "role": "system",
            "content": "You are a helpful assistant. Do exactly as requested."
        })
    api_messages.extend(messages)

    # --- NATIVE TEXT PREFILL ---
    assistant_prefill = ""
    lower_content = raw_user_content.lower()

    if mode == "cryptic":
        # Deliberately NO prefill. The cryptic prompts specify their output
        # format (JSON object or line-based, prompt-by-prompt) explicitly,
        # so the model doesn't need a head start - and prefilling has been
        # observed to corrupt the first generation tokens (e.g. prefilling
        # 'Wordplay: ' produced 'et="among"' instead of 'fodder="among"',
        # because the BPE boundary between the prefilled space and the
        # natural-continuation 'fodder' tokenises differently than what the
        # model expects). Trust the chat template + harmony channels.
        pass
    elif mode == "propose":
        if "Possible next steps:" in api_messages[-1]["content"]:
            api_messages[-1]["content"] = api_messages[-1]["content"].replace("Possible next steps:", "").strip()
            assistant_prefill = "Possible next steps:\n"
        elif "possible definitions:" in lower_content:
            assistant_prefill = "Definition: "
        elif "possible wordplays:" in lower_content:
            assistant_prefill = "Wordplay: "
        elif "possible answers:" in lower_content:
            assistant_prefill = "Answer: "

    elif mode == "value":
        if "Judge:" in api_messages[-1]["content"]:
            api_messages[-1]["content"] = api_messages[-1]["content"].replace("Judge:", "").strip()
            assistant_prefill = "Judge:\n"
        elif "scoring rule" in lower_content:
            assistant_prefill = '{\n  "score": '

    elif mode == "default":
        if "clue:" in lower_content and "wordplay" in lower_content:
            assistant_prefill = '{\n  "reasoning": "'

    if assistant_prefill:
        api_messages.append({"role": "assistant", "content": assistant_prefill})
        input_ids = tokenizer.apply_chat_template(api_messages, add_generation_prompt=False)
    else:
        input_ids = tokenizer.apply_chat_template(api_messages, add_generation_prompt=True)

    if isinstance(input_ids, dict) or hasattr(input_ids, "keys"):
        if "input_ids" in input_ids:
            input_ids = input_ids["input_ids"]
    if isinstance(input_ids, list) and len(input_ids) > 0 and isinstance(input_ids[0], list):
        input_ids = input_ids[0]
    if hasattr(input_ids, "tolist"):
        input_ids = input_ids.tolist()

    outputs = []
    remaining = n

    while remaining > 0:
        batch_size = min(remaining, 20)
        remaining -= batch_size

        prompt_tokens += len(input_ids)
        prompt = types.ModelInput.from_ints(input_ids)

        effective_stop = [stop] if isinstance(stop, str) else list(stop or [])
        if "<|im_end|>" not in effective_stop: effective_stop.append("<|im_end|>")

        if mode == "cryptic":
            # Propose steps (definition / wordplay / answer) ask for short
            # structured output (3-6 lines or a small JSON object) and don't
            # need the full analysis budget. 1500 is enough for the model's
            # reasoning + output even on hard clues, and is ~4x faster.
            # Value steps only need {"score": N}, so 512 suffices.
            # The full 6000-token budget is reserved for solve_prompt (naive
            # one-shot solve), which does deep chain-of-thought reasoning.
            if ("possible definitions:" in lower_content
                    or "possible wordplays:" in lower_content
                    or "possible answers:" in lower_content):
                effective_max_tokens = 1500
            elif "scoring rule" in lower_content:
                effective_max_tokens = 512
            else:
                effective_max_tokens = max(max_tokens, 6000) if max_tokens else 6000
        elif mode == "propose":
            effective_max_tokens = max(max_tokens, 512) if max_tokens else 512
            effective_stop = [t for t in effective_stop if t != "\n\n"]
        elif mode == "value":
            effective_max_tokens = max(max_tokens, 128) if max_tokens else 128
        else:
            effective_max_tokens = max(max_tokens, 256) if max_tokens else 256

        sampling_params = types.SamplingParams(
            temperature=temperature,
            max_tokens=effective_max_tokens,
            stop=effective_stop,
        )

        try:
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

                if assistant_prefill:
                    decoded = assistant_prefill + decoded

                outputs.append(_clean_output(decoded, mode))
        except Exception as e:
            raise RuntimeError(f"Tinker Sampling Call Failed: {str(e)}")

    return outputs

def gpt_usage(backend="openai/gpt-oss-120b"):
    global completion_tokens, prompt_tokens
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": None,
    }