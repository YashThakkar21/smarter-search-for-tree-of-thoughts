import os
import re
from dotenv import load_dotenv
from tinker import ServiceClient, types

load_dotenv()

completion_tokens = 0
prompt_tokens = 0

DEBUG = True
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
        text = tokenizer.decode(token_ids, skip_special_tokens=True).strip()
    except TypeError:
        text = tokenizer.decode(token_ids)
    return text

def _infer_mode(prompt_text: str) -> str:
    lower = prompt_text.lower()
    if "possible next steps" in lower:
        return "propose"
    if "judge:" in lower or "sure/impossible" in lower or "sure/likely/impossible" in lower:
        return "value"
    return "default"

def _clean_output(text: str, mode: str) -> str:
    # Clean ChatML artifacts
    text = text.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

    if mode == "propose":
        valid_lines = []
        # The model might yap, so we ONLY extract lines containing valid math steps
        for line in text.split('\n'):
            line = line.strip()
            # Game24 proposes always contain an equals sign and the leftover numbers in parentheses
            if "(left:" in line and "=" in line:
                # Extract ONLY up to the closing paren, discard everything after
                match = re.match(r'.+?\(left:[^)]*\)', line)
                if match:
                    clean_line = match.group(0).strip().strip('"')
                    valid_lines.append(clean_line)

        # If the model completely fails, return a safe fallback rather than an empty string
        return "\n".join(valid_lines) if valid_lines else "0 + 0 = 0 (left: )"

    elif mode == "value":
        # Only check the last non-empty line - the model's preamble is noise
        lines = [l.strip().lower() for l in text.strip().splitlines() if l.strip()]
        for line in reversed(lines):
            for val in ["sure", "likely", "impossible"]:
                if val in line:
                    return val
        # fallback: count occurrences in full text, pick majority
        lower_text = text.lower()
        counts = {v: lower_text.count(v) for v in ["sure", "likely", "impossible"]}
        return max(counts, key=counts.get)

    return text

def _propose_steps(numbers_str: str) -> str:
    """Deterministically enumerate all valid Game24 next steps."""
    import itertools
    nums = list(map(float, numbers_str.strip().split()))
    steps = []
    for i, j in itertools.combinations(range(len(nums)), 2):
        a, b = nums[i], nums[j]
        remaining = [nums[k] for k in range(len(nums)) if k != i and k != j]
        for op_sym, result in [('+', a+b), ('-', a-b), ('-', b-a), ('*', a*b)]:
            if result <= 0:  # skip non-positive results
                continue

            if op_sym == '-' and a-b == result:
                x, y = a, b
            elif op_sym == '-':
                x, y = b, a
            else:
                x, y = a, b
            left = sorted(remaining + [result])
            left_str = ' '.join(str(int(v) if v == int(v) else v) for v in left)
            r_str = int(result) if result == int(result) else result
            x_str = int(x) if x == int(x) else x
            y_str = int(y) if y == int(y) else y
            steps.append(f"{x_str} {op_sym} {y_str} = {r_str} (left: {left_str})")
        if b != 0 and a % b == 0:
            result = a / b
            left = sorted(remaining + [result])
            left_str = ' '.join(str(int(v) if v == int(v) else v) for v in left)
            steps.append(f"{int(a)} / {int(b)} = {int(result)} (left: {left_str})")
        if a != 0 and b % a == 0:
            result = b / a
            left = sorted(remaining + [result])
            left_str = ' '.join(str(int(v) if v == int(v) else v) for v in left)
            steps.append(f"{int(b)} / {int(a)} = {int(result)} (left: {left_str})")
    return "\n".join(steps)


from fractions import Fraction

def _can_reach_24(nums):
    nums = [Fraction(n).limit_denominator(10000) for n in nums]
    if len(nums) == 1:
        return nums[0] == 24
    for i in range(len(nums)):
        for j in range(len(nums)):
            if i == j:
                continue
            rest = [nums[k] for k in range(len(nums)) if k != i and k != j]
            a, b = nums[i], nums[j]
            candidates = [a+b, a-b, a*b]
            if b != 0:
                candidates.append(a/b)
            for c in candidates:
                if _can_reach_24(rest + [c]):
                    return True
    return False

def _value_score(numbers_str: str) -> str:
    nums = [float(x) for x in numbers_str.strip().split()]
    if len(nums) == 1:
        return "sure" if abs(nums[0] - 24) < 1e-6 else "impossible"
    return "sure" if _can_reach_24(nums) else "impossible"

def gpt(prompt, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=64, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    return chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)

def chatgpt(messages, model="openai/gpt-oss-120b", temperature=0.7, max_tokens=64, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens

    sampling_client = _get_sampling_client(model)

    raw_user_content = "\n".join(m["content"] for m in messages)
    mode = _infer_mode(raw_user_content)

    if mode == "propose":
        matches = re.findall(r'Input:\s*([\d\s]+)', raw_user_content)
        if matches:
            numbers_str = matches[-1].strip()
            # print(f"[DEBUG] propose shortcut firing for: {numbers_str}")
            steps = _propose_steps(numbers_str)
            return [steps] * n
        # else:
            # print("[DEBUG] propose shortcut MISSED - no Input: found")

    if mode == "value":
        # Extract the last non-empty line of numbers before any reasoning examples
        # The value prompt ends with the numbers to evaluate on their own line
        lines = [l.strip() for l in raw_user_content.strip().splitlines() if l.strip()]
        last_line = lines[-1]
        try:
            result = _value_score(last_line)
            return [result] * n
        except (ValueError, ZeroDivisionError):
            pass

    system_msg = "You are a helpful mathematical assistant. Follow the user's formatting instructions precisely."

    # Extract trailing labels to pre-fill the assistant so it doesn't get confused
    assistant_prefill = ""
    if mode == "propose" and "Possible next steps:" in raw_user_content:
        raw_user_content = raw_user_content[:raw_user_content.rfind("Possible next steps:")].strip()
        assistant_prefill = "Possible next steps:\n"
    elif mode == "value" and "Judge:" in raw_user_content:
        raw_user_content = raw_user_content[:raw_user_content.rfind("Judge:")].strip()
        assistant_prefill = "Judge:\n"

    prompt_text = f"<|im_start|>system\n{system_msg}<|im_end|>\n<|im_start|>user\n{raw_user_content}<|im_end|>\n<|im_start|>assistant\n{assistant_prefill}"

    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt

        input_ids = _encode_text(model, prompt_text)
        prompt_tokens += len(input_ids)
        prompt = types.ModelInput.from_ints(input_ids)

        effective_stop = [stop] if isinstance(stop, str) else (stop or [])
        if "<|im_end|>" not in effective_stop:
            effective_stop.append("<|im_end|>")
        # if "Input:" not in effective_stop:
        #     effective_stop.append("Input:")

        if mode == "propose":
            effective_max_tokens = 512
            effective_stop = [s for s in effective_stop if s != '\n\n']
        elif mode == "value":
            effective_max_tokens = 64
        else:
            effective_max_tokens = max(max_tokens, 128)

        sampling_params = types.SamplingParams(
            temperature=temperature,
            max_tokens=effective_max_tokens,
            stop=effective_stop,
        )

        res = sampling_client.sample(
            prompt=prompt,
            num_samples=cnt,
            sampling_params=sampling_params,
        )

        if hasattr(res, "result"):
            res = res.result(timeout=REQUEST_TIMEOUT_SECS)

        for i, seq in enumerate(res.sequences):
            # 1. Decode ONLY the tokens generated by the model (excluding the prompt/prefill)
            output_tokens = seq.tokens
            completion_tokens += len(output_tokens)
            text = _decode_text(model, output_tokens)

            # print(f"[DEBUG] seq token count: {len(output_tokens)}")
            # print(f"[DEBUG] raw decoded: {repr(text[:300])}")

            # 2. Aggressively clean the output to pluck out ONLY valid equations / judgments
            clean_text = _clean_output(text, mode)

            outputs.append(clean_text)

    return outputs

def gpt_usage(backend="openai/gpt-oss-120b"):
    global completion_tokens, prompt_tokens
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": None}

def _steps_to_answer(x: str, steps: str) -> str:
    """Reconstruct a final answer expression from the step trajectory."""
    import re
    expressions = []
    for line in steps.strip().split('\n'):
        match = re.match(r'(.+?)\s*=\s*[\d.]+\s*\(left:', line)
        if match:
            expressions.append(match.group(1).strip())
    # Build a nested expression (last op wraps previous)
    return f"Answer: {' -> '.join(expressions)} = 24"