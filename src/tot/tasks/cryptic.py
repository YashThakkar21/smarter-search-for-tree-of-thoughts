import re
import json
import os
from typing import Optional, Iterator
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.cryptic import (
    solve_prompt,
    propose_definition_prompt,
    propose_wordplay_prompt,
    propose_answer_prompt,
    value_prompts_ensemble,
)


# ============================================================================
# Helpers
# ============================================================================

# Trailing letter-count enumeration: (7), (3, 4), (2, 1, 4) etc.
# We look at the very end so we don't catch parenthetical asides mid-clue.
_ENUM_RE = re.compile(r'\(\s*([\d]+(?:\s*,\s*[\d]+)*)\s*\)\s*$')


def _parse_enumeration(clue: str) -> Optional[list]:
    """Pull `[7]` from `... (7)` or `[2, 1, 4]` from `... (2, 1, 4)`."""
    m = _ENUM_RE.search(clue.strip())
    if not m:
        return None
    try:
        return [int(x.strip()) for x in m.group(1).split(',')]
    except ValueError:
        return None


def _normalize_answer(s) -> str:
    """Uppercase, strip, collapse internal whitespace. Used for matching."""
    if s is None:
        return ""
    return re.sub(r'\s+', ' ', str(s).strip().upper())


# Patterns that indicate a CONFIDENT answer assertion. We look for these
# in analysis text when the model truncated before reaching the final
# channel. Order matters - earlier patterns are higher confidence.
# Each pattern's group(1) is the candidate answer string.
#
# NOTE on the candidate sub-pattern:
#   (?-i:...) forces case-sensitive matching INSIDE the group even when
#   the surrounding pattern uses (?i). Without this localisation, the
#   global (?i) makes [A-Z] match lowercase too, and the candidate gobbles
#   trailing context like "MINUS FOUR which fits the (5,4)" instead of
#   stopping at "MINUS FOUR". The inner pattern captures one or more runs
#   of 2+ uppercase letters, separated by single space-or-hyphen, up to
#   6 words total.
_CAND = r'(?-i:([A-Z]{2,}(?:[ \-][A-Z]{2,}){0,5}))'
_ANSWER_ASSERTION_PATTERNS = [
    # "Final answer: ROMANCE" / "final answer ROMANCE"
    re.compile(rf'(?i)final\s+answer\s*[:\-]?\s*"?{_CAND}"?'),
    # "Final: ROMANCE" / "final - ROMANCE"
    re.compile(rf'(?i)\bfinal\s*[:\-]\s*"?{_CAND}"?'),
    # "Therefore [...] the answer is ROMANCE" / "so the answer must be ROMANCE"
    re.compile(rf'(?i)(?:therefore|thus|so|hence)[^.]{{0,40}}\banswer\s+(?:is|must\s+be|=)\s*"?{_CAND}"?'),
    # "the answer is ROMANCE" / "the answer must be ROMANCE"
    re.compile(rf'(?i)\banswer\s+(?:is|must\s+be|=)\s*"?{_CAND}"?'),
    # "Answer: ROMANCE" anywhere
    re.compile(rf'(?i)\banswer\s*[:\-]\s*"?{_CAND}"?'),
    # "= ROMANCE" or "-> ROMANCE" at the end of a line (verdict)
    re.compile(rf'(?:=|->)\s*"?{_CAND}"?\s*\.?$', re.MULTILINE),
]


def _salvage_answer_from_analysis(clue: str, raw: str) -> Optional[str]:
    """
    Conservative last-ditch extraction of an answer from truncated analysis
    text. Returns None unless we find a candidate that:
      1. Matches one of the high-confidence assertion patterns above.
      2. Matches the clue's letter-count enumeration exactly.
      3. Appears in the LAST 30% of the analysis (the tail, where the
         model's final commitment lives - rejected candidates earlier
         in the stream don't qualify).

    This deliberately won't rescue tentative phrases like "Maybe X?" or
    "Could be X" - those are exploratory, not commitments. False
    negatives are acceptable; false positives (rescuing a rejected
    candidate as if it were the answer) would silently inflate accuracy.
    """
    if not raw or len(raw) < 200:
        return None

    enum = _parse_enumeration(clue)
    if enum is None:
        return None
    expected_total = sum(enum)
    expected_words = len(enum)

    # Restrict to the tail of the analysis - rejections happen earlier.
    tail_start = int(len(raw) * 0.7)
    tail = raw[tail_start:]

    for pattern in _ANSWER_ASSERTION_PATTERNS:
        for match in reversed(list(pattern.finditer(tail))):
            candidate = _normalize_answer(match.group(1))
            if not candidate:
                continue
            words = candidate.split(' ')
            word_lens = [len(w) for w in words if w]
            # Must match enumeration exactly: same number of words, same lengths.
            if len(words) != expected_words:
                continue
            if word_lens != enum:
                continue
            # Reject candidates that are obvious echoes from the prompt
            # (the clue itself, or words from the clue).
            clue_words = set(_normalize_answer(clue).split())
            if all(w in clue_words for w in words):
                continue
            return candidate

    return None


# ============================================================================
# Robust JSON extraction
#
# gpt-oss-120b emits "harmony"-format outputs that look like
#     analysis ... assistantfinal {"reasoning": "...", "answer": "..."}
# so a naive json.loads(output) fails. We try, in order:
#   1. Clean parse after stripping ```json fences
#   2. The LAST balanced {...} block in the output (handles nested braces
#      and quoted-brace edge cases via a manual scanner)
#   3. Regex fallback for "field": "value"
#
# This mirrors the strategy in tasks/game24.py:extract_numerical_score,
# but generalised to any string field.
# ============================================================================


def _balanced_json_blocks(text: str) -> Iterator[str]:
    """
    Yield substrings of `text` that are syntactically balanced {...} blocks,
    respecting JSON string literals and escapes. Yields outermost objects only.
    """
    depth = 0
    start = -1
    in_str = False
    escape = False
    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if in_str:
            if ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            if depth > 0:
                depth -= 1
                if depth == 0 and start >= 0:
                    yield text[start:i + 1]
                    start = -1


def _try_json_dict(blob: str) -> Optional[dict]:
    try:
        data = json.loads(blob)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    return None


def extract_json_field(output: str, field: str) -> Optional[str]:
    """
    Pull a string field out of a JSON-formatted model output.

    Order of attempts:
      1. Clean json.loads after stripping ```json``` fences.
      2. Walk every balanced {...} block in the output, last-first.
         (gpt-oss harmony format puts the real JSON at the END.)
      3. Regex over the whole output: `"field": "value"`.

    Returns None on total failure so callers can decide what to do
    (retry with more max_tokens, fall back, log, etc.) instead of being
    handed a silent None-as-empty-string.
    """
    if not output:
        return None

    cleaned = re.sub(r'```json|```', '', output).strip()
    data = _try_json_dict(cleaned)
    if data and field in data and data[field] is not None:
        return str(data[field])

    candidates = list(_balanced_json_blocks(output))
    for candidate in reversed(candidates):
        data = _try_json_dict(candidate)
        if data and field in data and data[field] is not None:
            return str(data[field])

    pattern = rf'"{re.escape(field)}"\s*:\s*"((?:[^"\\]|\\.)*)"'
    m = re.search(pattern, output, flags=re.DOTALL)
    if m:
        try:
            return bytes(m.group(1), 'utf-8').decode('unicode_escape', errors='replace')
        except Exception:
            return m.group(1)

    return None


def extract_numerical_score(output: str) -> Optional[float]:
    """Same robust strategy, specialised for `"score": <int 1-10>`."""
    if not output:
        return None

    cleaned = re.sub(r'```json|```', '', output).strip()
    data = _try_json_dict(cleaned)
    if data and "score" in data:
        try:
            score = float(data["score"])
            if 1 <= score <= 10:
                return score
        except (ValueError, TypeError):
            pass

    for candidate in reversed(list(_balanced_json_blocks(output))):
        data = _try_json_dict(candidate)
        if data and "score" in data:
            try:
                score = float(data["score"])
                if 1 <= score <= 10:
                    return score
            except (ValueError, TypeError):
                continue

    m = re.search(r'"score"\s*:\s*"?(\d+)"?\b', output)
    if m:
        try:
            score = float(m.group(1))
            if 1 <= score <= 10:
                return score
        except ValueError:
            pass

    return None


# ============================================================================
# Task
# ============================================================================

class CrypticTask(Task):
    """
    Cryptic crossword task using the Minute Cryptic annotated dataset.

    Input (x)   : a cryptic clue, e.g. "Love initially absent from prom dance (7)"
    Output (y)  : either
                    (a) a 3-line decomposition produced by ToT search:
                          Definition: Love
                          Wordplay: fodder="prom dance" | indicator="initially | absent from"
                          Answer: ROMANCE
                    (b) the JSON blob produced by solve_prompt for naive_solve.
                  test_output handles both transparently.
    Reward (r)  : 1 if the extracted answer matches the gold answer (case- and
                  whitespace-insensitive; we additionally accept matches that
                  ignore inter-word spacing, since models occasionally squash
                  multi-word answers into "AMILIVE").

    The annotated columns we consume from the xlsx are:
        Clue, Definition, Fodder, Indicator, Answer
    Other columns (Pun?, Is_Anagram, etc.) are ignored.
    """

    DEFAULT_FILE = 'minute_cryptic_complete_annotations.xlsx'

    def __init__(self, file: str = DEFAULT_FILE):
        """
        `file` may be either a bare filename (looked up under
        DATA_PATH/cryptic/) or an absolute path.
        """
        super().__init__()
        if os.path.isabs(file):
            path = file
        else:
            path = os.path.join(DATA_PATH, 'cryptic', file)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Cryptic data file not found at {path}. "
                f"Place {self.DEFAULT_FILE} under tot/data/cryptic/, "
                f"or pass an absolute path to CrypticTask(file=...)."
            )

        df = pd.read_excel(path)
        keep = ['Clue', 'Definition', 'Fodder', 'Indicator', 'Answer']
        missing = [c for c in keep if c not in df.columns]
        if missing:
            raise KeyError(
                f"Cryptic data file missing columns: {missing}. "
                f"Found: {list(df.columns)}"
            )
        df = df.dropna(subset=keep).reset_index(drop=True)
        self.data = df[keep].copy()

        self.value_cache = {}
        # Three steps: definition -> wordplay -> answer.
        self.steps = 3
        # Stops are per-step. Propose prompts emit MULTIPLE candidates
        # (one per line) so we must NOT stop at '\n' - we let the model
        # finish its list. Naive solve uses stop=None separately in bfs.py.
        self.stops = [None, None, None]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return str(self.data.iloc[idx]['Clue'])

    def get_gold(self, idx: int) -> dict:
        """Annotated decomposition - useful for oracle baselines & error analysis."""
        row = self.data.iloc[idx]
        return {
            'clue':       str(row['Clue']),
            'definition': str(row['Definition']),
            'fodder':     str(row['Fodder']),
            'indicator':  str(row['Indicator']),
            'answer':     str(row['Answer']),
            'enumeration': _parse_enumeration(str(row['Clue'])),
        }

    # ------------------------------------------------------------------------
    # Output testing - the heart of "test our model's performance" without
    # parsing errors getting in the way.
    # ------------------------------------------------------------------------

    def _extract_answer(self, output: str) -> Optional[str]:
        """
        Pull a candidate answer out of `output`. Tries, in order:
          1. JSON `"answer"` field via the harmony-aware extractor.
          2. The LAST `Answer: X` line in the output (case-insensitive).
        Returns None if neither yields anything.
        """
        ans = extract_json_field(output, 'answer')
        if ans:
            return ans

        matches = re.findall(r'(?im)^\s*Answer\s*:\s*(.+?)\s*$', output)
        if matches:
            return matches[-1]
        return None

    def test_output(self, idx: int, output: str) -> dict:
        """
        Score `output` against the gold answer for puzzle `idx`.

        Returns a dict with:
          r:           1 if match, 0 otherwise
          extracted:   the normalized extracted answer (or None)
          gold:        the normalized gold answer
          parse_failed (optional): True if we couldn't extract any answer at all
                       - distinguishes "model got it wrong" from "we couldn't
                       even parse what the model said". This is the metric you
                       want to drive to zero so performance reflects solving
                       ability, not formatting noise.
          enum_match (optional): whether the answer's word lengths match the
                       clue's enumeration. Diagnostic only - not used to gate
                       correctness, since gold itself is the ground truth.
          space_normalized (optional): True if we accepted a match by ignoring
                       inter-word spaces.
        """
        gold = _normalize_answer(self.data.iloc[idx]['Answer'])
        extracted = self._extract_answer(output)
        if extracted is None:
            return {
                'r': 0,
                'extracted': None,
                'gold': gold,
                'parse_failed': True,
            }

        norm = _normalize_answer(extracted)
        clue = str(self.data.iloc[idx]['Clue'])
        enum = _parse_enumeration(clue)
        enum_match = None
        if enum is not None:
            word_lens = [len(w) for w in norm.split(' ') if w]
            enum_match = (word_lens == enum)

        if norm == gold:
            return {'r': 1, 'extracted': norm, 'gold': gold, 'enum_match': enum_match}

        # Lenient: ignore inter-word spaces (model returns "AMILIVE" for "AM I LIVE")
        if norm.replace(' ', '') == gold.replace(' ', ''):
            return {
                'r': 1, 'extracted': norm, 'gold': gold,
                'enum_match': enum_match, 'space_normalized': True,
            }

        return {'r': 0, 'extracted': norm, 'gold': gold, 'enum_match': enum_match}

    # ------------------------------------------------------------------------
    # Naive (one-shot) prompt wraps used by bfs.naive_solve
    # ------------------------------------------------------------------------

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = '') -> str:
        # The "standard" and "cot" prompts are the same here: the JSON solve
        # prompt already requests structured chain-of-thought via the
        # "reasoning" field, and we want robust parsing in both modes.
        return solve_prompt.format(clue=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = '') -> str:
        return solve_prompt.format(clue=x) + y

    # ------------------------------------------------------------------------
    # ToT step prompts (propose) - line-based output for bfs.py compatibility
    # ------------------------------------------------------------------------

    @staticmethod
    def _y_lines(y: str):
        return [line for line in y.strip().split('\n') if line.strip()]

    @staticmethod
    def _strip_known_prefix(line: str, prefix: str) -> str:
        """Strip e.g. 'Definition: ' from a line; otherwise return line stripped."""
        line = line.strip()
        if line.lower().startswith(prefix.lower()):
            return line.split(':', 1)[1].strip()
        return line

    @classmethod
    def propose_prompt_wrap(cls, x: str, y: str = '') -> str:
        """
        Step is determined by line count of `y`:
          0 lines -> propose definition
          1 line  -> propose wordplay (uses line 1 as the chosen definition)
          2 lines -> propose answer   (uses lines 1 & 2 as def + wordplay)
        Lines that don't carry the expected prefix are passed through stripped,
        which means a malformed earlier step still produces a workable prompt
        downstream rather than crashing.
        """
        lines = cls._y_lines(y)
        n = len(lines)

        if n == 0:
            return propose_definition_prompt.format(clue=x)
        if n == 1:
            return propose_wordplay_prompt.format(
                clue=x,
                definition=cls._strip_known_prefix(lines[0], 'Definition:'),
            )
        # n >= 2: time to propose the answer.
        return propose_answer_prompt.format(
            clue=x,
            definition=cls._strip_known_prefix(lines[0], 'Definition:'),
            wordplay=cls._strip_known_prefix(lines[1], 'Wordplay:'),
        )

    @classmethod
    def propose_outputs_unwrap(cls, x: str, y: str, raw_output: str) -> list:
        """
        Filter a raw LLM output into well-formed proposal lines for the
        current decomposition step.

        bfs.py's get_proposals naively splits raw output on newlines and
        treats every non-empty line as a proposal. That's a disaster with
        gpt-oss-120b's harmony format, which leaks analysis-channel text
        ("...assistantanalysisWe need to parse...") into the visible output.
        Every preamble line ends up being treated as a search candidate.

        This method finds a JSON object in the output (last balanced {...}
        block, harmony-aware) and converts the structured proposals into
        canonical one-per-line strings:
          Step 1 -> 'Definition: <text>'
          Step 2 -> 'Wordplay: fodder="<text>" | indicator="<text>"'
          Step 3 -> 'Answer: <UPPERCASE>'

        If JSON parsing fails entirely, falls back to per-field regex over
        the whole output - so even a malformed/truncated response can yield
        usable proposals as long as some "definition": "..." substring
        survived.

        Returns [] if nothing recognisable can be salvaged. bfs.py treats
        an empty proposal list as the end of the search, which is the right
        behaviour: better to halt than to descend into a tree of preamble.

        Requires a small bfs.py patch (see docstring of CrypticTask) to be
        called by the BFS solver. Falls back gracefully if uncalled.
        """
        n_existing = len(cls._y_lines(y))

        # -- Strategy 1: find a JSON object somewhere in the output --
        data = None
        cleaned = re.sub(r'```json|```', '', raw_output).strip()
        candidate = _try_json_dict(cleaned)
        if candidate is not None and isinstance(candidate.get('proposals'), list):
            data = candidate
        else:
            for blob in reversed(list(_balanced_json_blocks(raw_output))):
                candidate = _try_json_dict(blob)
                if candidate is not None and isinstance(candidate.get('proposals'), list):
                    data = candidate
                    break

        proposals_list = data['proposals'] if data else []
        results = []

        if n_existing == 0:
            # Step 1: definitions
            for p in proposals_list:
                if isinstance(p, dict) and p.get('definition'):
                    txt = str(p['definition']).strip()
                    if txt:
                        results.append(f"Definition: {txt}")
            if not results:
                # Regex fallback: pluck "definition": "..." from anywhere
                for m in re.finditer(
                    r'"definition"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output,
                ):
                    txt = m.group(1).strip()
                    if txt:
                        results.append(f"Definition: {txt}")
                # Last-ditch: legacy "Definition: X" lines (one per line).
                # Useful if a non-cryptic-aware models.py prefilled the
                # response with "Definition: " and produced line-based output.
                if not results:
                    for m in re.finditer(
                        r'(?im)^\s*Definition\s*:\s*(.+?)\s*$', raw_output,
                    ):
                        txt = m.group(1).strip()
                        if txt and txt.lower() != 'definition':
                            results.append(f"Definition: {txt}")

        elif n_existing == 1:
            # Step 2: wordplays (need both fodder AND indicator)
            for p in proposals_list:
                if isinstance(p, dict) and p.get('fodder') and p.get('indicator'):
                    f = str(p['fodder']).strip()
                    i = str(p['indicator']).strip()
                    results.append(f'Wordplay: fodder="{f}" | indicator="{i}"')
            if not results:
                # Pair fodder/indicator regex matches in document order
                fodders = [m.group(1).strip() for m in re.finditer(
                    r'"fodder"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output,
                )]
                indicators = [m.group(1).strip() for m in re.finditer(
                    r'"indicator"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output,
                )]
                for f, i in zip(fodders, indicators):
                    if f and i:
                        results.append(f'Wordplay: fodder="{f}" | indicator="{i}"')
                # Last-ditch: legacy line `Wordplay: fodder="X" | indicator="Y"`,
                # tolerant to a corrupted prefill that may have eaten the
                # leading 'fodder' token (e.g. 'Wordplay: et="among" | ...'
                # from the prefill-tokenisation pathology). We accept any
                # `key="value" | indicator="value"` pattern and assume the
                # first quoted span is the fodder regardless of the key name.
                #
                # The values are matched with [^"]+ (not [^"']+) because
                # real clues contain apostrophes: indicator="Fig 'n'" must
                # match as a single value, not break at the inner '.
                if not results:
                    for m in re.finditer(
                        r'(?im)^\s*Wordplay\s*:\s*\w*=?"([^"]+)"\s*\|\s*'
                        r'indicator\s*=\s*"([^"]+)"\s*$',
                        raw_output,
                    ):
                        f, i = m.group(1).strip(), m.group(2).strip()
                        if f and i:
                            results.append(f'Wordplay: fodder="{f}" | indicator="{i}"')

        else:
            # Step 3: answers (normalise to uppercase + single spaces)
            for p in proposals_list:
                if isinstance(p, dict) and p.get('answer'):
                    norm = _normalize_answer(p['answer'])
                    if norm:
                        results.append(f"Answer: {norm}")
            if not results:
                # Regex fallback: "answer": "..." anywhere
                for m in re.finditer(
                    r'"answer"\s*:\s*"((?:[^"\\]|\\.)*)"', raw_output,
                ):
                    norm = _normalize_answer(m.group(1))
                    if norm:
                        results.append(f"Answer: {norm}")
                # Last-ditch: legacy "Answer: X" line format
                if not results:
                    for m in re.finditer(
                        r'(?im)^\s*Answer\s*:\s*(.+?)\s*$', raw_output,
                    ):
                        norm = _normalize_answer(m.group(1))
                        if norm:
                            results.append(f"Answer: {norm}")

        # Dedupe while preserving order.
        seen = set()
        deduped = []
        for r in results:
            if r not in seen:
                seen.add(r)
                deduped.append(r)

        if not deduped:
            # Salvage path for the ANSWER step (n_existing == 2) when the
            # model truncated mid-analysis without firing the final channel.
            # This rescues real solving ability from token-budget failures
            # WITHOUT inventing answers from nothing - we only accept a
            # candidate if (a) it's preceded by a confident assertion phrase
            # and (b) it matches the clue's enumeration. Rejected candidates
            # in analysis ("Maybe X? That's the wrong length.") are mid-line
            # and don't pass these filters.
            if n_existing == 2:
                salvaged = _salvage_answer_from_analysis(x, raw_output)
                if salvaged:
                    deduped.append(f"Answer: {salvaged}")
                    return deduped

            # Make the silent-failure case loud. Without this, bfs.py just
            # breaks out of its loop with an empty new_ys and returns ['']
            # to the caller, who prints nothing - the run looks like it
            # succeeded with no answer instead of failing.
            import sys
            raw_stripped = (raw_output or '').strip()
            preview = raw_stripped[:400]
            tail = raw_stripped[-200:] if len(raw_stripped) > 600 else ''
            step_num = n_existing + 1
            step_name = ('definition', 'wordplay', 'answer')[min(n_existing, 2)]
            sys.stderr.write(
                f"\n[CrypticTask] step {step_num}/3 ({step_name}): 0 proposals "
                f"extracted (raw output length = {len(raw_output or '')}).\n"
            )
            if not raw_stripped:
                sys.stderr.write(
                    "  Output is empty. Check that bfs.gpt(...) is actually returning\n"
                    "  generations and that max_tokens is reaching the API call.\n"
                )
            else:
                sys.stderr.write(f"  raw head: {preview!r}\n")
                if tail:
                    sys.stderr.write(f"  raw tail: {tail!r}\n")
                # Diagnose by content. Two distinct failure modes:
                #   A) "analysis"-like preamble with NO 'final' marker
                #      anywhere -> harmony format misconfigured (chat
                #      template not applied).
                #   B) 'analysis' present AND substantive (>500 chars) but
                #      no JSON / proper final output -> the model ran out
                #      of tokens mid-thought. Common on hard clues.
                in_analysis = 'analysis' in raw_stripped[:50].lower()
                has_final = 'assistantfinal' in raw_stripped.lower() or '<|channel|>final' in raw_stripped.lower()
                long_enough = len(raw_stripped) > 1500
                if in_analysis and not has_final and long_enough:
                    sys.stderr.write(
                        "  DIAGNOSIS: Model ran out of tokens in the analysis channel\n"
                        "  before reaching final. Bump max_tokens further (try 6000),\n"
                        "  and/or lower temperature (e.g. 0.3-0.5) so the model\n"
                        "  doesn't wander through as many candidate answers before\n"
                        "  converging. Cryptic reasoning is exploratory by nature,\n"
                        "  but high temperature compounds the wandering.\n"
                    )
                elif in_analysis and not has_final and not long_enough:
                    sys.stderr.write(
                        "  DIAGNOSIS: Model emitted analysis text but no final\n"
                        "  channel. If your tot/models.py is patched for 'cryptic'\n"
                        "  mode, this is unusual - check that the chat template\n"
                        "  is being applied (apply_chat_template) and that no\n"
                        "  prefill is being injected for cryptic prompts.\n"
                    )
                else:
                    sys.stderr.write(
                        "  DIAGNOSIS: Output present but didn't match any expected\n"
                        "  cryptic format (JSON {{\"proposals\": [...]}} or legacy\n"
                        "  line-based). Inspect the raw head/tail above.\n"
                    )
            sys.stderr.flush()

        return deduped

    # ------------------------------------------------------------------------
    # Value evaluation
    #
    # Two interfaces, matching the two solver code paths in your codebase:
    #   * BFS path (bfs.get_value):    calls value_prompt_wrap, then
    #                                  value_outputs_unwrap to aggregate.
    #   * MCTS path (_get_ensemble_value in mcts.py): calls
    #                                  get_ensemble_prompts directly and
    #                                  averages parsed scores.
    # ------------------------------------------------------------------------

    def get_ensemble_prompts(self, x: str, y: str) -> list:
        """List of prompts to be evaluated and averaged by the MCTS path."""
        partial = y.strip() if y.strip() else "(empty - clue only, no decomposition yet)"
        return [p.format(clue=x, partial=partial) for p in value_prompts_ensemble]

    @staticmethod
    def extract_numerical_score(output: str) -> Optional[float]:
        return extract_numerical_score(output)

    def value_prompt_wrap(self, x: str, y: str) -> str:
        """Single-prompt value used by bfs.get_value. Returns the v1 prompt."""
        partial = y.strip() if y.strip() else "(empty - clue only, no decomposition yet)"
        return value_prompts_ensemble[0].format(clue=x, partial=partial)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """
        Average parsed JSON scores. Failed parses are dropped (NOT treated as
        zero, which would silently penalise harmless format wobbles). If
        every output fails to parse, fall back to the midpoint 5.0 - same
        philosophy as test_evaluator.py for game24.
        """
        scores = [extract_numerical_score(o) for o in value_outputs]
        valid = [s for s in scores if s is not None]
        if not valid:
            return 5.0
        return sum(valid) / len(valid)

    # ------------------------------------------------------------------------
    # finalize_output: emit a clean trailing `Answer: NORM` line so downstream
    # printers / loggers see the canonical answer regardless of whether y came
    # from the JSON solve path or the line-based propose path.
    # ------------------------------------------------------------------------

    def finalize_output(self, x: str, y: str) -> str:
        ans = self._extract_answer(y)
        if ans is None:
            return y
        norm = _normalize_answer(ans)
        non_empty = [line for line in y.split('\n') if line.strip()]
        if non_empty and non_empty[-1].strip().lower() == f"answer: {norm}".lower():
            return y
        return y.rstrip() + f"\nAnswer: {norm}\n"