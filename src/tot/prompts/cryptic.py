# Cryptic crossword prompts.
#
# Design notes:
#   * All scoring/answer prompts return JSON. The extractor in tasks/cryptic.py
#     is built to handle gpt-oss-120b's harmony format ("analysis...assistantfinal
#     {json}"), markdown fences, and bare regex fallbacks - mirrors the pattern
#     already used in tasks/game24.py:extract_numerical_score.
#   * Propose prompts use one-candidate-per-line plain text so they remain
#     compatible with bfs.py's line-based proposal extraction (which just
#     splits the raw output on '\n').
#   * Few-shot examples are deliberately simple, well-known cryptic clues so
#     they don't risk leaking from the Minute Cryptic test set.

# ============================================================================
# Few-shot examples (used by the direct-solve prompt)
# ============================================================================

_FEW_SHOT_FULL = '''\
EXAMPLE 1
Clue: Love initially absent from prom dance (7)
Reasoning: "Love" is the definition (synonym for romance). "initially" picks the first letter of "prom" = "p"; "absent from prom dance" deletes that "p" from the phrase "prom dance", leaving the letters R-O-M-A-N-C-E. Concatenate -> ROMANCE.
Output:
{{
  "reasoning": "Love = definition. Wordplay: remove initial letter of 'prom' (= p) from 'prom dance' -> 'rom ance' -> ROMANCE.",
  "definition": "Love",
  "fodder": "prom dance",
  "indicator": "initially | absent from",
  "answer": "ROMANCE"
}}

EXAMPLE 2
Clue: Harp on about unfortunate issue (6)
Reasoning: "unfortunate issue" is the definition (an orphan is the issue/child of misfortune). "about" is a classic anagram indicator. Anagram of HARP ON -> ORPHAN.
Output:
{{
  "reasoning": "unfortunate issue = definition. 'about' indicates anagram of 'Harp on' -> ORPHAN.",
  "definition": "unfortunate issue",
  "fodder": "Harp on",
  "indicator": "about",
  "answer": "ORPHAN"
}}

EXAMPLE 3
Clue: Influencer's content goes viral? (3)
Reasoning: "goes viral?" is a tongue-in-cheek definition (the flu goes viral). "content" indicates the inner letters. Middle letters of "infLUencer" -> FLU.
Output:
{{
  "reasoning": "goes viral? = cryptic definition. 'content' = middle letters; middle of 'Influencer' is FLU.",
  "definition": "goes viral?",
  "fodder": "Influencer",
  "indicator": "content",
  "answer": "FLU"
}}

EXAMPLE 4
Clue: Throw some symbolic boomerangs (3)
Reasoning: "Throw" is the definition. "some" indicates a hidden word; "boomerangs" indicates reversal. Hidden in "symBOLic" -> "BOL", reversed -> LOB.
Output:
{{
  "reasoning": "Throw = definition. 'some' = hidden, 'boomerangs' = reversal. 'symBOLic' contains BOL; reversed = LOB.",
  "definition": "Throw",
  "fodder": "symbolic",
  "indicator": "some | boomerangs",
  "answer": "LOB"
}}

EXAMPLE 5
Clue: Christmas now? (7, 3)
Reasoning: Cryptic definition: "Christmas" is Christmas Day, and "now?" hints at "the present day". Two senses of "present" (gift / current) collide.
Output:
{{
  "reasoning": "Cryptic definition. 'Christmas' suggests presents/gifts; 'now?' = current day. PRESENT DAY = both Christmas and 'today'.",
  "definition": "Christmas",
  "fodder": "now",
  "indicator": "?",
  "answer": "PRESENT DAY"
}}
'''

# ============================================================================
# Direct solve prompt (one-shot CoT, JSON output).
# Used for: naive_solve baseline, and final-answer extraction in test_output.
# ============================================================================

_JSON_FORMAT_FULL = '''You MUST respond using this EXACT JSON format and nothing else:
{{
  "reasoning": "step-by-step work: identify the definition, the wordplay device (anagram / hidden / reversal / container / deletion / selection / homophone / charade / cryptic-definition), the fodder, and how the device produces the answer letters",
  "definition": "the substring of the clue that is the definition",
  "fodder": "the raw material the wordplay operates on",
  "indicator": "the words that signal the wordplay device(s)",
  "answer": "THEANSWER"
}}
The "answer" field must be UPPERCASE letters with single spaces between words for multi-word answers, and have exactly the letter count given by the enumeration in parentheses at the end of the clue.'''

solve_prompt = f'''You are an expert cryptic crossword solver. A cryptic clue has two interlocking parts:
- A DEFINITION: a synonym or short description of the answer, almost always a contiguous substring at the start or end of the clue.
- WORDPLAY: the rest of the clue. The wordplay encodes the answer letters through one or more devices: anagram, hidden word, reversal, container (one set of letters inside another), deletion, selection (initial / final / middle letters), homophone, abbreviation, charade (concatenation), or a pure cryptic definition.

The number(s) in parentheses at the end give the answer's letter count - per word for multi-word answers.

Your job: solve the clue, and explain how.

{_FEW_SHOT_FULL}

{_JSON_FORMAT_FULL}

Clue: {{clue}}
Output:'''


# ============================================================================
# Step-wise PROPOSE prompts (line-based output for bfs.py compatibility)
# ============================================================================

# --- Step 1: propose definition candidates ---------------------------------
propose_definition_prompt = '''You are an expert cryptic crossword solver. The first step in solving a clue is identifying the DEFINITION: a contiguous substring of the clue (almost always at the very start or very end) that is a synonym for the answer.

Examples:

Clue: Love initially absent from prom dance (7)
Possible definitions:
Definition: Love
Definition: prom dance

Clue: Harp on about unfortunate issue (6)
Possible definitions:
Definition: Harp on
Definition: unfortunate issue

Clue: Influencer's content goes viral? (3)
Possible definitions:
Definition: Influencer
Definition: goes viral?

Now propose 3-4 candidate definitions for the clue below. Output ONE PER LINE in the exact format `Definition: <substring>`. Each substring must be a contiguous span copied from the clue. No commentary, no numbering, no JSON, no blank lines.

Clue: {clue}
Possible definitions:
'''


# --- Step 2: propose wordplay (fodder + indicator) given a definition ------
propose_wordplay_prompt = '''You are an expert cryptic crossword solver. Given a clue and a chosen DEFINITION, the WORDPLAY is everything else. It has two parts:
- FODDER: the raw material the wordplay operates on (the letters / words that get manipulated).
- INDICATOR: the word(s) that signal the wordplay device. Common indicators: "about / mixed up / confused" -> anagram; "some / part of / contained in" -> hidden word; "back / returning / rising" -> reversal; "in / inside / around / hugging" -> container; "without / dropping / losing" -> deletion; "first / initially / heads of" -> selection of letters; "we hear / sounds like" -> homophone.

Examples:

Clue: Love initially absent from prom dance (7)
Definition: Love
Possible wordplays:
Wordplay: fodder="prom dance" | indicator="initially | absent from"

Clue: Harp on about unfortunate issue (6)
Definition: unfortunate issue
Possible wordplays:
Wordplay: fodder="Harp on" | indicator="about"

Clue: Influencer's content goes viral? (3)
Definition: goes viral?
Possible wordplays:
Wordplay: fodder="Influencer" | indicator="content"

Now propose 2-3 plausible wordplays for the clue below given the definition. Output ONE PER LINE in the exact format `Wordplay: fodder="<text>" | indicator="<text>"`. No commentary, no numbering, no JSON, no blank lines.

Clue: {clue}
Definition: {definition}
Possible wordplays:
'''


# --- Step 3: propose answer candidates given full decomposition ------------
propose_answer_prompt = '''You are an expert cryptic crossword solver. Given a clue and a decomposition into definition + wordplay, derive candidate answers. The answer must:
1. Match the enumeration (the letter counts in parentheses at the end of the clue).
2. Be a synonym for the definition.
3. Be reachable from the fodder by applying the indicated wordplay device.

Examples:

Clue: Love initially absent from prom dance (7)
Definition: Love
Wordplay: fodder="prom dance" | indicator="initially | absent from"
Possible answers:
Answer: ROMANCE

Clue: Harp on about unfortunate issue (6)
Definition: unfortunate issue
Wordplay: fodder="Harp on" | indicator="about"
Possible answers:
Answer: ORPHAN

Now propose 1-3 candidate answers for the clue below. Output ONE PER LINE in the exact format `Answer: WORD` (UPPERCASE letters, single spaces between words for multi-word answers). No commentary, no numbering, no JSON, no blank lines.

Clue: {clue}
Definition: {definition}
Wordplay: {wordplay}
Possible answers:
'''


# ============================================================================
# Value prompts (ensemble, JSON output, robust to harmony-format models)
# Used to score an intermediate state (clue + partial decomposition).
# Mirrors game24.py's value_prompts_ensemble pattern.
# ============================================================================

_SCORING_RULE = '''SCORING RULE (this is about the QUALITY of the partial solution, not your confidence in your judgement):
  1-2  = clearly wrong (definition isn't a synonym for any plausible answer; wordplay can't produce a real word; answer doesn't match the enumeration; answer isn't a synonym for the definition)
  3-4  = unlikely; significant flaws but not impossible
  5-6  = genuinely uncertain; some signal but unverified
  7-8  = looks promising; pieces line up but you can't fully verify
  9-10 = very likely correct (definition fits a real word; wordplay machinery actually produces the answer letters from the fodder; answer matches the enumeration)
Use the full 1-10 range. Do not default to only 1 or 10.'''

_JSON_FORMAT_SCORE = '''You MUST respond using this EXACT JSON format and nothing else:
{{
  "reasoning": "your brief assessment of how well the pieces fit",
  "score": <integer from 1 to 10>
}}'''

value_prompt_v1 = f'''You are evaluating a partial or full solution to a cryptic crossword clue. Judge how plausible the proposed decomposition is given standard cryptic conventions: the definition must be a synonym for the eventual answer; the wordplay must produce the answer letters via a recognized device (anagram / hidden / reversal / container / deletion / selection / homophone / charade / cryptic-def); the answer must match the enumeration.

{_SCORING_RULE}

{_JSON_FORMAT_SCORE}

Clue: {{clue}}
Partial solution so far:
{{partial}}
Output:'''

value_prompt_v2 = f'''Estimate how likely the partial solution below is on the path to (or already is) the correct answer to the cryptic clue. A complete cryptic must satisfy three constraints simultaneously: definition synonymous with answer, wordplay derivation sound, answer matches enumeration. Penalize solutions that satisfy only one or two.

{_SCORING_RULE}

{_JSON_FORMAT_SCORE}

Clue: {{clue}}
Partial solution so far:
{{partial}}
Output:'''

value_prompt_v3 = f'''Act as a strict cryptic-crossword judge. Look at the clue and the partial solution and ask: does the definition really mean the answer? Does the wordplay machinery actually produce those exact letters from the fodder? Does the letter count match the enumeration? Score accordingly.

{_SCORING_RULE}

{_JSON_FORMAT_SCORE}

Clue: {{clue}}
Partial solution so far:
{{partial}}
Output:'''

value_prompts_ensemble = [value_prompt_v1, value_prompt_v2, value_prompt_v3]