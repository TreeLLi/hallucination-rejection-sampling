"""Utility functions."""
import os
import logging
import pickle
from pathlib import Path
import hashlib

import wandb
import numpy as np

def wandb_restore(wandb_run, filename):
    files_dir = 'tmp_wandb/'
    os.system(f'rm -rf {files_dir}')
    os.system(f'mkdir -p {files_dir}')

    run = api.run(wandb_run)
    run.file(filename).download(
        root=files_dir, replace=True, exist_ok=False)
    with open(f'{files_dir}/{filename}', 'rb') as f:
        out = pickle.load(f)
    return out, run.config

def setup_wandb(proj_id, args):
    wandb.login()
    run = wandb.init(project=proj_id, config=args)
    return run

def setup_logger(log_path):
    """Setup logger to always print time and level."""

    # Create parent directory if it doesn't exist
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        filename=log_path,
        filemode='w+',
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.getLogger().setLevel(logging.INFO)  # logging.DEBUG


def log_w_indent(text, indent=4, symbol='>>'):
    """Log and add indent."""
    ind2col = {i: f"\x1b[{a}" for i, a in enumerate([
        '1m', '31m', '33m', '34m', '35m'])}
    reset = "\x1b[0m"

    if indent > 0:
        logging.info(ind2col[indent] + (indent * 2) * symbol + ' ' + text + reset)
    else:
        logging.info(ind2col[indent] + text + reset)


def get_sentences(response):
    """Extract sentences from response."""
    # Some manual exceptions for sentence extraction.
    facts = response.replace('Ph.D.', 'PhD').replace('\n', ' ').split('. ')
    facts = [f.strip() + '.' if i < len(facts) - 1 else f.strip() for i, f in enumerate(facts)]
    for i in facts:
        print(i)


def cluster_assignment_entropy(semantic_ids):
    """Estimate semantic uncertainty from how often different clusters get assigned.

    We estimate the categorical distribution over cluster assignments from the
    semantic ids. The uncertainty is then given by the entropy of that
    distribution. This estimate does not use token likelihoods, it relies soley
    on the cluster assignments. If probability mass is spread of between many
    clusters, entropy is larger. If probability mass is concentrated on a few
    clusters, entropy is small.

    Input:
        semantic_ids: List of semantic ids, e.g. [0, 1, 2, 1].
    Output:
        cluster_entropy: Entropy, e.g. (-p log p).sum() for p = [1/4, 2/4, 1/4].
    """

    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy


def extract_questions(gen_questions):

    compatibility = (
        os.environ['HALLU_RESTORE_ID'] in ['hallu_long/5yfel47n', 'hallu_long/rok13nf2'] and
        'gen_qs' in os.environ['HALLU_RESTORE_STAGES'])

    questions = []
    for i, q in enumerate(gen_questions.split('\n')):
        if q.startswith(f'{i + 1}. '):
            questions.append(q[3:])
        else:
            if not compatibility:
                questions.append(q)
            else:
                questions.append(q[3:])

    return questions


def get_yes_no(response):
    binary_response = response.lower()[:10]
    if 'yes' in binary_response:
        uncertainty = 0
    elif 'no' in binary_response:
        uncertainty = 1
    else:
        uncertainty = 1
        logging.warning('MANUAL NO!')
    return uncertainty



import re, nltk
from typing import Iterable, List, Optional, Tuple

# Ensure punkt is available
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
# Some NLTK versions also require punkt_tab
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    try:
        nltk.download("punkt_tab", quiet=True)
    except Exception:
        pass

tok = nltk.data.load("tokenizers/punkt/english.pickle")

def split_sentences(
    text: str,
    lang: str = "en",
    extra_abbrevs: Optional[Iterable[str]] = None
) -> List[str]:
    """
    Exact-substring sentence splitter using NLTK Punkt with robust boundary fixes
    and explicit newline splitting.

    Behaviors
    ---------
    - Only user-provided `extra_abbrevs` are added to Punkt (no hardcoded defaults).
    - Splits ARE allowed after the end of compact multi-dot abbreviations (e.g., 'U.S.').
    - Splits are NOT allowed inside:
        * compact multi-dot abbreviations (e.g., 'U.|S.' or 'Ph.|D.'),
        * spaced-initial chains (e.g., 'J.| K.' or 'J. K.| Rowling'),
        * the cross-line variant of the above (e.g., 'U.\\nS.' or 'J.\\nK.').
    - Newlines are treated as additional boundaries:
        * A run of line breaks (\\n or \\r\\n) splits segments.
        * The entire run is attached to the LEFT/preceding sentence.
        * Newline boundaries obey the same “don’t split inside abbrev/name” rules.
    - Quote-aware realignment: closing quotes/brackets following sentence-ending
      punctuation are kept with the preceding sentence.

    Returns
    -------
    List[str]
        Exact slices from `text` (no normalization). Pure-whitespace slices
        resulting from leading/trailing blank lines are omitted.
    """

    if not text:
        return []

    # Add ONLY user-supplied abbreviations to Punkt's abbrev table
    if extra_abbrevs:
        cleaned = set()
        for a in extra_abbrevs:
            a = a.strip().lower().rstrip(".")
            if a:
                cleaned.add(a)
        try:
            tok._params.abbrev_types.update(cleaned)
        except Exception:
            pass

    # --- Initial spans from Punkt ---
    spans = list(tok.span_tokenize(text))
    if not spans:
        return []

    # ---------- Helpers for boundary checks ----------
    # 1) Compact multi-dot abbr mid-split: 'U.|S.' or 'Ph.|D.'
    def is_mid_multidot_compact(t: str, idx: int) -> bool:
        if idx <= 0 or idx + 1 >= len(t):
            return False
        return (t[idx - 1] == '.') and t[idx].isalpha() and (idx + 1 < len(t) and t[idx + 1] == '.')

    # 2) Between spaced initials: 'J.| K.'  (allow any whitespace in between)
    _L_BETWEEN_INITIAL = re.compile(r'[A-Z]\.\s*$', re.ASCII)
    _R_BETWEEN_INITIAL = re.compile(r'^\s+[A-Z]\.', re.ASCII)
    def is_between_spaced_initials(t: str, idx: int) -> bool:
        left_win  = t[max(0, idx - 20): idx]
        right_win = t[idx: idx + 20]
        return bool(_L_BETWEEN_INITIAL.search(left_win) and _R_BETWEEN_INITIAL.search(right_win))

    # 3) After last spaced initial, before a capitalized surname: 'J. K.| Rowling'
    _L_HAS_CHAIN_OF_INITIALS = re.compile(r'(?:[A-Z]\.\s+){1,}[A-Z]\.\s*$', re.ASCII)
    _R_SURNAME               = re.compile(r'^\s+([A-Z][a-z]{2,})\b')
    _DISALLOW_AS_SURNAME = {
        "The","This","That","These","Those","There",
        "However","Moreover","Furthermore","But","And","Or","Nor","So","Yet",
        "Because","While","When","Where","Whether","Although","Though",
        "Today","Tomorrow","Yesterday",
    }
    def is_after_last_initial_before_surname(t: str, idx: int) -> bool:
        if idx <= 0 or t[idx - 1] != '.':
            return False
        left_win  = t[max(0, idx - 40): idx]
        right_win = t[idx: idx + 40]
        if not _L_HAS_CHAIN_OF_INITIALS.search(left_win):
            return False
        m = _R_SURNAME.match(right_win)
        return bool(m and m.group(1) not in _DISALLOW_AS_SURNAME)

    # 4) Cross-line compact multi-dot: e.g., 'U.\\nS.' or 'Ph.\\nD.'
    _LINEBREAK_RUN = re.compile(r'(?:\r\n|\n)+')
    _LEFT_DOT_THEN_LB = re.compile(r'[A-Za-z]\.(?:\r\n|\n)+$')
    def is_mid_multidot_across_linebreak(t: str, idx: int) -> bool:
        # boundary `idx` should be placed *after* a linebreak run for this to trigger
        left_win  = t[max(0, idx - 40): idx]
        right_win = t[idx: idx + 20]
        return bool(_LEFT_DOT_THEN_LB.search(left_win) and re.match(r'^[A-Za-z]\.', right_win))

    def boundary_is_inside_name_or_abbrev(t: str, idx: int) -> bool:
        return (
            is_mid_multidot_compact(t, idx) or
            is_between_spaced_initials(t, idx) or
            is_after_last_initial_before_surname(t, idx) or
            is_mid_multidot_across_linebreak(t, idx)
        )

    # ---------- Pass 1: merge Punkt boundaries that fall inside abbrev/name patterns ----------
    segs: List[List[int]] = []
    for start, end in spans:
        if not segs:
            segs.append([start, end])
            continue
        prev_start, prev_end = segs[-1]
        if boundary_is_inside_name_or_abbrev(text, prev_end):
            segs[-1][1] = end
        else:
            segs.append([start, end])

    # ---------- Pass 2: split each segment on runs of line breaks ----------
    # We attach the entire run of line breaks to the LEFT segment.
    split_segs: List[List[int]] = []
    for s, e in segs:
        cur = s
        for m in _LINEBREAK_RUN.finditer(text, s, e):
            cut = m.end()  # boundary AFTER the whole run of line breaks
            # Skip if this boundary would fall inside a protected pattern
            if cur < cut < e and not boundary_is_inside_name_or_abbrev(text, cut):
                split_segs.append([cur, cut])
                cur = cut
        if cur < e:
            split_segs.append([cur, e])

    segs = split_segs

    # ---------- Pass 3: quote-aware realignment (American style punctuation inside quotes) ----------
    CLOSERS = set('"\')]}»›”’')
    SENT_END = set('.?!')
    for i in range(len(segs) - 1):
        end_i = segs[i][1]
        if end_i <= 0 or end_i > len(text):
            continue
        if text[end_i - 1] in SENT_END:
            j = end_i
            while j < len(text) and text[j] in CLOSERS:
                j += 1
            if j != end_i:
                segs[i][1] = j
                if segs[i + 1][0] < j:
                    segs[i + 1][0] = j

    # Build exact substrings; drop whitespace-only slices (e.g., leading blank lines)
    out = []
    for s, e in segs:
        piece = text[s:e]
        if piece.strip():  # keep only non-empty after stripping whitespace
            out.append(piece)
    return out


def find_substring_ignore_case(A: str, B: str) -> str | None:
    """
    Returns the substring from B that matches A (ignoring case).
    If not found, returns None.
    """
    A_lower, B_lower = A.lower(), B.lower()
    idx = B_lower.find(A_lower)
    if idx == -1:
        return None
    return B[idx:idx + len(A)]


def is_non_answer(response: str) -> bool:
    """
    Detect if an LLM-generated response does not actually answer the question,
    but instead gives a disclaimer or refusal (e.g., "there is no widely known person named XX").
    
    Returns True if the response is a non-answer, False otherwise.
    """
    if not response or not isinstance(response, str):
        return True
    
    response = response.strip().lower()

    # Common patterns for non-answers / disclaimers
    non_answer_patterns = [
        "no widely known",
        "not widely known",
        "not widely available",
        "not publicly available",
        "not a well-known",
        "not well-known",
        "not known",
        "not publicly",
        "no prominent",
        "no notable",
        "no known",
        "no famously known",
        "no widely known",        
        "i could not find",
        "couldn't find",
        "does not exist",
        "not a real person",
        "i'm not aware",
        "i do not know",
        "not available",
        "no information available",
        "cannot guarantee",
        "recommend checking",
        "may want to check",
        "please check",
        "consider checking",
        "little information",
        "common name",
        "can refer to",        
        "several individuals",
        "multiple individuals",
        "few individuals",
        "not a widely recognized",
        "further clarification may help",
        "i do not have information",
        "i don't have information",
        "i have no information",
        "i lack information",
        "i'm not aware",
        "i do not know",
        "i'm not sure",
        "i cannot confirm",
        "i cannot guarantee",
        "i cannot verify",
        "may not be accurate",
        "information may be incomplete",
        "information may not be reliable",
        "details are limited"
    ]

    for pat in non_answer_patterns:
        if pat in response:
            print(f"Detected non answer pattern: {pat}")
            return True
    
    return False

