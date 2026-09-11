import re
import unicodedata
from typing import List, Tuple

from eval_metrics import normalize_bengali_text, tokenize, token_jaccard

_VISIBLE_GROUNDING_TERMS = {
    "লোক", "মানুষ", "ছেলে", "মেয়ে", "নারী", "পুরুষ", "শিশু", "বাচ্চা",
    "কুকুর", "বিড়াল", "বিড়াল", "গরু", "পাখি", "ঘোড়া", "ঘোড়া",
    "গাছ", "ফুল", "নদী", "পানি", "রাস্তা", "মাঠ", "ঘর", "বাড়ি", "বাড়ি",
    "গাড়ি", "গাড়ি", "বাস", "ট্রেন", "সাইকেল", "নৌকা", "আকাশ",
    "চেয়ার", "টেবিল", "বই", "ক্যামেরা", "বল", "খাবার", "প্লেট",
}

_ACTION_TERMS = {
    "আছে", "আছেন", "করছে", "করছেন", "বসে", "দাঁড়িয়ে", "দাঁড়িয়ে",
    "হাঁটছে", "হাটছে", "খাচ্ছে", "ধরেছে", "ধরে", "পরে", "পরেছে",
    "দেখা", "খেলছে", "চলছে", "রয়েছে", "রয়েছে",
}

_ATTRIBUTE_TERMS = {
    "লাল", "নীল", "সবুজ", "কালো", "সাদা", "হলুদ", "ধূসর", "ধুসর",
    "বড়", "বড়", "ছোট", "দুই", "দুইটি", "তিন", "তিনটি", "একটি",
    "উপর", "নিচে", "পাশে", "সামনে", "পেছনে", "মাথায়", "হাতে",
}

_ABSTRACT_TERMS = {
    "আশা", "আলো", "জীবন", "ছন্দ", "সৌন্দর্য", "দুর্দশা", "অনুভূতি",
    "স্বপ্ন", "ভালোবাসা", "প্রকৃতির", "মিশ্রতা",
}


def normalize_caption_for_selection(caption: str) -> str:
    """Normalize caption text for deterministic filtering and duplicate checks."""
    return normalize_bengali_text(caption)


def caption_tokens(caption: str) -> List[str]:
    """Tokenize caption after Bengali normalization."""
    return tokenize(normalize_bengali_text(caption))


def is_repetitive_caption(tokens: List[str], repetition_threshold: float = 0.66) -> bool:
    """Check if token list exhibits excessive repetitive degeneration."""
    if not tokens:
        return True
    unique_ratio = len(set(tokens)) / len(tokens)
    return unique_ratio < (1.0 - repetition_threshold)


def caption_quality_score(caption: str) -> float:
    """Compute heuristic quality score for a Bengali caption."""
    tokens = caption_tokens(caption)
    token_count = len(tokens)
    if token_count == 0:
        return -100.0

    score = 0.0
    token_set = set(tokens)

    if token_count < 3:
        score -= 4.0
    elif 5 <= token_count <= 14:
        score += 2.0
    elif token_count <= 20:
        score += 1.0
    else:
        score -= 2.0

    if is_repetitive_caption(tokens):
        score -= 4.0

    bengali_chars = sum(1 for ch in caption if "\u0980" <= ch <= "\u09ff")
    letter_chars = sum(1 for ch in caption if ch.isalpha())
    if letter_chars and bengali_chars / letter_chars >= 0.8:
        score += 2.0
    else:
        score -= 2.0

    visible_hits = len(token_set & _VISIBLE_GROUNDING_TERMS)
    action_hits = len(token_set & _ACTION_TERMS)
    attribute_hits = len(token_set & _ATTRIBUTE_TERMS)
    abstract_hits = len(token_set & _ABSTRACT_TERMS)

    score += min(visible_hits, 3) * 2.0
    score += min(action_hits, 2) * 1.5
    score += min(attribute_hits, 3) * 1.0

    if visible_hits == 0 and action_hits == 0:
        score -= 3.0
    if abstract_hits and visible_hits == 0:
        score -= abstract_hits * 2.0

    return score


def select_top_captions_for_image(captions: List[str], max_captions: int = 2) -> List[str]:
    """
    Select a deterministic quality-first, diverse subset of captions for one image.

    The policy keeps visually grounded, readable Bengali captions first, then chooses
    a second caption that adds information rather than duplicating the first.
    """
    if max_captions <= 0:
        return []

    normalized: List[Tuple[str, List[str], float, int]] = []
    fallback: List[Tuple[str, List[str], float, int]] = []
    seen = set()
    for index, caption in enumerate(captions or []):
        norm_caption = normalize_caption_for_selection(caption)
        if not norm_caption:
            continue
        tokens = caption_tokens(norm_caption)
        duplicate_key = " ".join(tokens).casefold()
        if duplicate_key in seen:
            continue
        seen.add(duplicate_key)

        score = caption_quality_score(norm_caption)
        item = (norm_caption, tokens, score, index)
        fallback.append(item)
        if score <= -6.0:
            continue
        normalized.append(item)

    if not normalized:
        normalized = fallback

    if len(normalized) <= max_captions:
        return [caption for caption, _tokens, _score, _index in normalized]

    normalized.sort(key=lambda item: (-item[2], item[3], item[0]))
    selected = [normalized[0]]

    while len(selected) < max_captions and len(selected) < len(normalized):
        best_candidate = None
        best_rank = None
        for candidate in normalized:
            if candidate in selected:
                continue
            max_similarity = max(
                token_jaccard(candidate[1], selected_item[1])
                for selected_item in selected
            )
            diversity_bonus = 1.0 - max_similarity
            rank = (candidate[2] + diversity_bonus * 2.0, -max_similarity, -candidate[3])
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_candidate = candidate
        selected.append(best_candidate)

    return [caption for caption, _tokens, _score, _index in selected]
