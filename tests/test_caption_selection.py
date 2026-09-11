import os
import sys
import unicodedata

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "code")))

from caption_selection.selector import (
    caption_quality_score,
    caption_tokens,
    is_repetitive_caption,
    normalize_caption_for_selection,
    select_top_captions_for_image,
)


def normalize_for_assertion(caption):
    return unicodedata.normalize("NFC", caption)


def test_normalize_caption_for_selection_preserves_bengali_vowel_marks():
    caption = "  আজকে আবহাওয়া খুব ভালো।  "

    normalized = normalize_caption_for_selection(caption)

    assert normalized == "আজকে আবহাওয়া খুব ভালো"
    assert "।" not in normalized


def test_caption_tokens_remove_punctuation_without_dropping_bengali_marks():
    tokens = caption_tokens("লোকটির হাতে একটি বই রয়েছে।")

    assert tokens == ["লোকটির", "হাতে", "একটি", "বই", "রয়েছে"]


def test_caption_quality_prefers_grounded_caption_over_abstract_caption():
    grounded = "একটি লোক লাল টুপি পরে বসে আছে"
    abstract = "জীবনের ছন্দ"

    assert caption_quality_score(grounded) > caption_quality_score(abstract)


def test_is_repetitive_caption_detects_degenerate_repetition():
    assert is_repetitive_caption(["লোক", "লোক", "লোক", "লোক", "লোক"])
    assert not is_repetitive_caption(["একটি", "লোক", "লাল", "টুপি", "পরে", "বসে", "আছে"])


def test_select_top_captions_returns_empty_for_empty_or_non_positive_limit():
    assert select_top_captions_for_image([]) == []
    assert select_top_captions_for_image(["একটি লোক বসে আছে"], max_captions=0) == []


def test_select_top_captions_keeps_existing_two_valid_captions():
    captions = [
        "একটি লোক লাল টুপি পরে বসে আছে",
        "লোকটির হাতে একটি বই রয়েছে",
    ]

    selected = select_top_captions_for_image(captions)

    assert selected == [normalize_for_assertion(caption) for caption in captions]


def test_select_top_captions_removes_duplicates_and_selects_two():
    duplicate = "একটি লোক লাল টুপি পরে বসে আছে"
    captions = [
        duplicate,
        duplicate,
        "একটি লোক লাল টুপি পরে বসে আছে।",
        "লোকটির হাতে একটি বই রয়েছে",
    ]

    selected = select_top_captions_for_image(captions)

    assert len(selected) == 2
    assert selected[0] == normalize_for_assertion(duplicate)
    assert normalize_for_assertion("লোকটির হাতে একটি বই রয়েছে") in selected


def test_select_top_captions_prefers_diverse_second_caption():
    captions = [
        "একটি লোক লাল টুপি পরে বসে আছে",
        "একটি লোক লাল টুপি পরে আছে",
        "লোকটির হাতে একটি বই রয়েছে",
        "জীবনের ছন্দ",
    ]

    selected = select_top_captions_for_image(captions)

    assert len(selected) == 2
    assert "একটি লোক লাল টুপি পরে বসে আছে" in selected
    assert normalize_for_assertion("লোকটির হাতে একটি বই রয়েছে") in selected
    assert "একটি লোক লাল টুপি পরে আছে" not in selected


def test_select_top_captions_caps_custom_max_captions():
    captions = [
        "একটি লোক লাল টুপি পরে বসে আছে",
        "লোকটির হাতে একটি বই রয়েছে",
        "একটি গরু মাঠে দাঁড়িয়ে আছে",
    ]

    selected = select_top_captions_for_image(captions, max_captions=1)

    assert selected == ["একটি লোক লাল টুপি পরে বসে আছে"]


def test_select_top_captions_falls_back_when_all_captions_are_low_quality():
    captions = ["Caption 1", "Caption 2", "Caption 1"]

    selected = select_top_captions_for_image(captions)

    assert selected == ["Caption 1", "Caption 2"]
