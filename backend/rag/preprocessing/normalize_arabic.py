"""
normalize_arabic.py
-------------------
Utility functions to normalize Arabic text before embedding or indexing.
- Remove diacritics (tashkeel/harakat)
- Normalize letter variants (Alef, Teh marbuta, etc.)
- Strip extra whitespace
"""

import re
import unicodedata


# Arabic diacritics Unicode range
ARABIC_DIACRITICS_PATTERN = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06DC\u06DF-\u06E4\u06E7\u06E8\u06EA-\u06ED]"
)

# Alef variants → plain Alef ا
ALEF_PATTERN = re.compile(r"[إأآا]")

# Teh marbuta → Heh
TEH_MARBUTA_PATTERN = re.compile(r"ة")

# Waw with hamza above → plain Waw
WAW_HAMZA_PATTERN = re.compile(r"ؤ")

# Yeh with hamza above → plain Yeh
YEH_HAMZA_PATTERN = re.compile(r"ئ")

# Alef maqsura → Yeh
ALEF_MAQSURA_PATTERN = re.compile(r"ى")


def remove_diacritics(text: str) -> str:
    """Remove Arabic tashkeel/harakat from text."""
    return ARABIC_DIACRITICS_PATTERN.sub("", text)


def normalize_letters(text: str) -> str:
    """Normalize common Arabic letter variants to their base forms."""
    text = ALEF_PATTERN.sub("ا", text)
    text = TEH_MARBUTA_PATTERN.sub("ة", text)   # keep teh marbuta as-is (important for matching)
    text = WAW_HAMZA_PATTERN.sub("و", text)
    text = YEH_HAMZA_PATTERN.sub("ي", text)
    text = ALEF_MAQSURA_PATTERN.sub("ي", text)
    return text


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces/newlines into a single space."""
    return re.sub(r"\s+", " ", text).strip()


def normalize_arabic(text: str) -> str:
    """
    Full normalization pipeline:
    1. Remove diacritics
    2. Normalize letter variants
    3. Normalize whitespace
    """
    if not text:
        return ""
    text = remove_diacritics(text)
    text = normalize_letters(text)
    text = normalize_whitespace(text)
    return text


if __name__ == "__main__":
    sample = "الْمَادَّةُ 124 : كُلُّ فِعْلٍ أَيًّا كَانَ يَرْتَكِبُهُ الشَّخْصُ بِخَطَئِهِ"
    print("Original :", sample)
    print("Normalized:", normalize_arabic(sample))
