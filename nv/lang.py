"""Yaziya cevrilen konusma Turkce mi? Gurultu cogu zaman Almanca/Ispanyolca/Hintce kelimelere
donusuyor ("Achtzig", "querido niño", "साधनम"); bunlarla robot hareket etmemeli.

Kaba ama hizli: Turkce disi alfabe ya da harf, veya cogunlugu yabanci yaygin kelimelerden olusan cumle.
"""

import re

# Turkce alfabe; disindaki harf (q, w, x, ñ, ß, é, Kiril, Devanagari...) yabanci
TURKISH_LETTERS = set("abcçdefgğhıijklmnoöprsştuüvyzâîû")
# Turkce kelimelerde gecmeyen harf ciftleri ("Achtzig", "Faden" degil ama "nicht", "the")
FOREIGN_PAIRS = re.compile(r"ch|th|ck|tz|sh|ph|oo|ee|ou")
FOREIGN_WORDS = {
    # Ingilizce
    "the", "and", "you", "is", "are", "what", "english", "this", "that", "with", "for", "of",
    "it", "my", "your", "be", "was", "have", "i'm", "it's", "don't", "can",
    # Almanca
    "und", "ich", "nicht", "die", "der", "das", "ist", "ein", "eine", "sie", "wir", "mit", "auf",
    # Ispanyolca / Portekizce / Italyanca / Fransizca
    "que", "el", "los", "las", "una", "un", "pero", "tiene", "por", "como", "muy", "né", "é", "não",
    "il", "di", "che", "le", "les", "est", "et", "je", "pas", "vous",
}


def looks_foreign(text: str) -> bool:
    letters = [c for c in text.lower() if c.isalpha()]
    if not letters:
        return False
    if any(c not in TURKISH_LETTERS for c in letters):
        return True
    words = re.findall(r"[\w']+", text.lower())
    foreign = sum(w in FOREIGN_WORDS or bool(FOREIGN_PAIRS.search(w)) for w in words)
    return foreign / len(words) >= 0.4 if words else False
