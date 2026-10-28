"""German words for what the robot sees and says: nouns with gender, colours, articles, numbers.

The copied NeuroVision modules keep their internal names (COCO classes "cup", colours
"kırmızı"); this module turns them into correct German phrases:

    phrase("cup", "kırmızı", "acc", "indef") -> "eine rote Tasse"
    phrase("dining table", None, "dat", "def") -> "dem Tisch"
"""

import re

from .vision import COLORED

# COCO class -> (singular, gender m/f/n, plural)
NOUNS = {
    "person": ("Person", "f", "Personen"), "bicycle": ("Fahrrad", "n", "Fahrräder"),
    "car": ("Auto", "n", "Autos"), "motorcycle": ("Motorrad", "n", "Motorräder"),
    "airplane": ("Flugzeug", "n", "Flugzeuge"), "bus": ("Bus", "m", "Busse"), "train": ("Zug", "m", "Züge"),
    "truck": ("Lastwagen", "m", "Lastwagen"), "boat": ("Boot", "n", "Boote"),
    "traffic light": ("Ampel", "f", "Ampeln"), "fire hydrant": ("Hydrant", "m", "Hydranten"),
    "stop sign": ("Stoppschild", "n", "Stoppschilder"), "parking meter": ("Parkuhr", "f", "Parkuhren"),
    "bench": ("Bank", "f", "Bänke"), "bird": ("Vogel", "m", "Vögel"), "cat": ("Katze", "f", "Katzen"),
    "dog": ("Hund", "m", "Hunde"), "horse": ("Pferd", "n", "Pferde"), "sheep": ("Schaf", "n", "Schafe"),
    "cow": ("Kuh", "f", "Kühe"), "elephant": ("Elefant", "m", "Elefanten"), "bear": ("Bär", "m", "Bären"),
    "zebra": ("Zebra", "n", "Zebras"), "giraffe": ("Giraffe", "f", "Giraffen"),
    "backpack": ("Rucksack", "m", "Rucksäcke"), "umbrella": ("Regenschirm", "m", "Regenschirme"),
    "handbag": ("Handtasche", "f", "Handtaschen"), "tie": ("Krawatte", "f", "Krawatten"),
    "suitcase": ("Koffer", "m", "Koffer"), "frisbee": ("Frisbee", "n", "Frisbees"), "skis": ("Ski", "m", "Skier"),
    "snowboard": ("Snowboard", "n", "Snowboards"), "sports ball": ("Ball", "m", "Bälle"),
    "kite": ("Drachen", "m", "Drachen"), "baseball bat": ("Baseballschläger", "m", "Baseballschläger"),
    "baseball glove": ("Baseballhandschuh", "m", "Baseballhandschuhe"),
    "skateboard": ("Skateboard", "n", "Skateboards"), "surfboard": ("Surfbrett", "n", "Surfbretter"),
    "tennis racket": ("Tennisschläger", "m", "Tennisschläger"), "bottle": ("Flasche", "f", "Flaschen"),
    "wine glass": ("Weinglas", "n", "Weingläser"), "cup": ("Tasse", "f", "Tassen"), "fork": ("Gabel", "f", "Gabeln"),
    "knife": ("Messer", "n", "Messer"), "spoon": ("Löffel", "m", "Löffel"), "bowl": ("Schüssel", "f", "Schüsseln"),
    "banana": ("Banane", "f", "Bananen"), "apple": ("Apfel", "m", "Äpfel"), "sandwich": ("Sandwich", "n", "Sandwiches"),
    "orange": ("Orange", "f", "Orangen"), "broccoli": ("Brokkoli", "m", "Brokkoli"),
    "carrot": ("Karotte", "f", "Karotten"), "hot dog": ("Hotdog", "m", "Hotdogs"), "pizza": ("Pizza", "f", "Pizzen"),
    "donut": ("Donut", "m", "Donuts"), "cake": ("Kuchen", "m", "Kuchen"), "chair": ("Stuhl", "m", "Stühle"),
    "couch": ("Sofa", "n", "Sofas"), "potted plant": ("Pflanze", "f", "Pflanzen"), "bed": ("Bett", "n", "Betten"),
    "dining table": ("Tisch", "m", "Tische"), "toilet": ("Toilette", "f", "Toiletten"),
    "tv": ("Fernseher", "m", "Fernseher"), "laptop": ("Laptop", "m", "Laptops"), "mouse": ("Maus", "f", "Mäuse"),
    "remote": ("Fernbedienung", "f", "Fernbedienungen"), "keyboard": ("Tastatur", "f", "Tastaturen"),
    "cell phone": ("Handy", "n", "Handys"), "microwave": ("Mikrowelle", "f", "Mikrowellen"),
    "oven": ("Ofen", "m", "Öfen"), "toaster": ("Toaster", "m", "Toaster"), "sink": ("Waschbecken", "n", "Waschbecken"),
    "refrigerator": ("Kühlschrank", "m", "Kühlschränke"), "book": ("Buch", "n", "Bücher"), "clock": ("Uhr", "f", "Uhren"),
    "vase": ("Vase", "f", "Vasen"), "scissors": ("Schere", "f", "Scheren"),
    "teddy bear": ("Teddybär", "m", "Teddybären"), "hair drier": ("Föhn", "m", "Föhne"),
    "toothbrush": ("Zahnbürste", "f", "Zahnbürsten"), COLORED: ("Gegenstand", "m", "Gegenstände"),
}

# NeuroVision colour -> German adjective stem ("lila"/"rosa" do not decline, so the -farben forms)
COLORS = {"kırmızı": "rot", "turuncu": "orangefarben", "sarı": "gelb", "yeşil": "grün", "mavi": "blau",
          "mor": "lilafarben", "pembe": "rosafarben", "kahverengi": "braun", "siyah": "schwarz",
          "beyaz": "weiß", "gri": "grau"}

ARTICLES = {
    "def": {"nom": {"m": "der", "f": "die", "n": "das"}, "acc": {"m": "den", "f": "die", "n": "das"},
            "dat": {"m": "dem", "f": "der", "n": "dem"}},
    "indef": {"nom": {"m": "ein", "f": "eine", "n": "ein"}, "acc": {"m": "einen", "f": "eine", "n": "ein"},
              "dat": {"m": "einem", "f": "einer", "n": "einem"}},
}
# Adjective endings after the article
ENDINGS = {
    "def": {"nom": {"m": "e", "f": "e", "n": "e"}, "acc": {"m": "en", "f": "e", "n": "e"},
            "dat": {"m": "en", "f": "en", "n": "en"}},
    "indef": {"nom": {"m": "er", "f": "e", "n": "es"}, "acc": {"m": "en", "f": "e", "n": "es"},
              "dat": {"m": "en", "f": "en", "n": "en"}},
}


def noun(name: str) -> tuple[str, str, str]:
    return NOUNS.get(name, (name, "n", name))


def phrase(name: str, color: str | None = None, case: str = "acc", article: str = "indef",
           negative: bool = False) -> str:
    """'eine rote Tasse', 'den Hund', 'keinen blauen Ball' (negative: kein- instead of ein-)."""
    word, g, _ = noun(name)
    art = ARTICLES[article][case][g]
    if negative and article == "indef":
        art = "k" + art
    adj = COLORS.get(color or "")
    return f"{art} {adj + ENDINGS[article][case][g] + ' ' if adj else ''}{word}"


def plural(n: int, name: str, color: str | None = None) -> str:
    """'zwei rote Tassen' (nominative and accusative are the same here)."""
    adj = COLORS.get(color or "")
    return f"{number_word(n)} {adj + 'e ' if adj else ''}{noun(name)[2]}"


def color_word(color: str | None) -> str:
    """Predicative colour ('rot'), for 'etwas Rotes' style phrases."""
    return COLORS.get(color or "", "")


# --- numbers ---
UNITS = {"eins": 1, "ein": 1, "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "sechs": 6, "sieben": 7, "acht": 8,
         "neun": 9}
TEENS = {"zehn": 10, "elf": 11, "zwölf": 12, "dreizehn": 13, "vierzehn": 14, "fünfzehn": 15, "sechzehn": 16,
         "siebzehn": 17, "achtzehn": 18, "neunzehn": 19}
TENS = {"zwanzig": 20, "dreißig": 30, "dreissig": 30, "vierzig": 40, "fünfzig": 50, "sechzig": 60,
        "siebzig": 70, "achtzig": 80, "neunzig": 90}
WORDS = {v: k for k, v in {**UNITS, **TEENS}.items() if k != "ein"}


def number_word(n: int) -> str:
    return WORDS.get(n, str(n))


def _parse(word: str) -> int | None:
    if word in ("hundert", "einhundert"):
        return 100
    if word in TEENS:
        return TEENS[word]
    if word in TENS:
        return TENS[word]
    m = re.fullmatch(r"(ein|zwei|drei|vier|fünf|sechs|sieben|acht|neun)und(\w+ig)", word)
    if m and m.group(2) in TENS:
        return UNITS[m.group(1)] + TENS[m.group(2)]
    if word in UNITS and word != "ein":
        return UNITS[word]
    return None


def numbers_to_digits(text: str) -> str:
    """'auf sechzig' -> 'auf 60'; 'ein'/'eine' stay words (they are mostly articles)."""
    return re.sub(r"\b[a-zäöüß]+\b", lambda m: str(n) if (n := _parse(m.group(0))) is not None else m.group(0), text)
