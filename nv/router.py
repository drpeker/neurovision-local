"""German command router: robot commands are matched by words, not by the LLM.

Measured with Qwen3 1.7B on a Pi 4: letting the LLM pick a tool took 7-13 s per command
(even "Stopp!") and chose the right tool 9 times out of 12. Here a command is understood
in well under a millisecond and anything that is not a command goes to the LLM as chat.

route(text) -> (tool, args). The tool names and argument values are the ones the copied
NeuroVision modules use (motors: "ileri"/"sol", colours: "kırmızı", emotions: "mutlu").
German puts the verb last in many requests ("Kannst du nach links fahren?"), so driving and
following look for the verb anywhere in the sentence, not only at the start.
"""

import re

from .german import NOUNS, numbers_to_digits
from .vision import COLORED

# Spoken names -> COCO class (singular and plural of every German noun, plus everyday words)
OBJECTS: dict[str, str] = {}
for _coco, (_one, _, _many) in NOUNS.items():
    if _coco != COLORED:
        OBJECTS[_one.lower()] = OBJECTS[_many.lower()] = _coco
OBJECTS.update({
    "mich": "person", "mir": "person", "uns": "person", "jemand": "person", "jemanden": "person",
    "mensch": "person", "menschen": "person", "leute": "person", "mann": "person", "frau": "person",
    "kind": "person", "kinder": "person", "becher": "cup", "glas": "cup", "tisch": "dining table",
    "schreibtisch": "dining table", "couch": "couch", "blume": "potted plant", "blumen": "potted plant",
    "bildschirm": "tv", "monitor": "tv", "computer": "laptop", "rechner": "laptop", "notebook": "laptop",
    "telefon": "cell phone", "smartphone": "cell phone", "tablet": "cell phone", "tasche": "handbag",
    "teddy": "teddy bear", "kätzchen": "cat", "welpe": "dog", "rad": "bicycle", "fußball": "sports ball",
    "controller": "remote", "glases": "cup", "balls": "sports ball",
})
OBJECT_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, OBJECTS), key=len, reverse=True)) + r")(?:e|n|en|s|es)?\b")

COLORS = {"rot": "kırmızı", "orange": "turuncu", "orangefarben": "turuncu", "gelb": "sarı", "grün": "yeşil",
          "blau": "mavi", "lila": "mor", "violett": "mor", "rosa": "pembe", "pink": "pembe",
          "braun": "kahverengi", "schwarz": "siyah", "weiß": "beyaz", "weiss": "beyaz", "grau": "gri"}
COLOR_RE = re.compile(r"\b(" + "|".join(sorted(COLORS, key=len, reverse=True)) + r")(?:e|en|er|es|em)?\b")

EMOTIONS = {"glücklich": "mutlu", "fröhlich": "mutlu", "lächle": "mutlu", "lächeln": "mutlu", "lach": "mutlu",
            "traurig": "üzgün", "überrascht": "şaşkın", "erstaunt": "şaşkın", "wütend": "kızgın",
            "böse": "kızgın", "sauer": "kızgın", "müde": "uykulu", "schläfrig": "uykulu", "neutral": "nötr",
            "normal": "nötr"}
EMOTION_RE = r"(glücklich|fröhlich|traurig|überrascht|erstaunt|wütend|böse|sauer|müde|schläfrig|neutral|normal)(?:e|en|er|es)?"

NUMBERS = {"eine halbe": 0.5, "halbe": 0.5, "anderthalb": 1.5, "eineinhalb": 1.5, "eine": 1, "einer": 1}

# A request may be polite ("kannst du bitte ..."); a question about something is not a command
POLITE = r"^(?:hey |hallo |ok |okay |bitte |jetzt |los |neurovision |roboter )*" \
         r"(?:kannst du |könntest du |würdest du |kann ich dich bitten )?(?:bitte |mal |jetzt )*"
QUESTION = re.compile(r"^(wie|warum|wieso|weshalb|was|wann|wo|wohin|woher|wer|welche\w*|ist|sind|war|hast|hat|bist|"
                      r"gibt|erzähl\w*|sag mir|weißt)\b")
STATEMENT = re.compile(r"^(ich|wir|er|sie|es|man|der|die|das)\b")


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"(\d),(\d)", r"\1.\2", t)  # "1,5 sekunden"
    t = re.sub(r"[^a-zäöüß0-9.' ]", " ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)  # keep "1.5", drop sentence dots
    return numbers_to_digits(re.sub(r" +", " ", t).strip())


def _order(t: str, verbs: str) -> re.Match | None:
    """The sentence starts with one of the verbs (after polite words)."""
    return re.match(POLITE + r"(?:" + verbs + r")\b", t)


def _request(t: str) -> bool:
    """An order or a polite request, not a question or a statement about oneself."""
    return not QUESTION.match(t) and not STATEMENT.match(t)


def _seconds(t: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?|eine halbe|halbe|anderthalb|eineinhalb|eine|einer) (?:halbe )?sekunden?", t)
    if m:
        return NUMBERS.get(m.group(1)) or float(m.group(1))
    if re.search(r"\b(ein bisschen|ein bißchen|bisschen|ein wenig|etwas|ein stück|ein stückchen|kurz|leicht)\b", t):
        return 0.3
    if re.search(r"\b(weit|viel|ein ganzes stück|lange|länger|mehr)\b", t):
        return 2.0
    return 1.0


def _speed(t: str) -> str:
    if re.search(r"\bsehr langsam\b|\bganz langsam\b", t):
        return "çok yavaş"
    if re.search(r"\b(langsam|vorsichtig|sachte|behutsam)\b", t):
        return "yavaş"
    if re.search(r"\b(schnell|zügig|flott)\b", t):
        return "hızlı"
    return "normal"


def _target(phrase: str) -> tuple[str | None, str | None]:
    """'die rote tasse' -> ("cup", "kırmızı"); 'das blaue ding' -> ("colored object", "mavi")."""
    color = None
    m = COLOR_RE.search(phrase)
    # "Orange" alone is the fruit; with another object word after it, it is a colour
    if m and not (m.group(1) == "orange" and not OBJECT_RE.search(phrase[m.end():])):
        color = COLORS[m.group(1)]
        phrase = phrase[:m.start()] + phrase[m.end():]
    o = OBJECT_RE.search(phrase)
    if o:
        return OBJECTS[o.group(1)], color
    return (COLORED, color) if color else (None, None)


def _strip_that(text: str) -> str:
    """'Merk dir, dass meine Schlüssel in der Schale sind' -> 'Dass meine Schlüssel in der Schale sind'."""
    t = text.strip().rstrip(".!? ")
    t = re.sub(r"^(?:\w+\s+)*?(?:merk|merke|notier|notiere|vergiss nicht|denk daran|denke daran)\w*(?:\s+dir)?[,:]?\s*", "",
               t, flags=re.I)
    return t[:1].upper() + t[1:]


def fact(text: str) -> str:
    """Stored facts quote the user: 'Der Nutzer sagte: „Ich heiße Mustafa.“'. In the first person
    the LLM read 'my name' as its own name, so the speaker is always named."""
    return f"Der Nutzer sagte: „{_strip_that(text)}.“"


def route(text: str) -> tuple[str, dict]:
    t = normalize(text)
    if not t:
        return "chat", {}

    # --- safety first ---
    if _order(t, r"stopp|stop|stoppe|halt|anhalten|halte an|halt an|bleib stehen|bleib stehn|stehen bleiben|"
                 r"nicht bewegen|beweg dich nicht|hör auf|aufhören|warte|genug|schluss|aus jetzt") \
            or t in ("warte", "genug", "stopp", "halt"):
        return "stop", {}
    if _order(t, r"(halten|bleiben) sie (an|stehen)|hören sie auf"):
        return "stop", {}
    # A short "Stopp!" often loses its first sound or comes out English; stopping is always safe
    if re.fullmatch(r"(top|tops|stopp?t|stoppen|shop|stop it|hopp|s top|stob|stopf)( jetzt| bitte| sofort)?", t):
        return "stop", {}

    # --- power ---
    if re.search(r"\b(schalt\w* (dich|das system|den pi|den roboter|alles|den computer|das programm) (aus|ab)|"
                 r"ausschalten|abschalten|herunterfahren|runterfahren|fahr\w* (dich |das system |den pi )?(herunter|runter)|"
                 r"beende\w*|beenden)\b", t) and not QUESTION.match(t):
        if re.search(r"\b(system|pi|raspberry|computer|komplett|ganz|alles|herunterfahren|runterfahren|herunter|runter)\b", t):
            return "shutdown", {"target": "sistem"}
        if re.search(r"\b(dich|roboter|programm|neurovision|app|beende\w*|beenden)\b", t):
            return "shutdown", {"target": "robot"}

    # --- vision (questions are allowed here) ---
    # Anything asking what the robot sees is a look; the LLM would make a scene up
    if re.search(r"\bwas siehst du\b|\bwas\b.*\bdu\b.*\b(sehen|siehst)\b|\bwas\b.*\bsiehst\b"
                 r"|^(schau|guck|sieh|blick)( dich| mal)*( um| herum)?$|\bwas ist (vor dir|um dich|da vorne|vor dir los)\b"
                 r"|\b(schau|guck|sieh) dich (mal )?um\b|\bbeschreib\w* (was du siehst|den raum|deine umgebung|das bild)\b"
                 r"|\bwas ist (auf|in) (der|deiner) kamera\b", t):
        return "look", {}
    m = re.search(r"\b(?:siehst|erkennst|findest) du (?:hier |gerade |irgendwo )?(.+)|\bkannst du (.+?) sehen\b", t)
    if m:
        obj, color = _target(m.group(1) or m.group(2))
        return "look", {"find": obj, "color": color}

    # --- following and turning to things ---
    if re.search(r"\b(schau|guck|sieh|blick)\w* (sie )?(mich|uns) an\b", t) and not QUESTION.match(t):
        return "track", {"target": "person", "color": None}
    m = re.search(r"\b(?:dreh|wend)\w* dich (?:zu|zur|zum|in richtung|nach) (.+)|\b(?:schau|guck|sieh|blick)\w* (?:zu|zur|zum|auf) (.+)", t)
    if m and _request(t):
        rest = m.group(1) or m.group(2)
        if not re.match(r"(der |die |dem )?(links|rechts|linken|rechten)\b", rest):
            obj, color = _target(rest)
            if obj == "person":
                return "track", {"target": "person", "color": None}
            if obj:
                return "turn_to", {"target": obj, "color": color}
    m = re.search(r"\b(folg|folge|folgen|verfolg|verfolge|verfolgen|beobacht|beobachte|beobachten|follow)\b|\bbehalt\w* .* im auge\b", t)
    if m and _request(t):
        obj, color = _target(t[:m.start()] + " " + t[m.end():])
        return "track", {"target": obj, "color": color}

    # --- speaker and motor levels ---
    m = re.search(r"\blautstärke\b.*?\b(\d{1,3})\b|\b(\d{1,3})\s*(?:prozent )?lautstärke\b", t)
    if m:
        return "volume", {"action": "ayarla", "level": int(m.group(1) or m.group(2))}
    if re.search(r"\b(lauter|lautstärke (hoch|rauf|erhöhen)|(erhöh|dreh)\w* (die |deine )?lautstärke( hoch| auf)?|sprich lauter)\b", t):
        return "volume", {"action": "artır"}
    if re.search(r"\b(leiser|zu laut|lautstärke (runter|herunter|verringern)|(verringer|senk|reduzier)\w* (die |deine )?lautstärke"
                 r"|dreh\w* (die |deine )?lautstärke (runter|herunter))\b", t):
        return "volume", {"action": "azalt"}
    if re.search(r"\b(wie laut bist du|wie (hoch )?ist (deine|die) lautstärke|welche lautstärke)\b", t):
        return "volume", {"action": "sor"}
    m = re.search(r"\b(?:motor|fahr)?geschwindigkeit\b.*?\b(\d{1,3})\b", t)
    if m:
        return "motor_speed", {"action": "ayarla", "level": int(m.group(1))}
    if re.search(r"^(fahr|fahre|beweg dich) (ab jetzt |in zukunft )?schneller( ab jetzt| in zukunft)?$"
                 r"|\b(erhöh|steiger)\w* (die |deine )?(motor|fahr)?geschwindigkeit\b", t):
        return "motor_speed", {"action": "artır"}
    if re.search(r"^(fahr|fahre|beweg dich) (ab jetzt |in zukunft )?langsamer( ab jetzt| in zukunft)?$"
                 r"|\b(verringer|senk|reduzier)\w* (die |deine )?(motor|fahr)?geschwindigkeit\b", t):
        return "motor_speed", {"action": "azalt"}
    if re.search(r"\b(wie schnell (bist|fährst) du|wie (hoch )?ist (deine|die) (motor|fahr)?geschwindigkeit)\b", t):
        return "motor_speed", {"action": "sor"}

    # --- driving ---
    if _request(t):
        direction = None
        # Parakeet recognises 25 languages and now and then writes a short German command in
        # English ("Fahr vorwärts." -> "Far forwards."), so the English words count too
        verb = re.search(r"\b(fahr|fahre|fahren|dreh|drehe|drehen|geh|gehe|gehen|beweg|bewege|bewegen|roll|rolle|rollen|"
                         r"lenk|lenke|lenken|steuer|steuere|steuern|wend|wende|wenden|setz|setze|setzen|schau|guck|"
                         r"far|go|turn|drive|move)\b", t)
        # Short orders whose verb was lost or misheard: "Nach links!", "Rehen Sie nach links."
        alone = len(t.split()) <= 5 and re.search(r"\b(nach (links|rechts|vorne|vorn|hinten)|vorwärts|rückwärts|geradeaus)\b"
                                                  r"|^(links|rechts|zurück|vor|vorne)\b", t)
        if verb or alone:
            if re.search(r"\b(links|linksrum|left)\b", t):
                direction = "sol"
            elif re.search(r"\b(rechts|rechtsrum|right)\b", t):
                direction = "sag"
            elif re.search(r"\b(rückwärts|zurück|nach hinten|retour|back|backward|backwards)\b", t):
                direction = "geri"
            elif re.search(r"\b(vorwärts|vor|vorne|vorn|geradeaus|weiter|forward|forwards|ahead|straight)\b", t):
                direction = "ileri"
        # "Komm zurück" / "komm her" said to a robot means "come to me": drive toward the user
        if _order(t, r"komm|komme|kommen|come"):
            direction = "ileri"
        if direction:
            return "move", {"direction": direction, "seconds": _seconds(t), "speed": _speed(t)}

    # --- the robot's own body ---
    sections = [("isi", r"\b(temperatur(?! (in|draußen|morgen))|wie heiß|wie warm|heiß|warm|überhitz\w*)\b"),
                ("cpu", r"\b(cpu|prozessor\w*|auslastung|last)\b"), ("ram", r"\b(arbeitsspeicher|ram|speicherauslastung)\b"),
                ("guc", r"\b(akku|batterie|stromversorgung|netzteil|spannung|unterspannung)\b"),
                ("disk", r"\b(festplatte|speicherplatz|sd karte|disk)\b"),
                ("ag", r"\b(wlan|wifi|wi fi|netzwerk|internetverbindung|signal|ip adresse)\b")]
    if re.search(r"\b(du|dein|deine|deinen|deiner|deinem|dir|dich)\b", t) or re.search(r"\bstatus\b", t):
        for section, pattern in sections:
            if re.search(pattern, t):
                return "body_status", {"section": section}
        if re.search(r"\bwie geht es dir\b|\bwie geht's dir\b|\bwie fühlst du dich\b"
                     r"|\b(statusbericht|systemstatus|selbsttest|status)\b", t):
            return "body_status", {"section": "ozet"}

    # --- face ---
    m = re.search(r"\b(?:sei|schau|guck|sieh)\w* (?:mal )?(?:ganz |so )?" + EMOTION_RE + r"(?: aus)?\b"
                  r"|\b(?:zeig|mach)\w* (?:mir )?(?:mal )?(?:ein|dein)\w* " + EMOTION_RE + r" gesicht\b"
                  r"|^(?:bitte )?(lächle|lächeln|lach)\b", t)
    if m and not QUESTION.match(t):
        word = next(g for g in m.groups() if g)
        return "express", {"emotion": EMOTIONS[word]}

    # --- memory ---
    m = re.match(POLITE + r"(?:merk dir|merke dir|vergiss nicht|denk daran|denke daran|notier dir|notiere)\b,? ?(?:dass )?(.+)", t)
    if m and not re.match(r"(was|wann|wo|wer|mich|meinen namen)\b", m.group(1)):
        return "remember", {"text": fact(text)}
    if re.match(r"^(?:mein name ist|mein geburtstag|meine? lieblings\w+|meine? \w+ heißt|meine? \w+ ist|ich heiße|ich wohne in|"
                r"ich lebe in|ich komme aus|ich bin aus|ich arbeite als|ich arbeite bei)\b", t) \
            and not re.match(r"^meine? \w+ ist (das|es|so|sehr|nicht|kaputt)\b", t):
        return "remember", {"text": fact(text)}
    m = re.match(POLITE + r"vergiss (?:das mit |alles über |was ich über )?(.+?)(?: gesagt habe)?$", t)
    if m:
        return "forget", {"query": m.group(1)}
    m = re.search(r"\bwas weißt du (?:noch )?über (.+)|\berinnerst du dich an (.+)|\bwo (?:sind|ist|habe ich|hab ich) meine?n? (.+?)"
                  r"(?: hingelegt| gelassen| gelegt)?$|\bwie (?:heißt|ist) meine?r? (.+)|\bwer bin ich\b|\bwie heiße ich\b"
                  r"|\bworüber haben wir (?:gesprochen|geredet)\b", t)
    if m:
        query = next((g for g in m.groups() if g), None) or ("heiße name" if re.search(r"wer bin ich|wie heiße ich", t) else "gesprochen")
        return "recall", {"query": re.sub(r"^(name|vorname)$", "heiße name", query)}

    # --- information ---
    if re.search(r"\b(wetter|wettervorhersage|vorhersage|regnet|regnen|regen|schnee\w*|schneit|regenschirm|gewitter"
                 r"|temperatur draußen|wie (warm|kalt|heiß) (ist|wird) es)\b", t):
        place = re.search(r"\b(?:in|für) ([a-zäöüß][a-zäöüß .'-]+?)(?: heute| morgen| übermorgen| diese woche| jetzt| gerade)*$", t)
        days = 2 if re.search(r"\bmorgen\b", t) else 7 if re.search(r"\b(diese woche|woche)\b", t) else 1
        return "weather", {"place": place.group(1).strip() if place else None, "days": days}
    if re.search(r"\bwie spät ist es\b|\bwie viel uhr\b|\bwieviel uhr\b|\bwelche uhrzeit\b|\bwelcher tag ist (es|heute)\b"
                 r"|\bwelches datum\b|\bder wievielte\b|\bden wievielten\b", t):
        return "time", {}
    m = re.match(POLITE + r"(?:such nach|suche nach|schlag|schlage|google|wikipedia|recherchiere) (.+?)(?: nach)?$", t) \
        or re.match(r"^wer (?:ist|war) (?!du\b|das\b|da\b|dein\b|deine\b)(.+)", t)
    if m:
        return "wiki", {"query": m.group(1)}

    return "chat", {}
