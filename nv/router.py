"""English command router: robot commands are matched by words, not by the LLM.

Measured with Qwen3 1.7B on a Pi 4: letting the LLM pick a tool took 7-13 s per command
(even "Stop!") and chose the right tool 9 times out of 12. Here a command is understood
in well under a millisecond and anything that is not a command goes to the LLM as chat.

route(text) -> (tool, args). The tool names and argument values are the ones the copied
NeuroVision modules use (motors: "ileri"/"sol", colours: "kırmızı", emotions: "mutlu").
"""

import re

from .vision import COCO80, COLORED

# Spoken names -> COCO class (longest first so "cell phone" wins over "phone")
OBJECTS = {name: name for name in COCO80}
OBJECTS.update({
    "me": "person", "us": "person", "you": "person", "him": "person", "her": "person", "someone": "person",
    "people": "person", "man": "person", "woman": "person", "kid": "person", "child": "person",
    "phone": "cell phone", "mobile": "cell phone", "smartphone": "cell phone", "mug": "cup", "glass": "cup",
    "ball": "sports ball", "football": "sports ball", "table": "dining table", "desk": "dining table",
    "sofa": "couch", "plant": "potted plant", "flower": "potted plant", "television": "tv", "monitor": "tv",
    "screen": "tv", "computer": "laptop", "notebook": "laptop", "fridge": "refrigerator", "bag": "backpack",
    "teddy": "teddy bear", "kitty": "cat", "puppy": "dog", "bike": "bicycle", "glasses": "wine glass",
    "controller": "remote", "tablet": "cell phone",
})
OBJECT_RE = re.compile(r"\b(" + "|".join(sorted(map(re.escape, OBJECTS), key=len, reverse=True)) + r")s?\b")

COLORS = {"red": "kırmızı", "orange": "turuncu", "yellow": "sarı", "green": "yeşil", "blue": "mavi",
          "purple": "mor", "violet": "mor", "pink": "pembe", "brown": "kahverengi", "black": "siyah",
          "white": "beyaz", "gray": "gri", "grey": "gri"}
COLOR_RE = re.compile(r"\b(" + "|".join(COLORS) + r")\b")

EMOTIONS = {"happy": "mutlu", "smile": "mutlu", "sad": "üzgün", "surprised": "şaşkın", "shocked": "şaşkın",
            "angry": "kızgın", "mad": "kızgın", "sleepy": "uykulu", "tired": "uykulu", "neutral": "nötr",
            "normal": "nötr"}

NUMBERS = {"half a": 0.5, "a half": 0.5, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5}
# Levels heard as words ("microphone level two forty percent")
TENS = {"ten": 10, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90, "hundred": 100}

# A request may be polite ("can you turn left?"); a question about something is not a command
POLITE = r"^(?:hey |ok |okay |please |now |neurovision |robot )*(?:can you |could you |would you |will you )?(?:please )?"
QUESTION = re.compile(r"^(how|why|what|when|where|who|which|do|does|did|is|are|was|tell me)\b")


def normalize(text: str) -> str:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z0-9.' ]", " ", t)
    t = re.sub(r"(?<![0-9])\.|\.(?![0-9])", " ", t)  # keep "1.5", drop sentence dots
    return re.sub(r" +", " ", t).strip()


def _order(t: str, verbs: str) -> re.Match | None:
    """The sentence starts with one of the verbs (after polite words)."""
    return re.match(POLITE + r"(?:" + verbs + r")\b", t)


def _seconds(t: str) -> float:
    m = re.search(r"(\d+(?:\.\d+)?|half a|a half|one|two|three|four|five) (?:and a half )?seconds?", t)
    if m:
        value = NUMBERS.get(m.group(1)) or float(m.group(1))
        return value + (0.5 if "and a half" in m.group(0) else 0)
    if re.search(r"\b(a little|a bit|a little bit|slightly|a tiny bit|bit)\b", t):
        return 0.3
    if re.search(r"\b(a lot|far|a long way|more)\b", t):
        return 2.0
    return 1.0


def _speed(t: str) -> str:
    if re.search(r"\bvery slow(ly)?\b", t):
        return "çok yavaş"
    if re.search(r"\b(slow|slowly|gently|carefully)\b", t):
        return "yavaş"
    if re.search(r"\b(fast|quickly|quick)\b", t):
        return "hızlı"
    return "normal"


def _target(phrase: str) -> tuple[str | None, str | None]:
    """'the red cup' -> ("cup", "kırmızı"); 'the blue thing' -> ("colored object", "mavi")."""
    color = None
    m = COLOR_RE.search(phrase)
    # "orange" alone is the fruit; with another object word after it, it is a colour
    if m and not (m.group(1) == "orange" and not OBJECT_RE.search(phrase[m.end():])):
        color = COLORS[m.group(1)]
        phrase = phrase[:m.start()] + phrase[m.end():]
    o = OBJECT_RE.search(phrase)
    if o:
        return OBJECTS[o.group(1)], color
    return (COLORED, color) if color else (None, None)


def _third_person(text: str) -> str:
    """'my daughter is Elif' -> 'The user's daughter is Elif.' (stored facts are about the user)."""
    t = text.strip().rstrip(".!? ")
    t = re.sub(r"\bi (live|work|like|love|prefer|hate|own|use|play|drink|eat|need|want)\b", r"the user \1s", t, flags=re.I)
    t = re.sub(r"\bi have\b", "the user has", t, flags=re.I)
    for a, b in ((r"\bi am\b", "the user is"), (r"\bi'm\b", "the user is"), (r"\bi\b", "the user"),
                 (r"\bmy\b", "the user's"), (r"\bme\b", "the user"), (r"\bmine\b", "the user's")):
        t = re.sub(a, b, t, flags=re.I)
    return t[:1].upper() + t[1:] + "."


def route(text: str) -> tuple[str, dict]:
    t = normalize(text)
    if not t:
        return "chat", {}

    # --- safety first ---
    if _order(t, r"stop|halt|freeze|stay still|stand still|don't move|do not move") or t in ("wait", "enough"):
        return "stop", {}
    # A short "Stop!" often loses its first sound ("Top!", "Stopped", "Shop"); stopping is always safe
    if re.fullmatch(r"(top|tops|stopped|stops|stopping|shop|sup|stop it|top it|stoppit|hop|s top)( now| it| please)?", t):
        return "stop", {}

    # --- power ---
    if _order(t, r"shut down|shutdown|power off|power down|turn off|switch off|quit|exit|close") \
            or _order(t, r"(?:turn|switch|shut) (?:yourself|the robot|the system|the pi|everything) (?:off|down)"):
        if re.search(r"\b(system|pi|raspberry|computer|completely|everything|power off|power down)\b", t):
            return "shutdown", {"target": "sistem"}
        if re.search(r"\b(yourself|robot|program|neurovision|app)\b", t):
            return "shutdown", {"target": "robot"}

    # --- vision (questions are allowed here) ---
    # The LLM cannot see: asked to describe the user it made up eyes and posture for 30 s
    if re.search(r"\bdescribe (me|myself|us)\b|\bwhat do i look like\b|\bhow do i look\b", t):
        return "look", {"find": "person", "color": None, "me": True}
    # Whisper often drops "do" ("What you see at the moment?"); the LLM then got the question and
    # made up a scene, so anything asking what the robot sees is a look
    if re.search(r"\bwhat\b.*\b(you|u) (see|seeing)\b|\b(tell|show) me what you see\b|^look( around)?$"
                 r"|\bwhat('s| is) (in front|around)\b", t):
        return "look", {}
    if re.search(r"\bwhat (do|can) you see\b|\bwhat are you (looking at|seeing)\b|\blook around\b"
                 r"|\bwhat('s| is) (in front of you|around you|there|ahead)\b|\bdescribe (what you see|the room|your surroundings|it)\b"
                 r"|\bwhat('s| is) on the (camera|screen)\b", t):
        return "look", {}
    m = re.search(r"\b(?:can|do) you see (?:a |an |the |any |my )?(.+)", t)
    if m:
        obj, color = _target(m.group(1))
        if re.match(r"(me|us)\b", m.group(1)):
            return "look", {"find": "person", "color": None, "me": True}
        return "look", {"find": obj, "color": color}

    # --- following and turning to things ---
    # "Find and follow the TV", "Look for the ball and track it": the thing is before or after the verb
    # "on the left", "to your right", "left side": the robot's own side; the search starts there
    s = re.search(r"\b(?:on|to|at) (?:the |your |my )?(left|right)\b|\b(left|right) side\b", t)
    side = {"left": "sol", "right": "sag"}[s.group(1) or s.group(2)] if s else None
    m = _order(t, r"(?:(?:go )?(?:find|look for|search for) ((?:\w+ ){0,6}?)and )?(?:follow|track|watch|keep an eye on|keep looking at)")
    if m:
        obj, color = _target(t[m.end():])
        if obj is None and m.group(1):
            obj, color = _target(m.group(1))
        return "track", {"target": obj, "color": color, "side": side}
    # "Find the TV on the left": look for it and keep it in view (a thing it can recognise only;
    # "find my keys" is a question for memory)
    m = _order(t, r"(?:go )?(?:find|look for|search for)")
    if m:
        obj, color = _target(t[m.end():])
        if obj and obj != "person":
            return "track", {"target": obj, "color": color, "side": side}
    m = _order(t, r"look at|turn to|turn towards|turn toward|face")
    if m:
        rest = t[m.end():]
        if not re.match(r"\s*(the )?(left|right)\b", rest):
            obj, color = _target(rest)
            if obj == "person":
                return "track", {"target": "person", "color": None}
            return "turn_to", {"target": obj, "color": color}

    # --- speaker, microphone and motor levels ---
    # A question about the levels: both are answered from the real settings (Qwen does not know them)
    if (QUESTION.match(t) or t.startswith(("say ", "check "))) and re.search(r"\b(volume|level|levels|gain)\b", t) \
            and re.search(r"\b(microphone|mic|speaker|sound|audio|system|levels)\b", t):
        return "levels", {}
    if re.search(r"\b(microphone|mic)\b", t):
        m = re.search(r"\b(\d{1,3})\b", t) or re.search(r"\b(" + "|".join(TENS) + r")\b", t)
        if m:
            level = int(m.group(1)) if m.group(1).isdigit() else TENS[m.group(1)]
            return "mic", {"action": "ayarla", "level": level}
        if re.search(r"\b(up|louder|higher|more|increase|raise|boost)\b", t):
            return "mic", {"action": "artır"}
        if re.search(r"\b(down|quieter|lower|less|decrease|reduce)\b", t):
            return "mic", {"action": "azalt"}
    m = re.search(r"\bvolume\b.*?\b(\d{1,3})\b|\b(\d{1,3})\s*(?:percent|%)? volume\b", t)
    if m:
        return "volume", {"action": "ayarla", "level": int(m.group(1) or m.group(2))}
    if re.search(r"\b(louder|speak up|volume up|turn (it|yourself|the volume|your volume) up|increase (the |your )?volume|raise (the |your )?volume)\b", t):
        return "volume", {"action": "artır"}
    if re.search(r"\b(quieter|softer|too loud|volume down|turn (it|yourself|the volume|your volume) down|lower (the |your )?(volume|voice)|decrease (the |your )?volume)\b", t):
        return "volume", {"action": "azalt"}
    if re.search(r"\b(what('s| is)|how loud is) (your|the) volume\b", t):
        return "volume", {"action": "sor"}
    m = re.search(r"\b(?:motor|driving|drive) speed\b.*?\b(\d{1,3})\b", t)
    if m:
        return "motor_speed", {"action": "ayarla", "level": int(m.group(1))}
    if re.search(r"\b(move|drive|go) (faster|quicker)( from now on)?$|\b(increase|raise) (the |your )?(motor )?speed\b", t):
        return "motor_speed", {"action": "artır"}
    if re.search(r"\b(move|drive|go) slower( from now on)?$|\b(decrease|lower|reduce) (the |your )?(motor )?speed\b", t):
        return "motor_speed", {"action": "azalt"}
    if re.search(r"\b(what('s| is)) (your|the) (motor |driving )?speed\b", t):
        return "motor_speed", {"action": "sor"}

    # --- driving ---
    if not QUESTION.match(t):
        direction = None
        if _order(t, r"turn|rotate|spin|go|move|drive|head|steer") or re.match(POLITE + r"(left|right)\b", t):
            if re.search(r"\bleft\b", t):
                direction = "sol"
            elif re.search(r"\bright\b", t) and not re.search(r"\bright (now|away)\b", t):
                direction = "sag"
        # "Come back" / "come here" said to a robot means "come to me": drive toward the user
        if not direction and _order(t, r"come"):
            direction = "ileri"
        if not direction and (_order(t, r"go|move|drive|roll|head|walk|step|advance|forward|ahead|straight|onward")):
            if re.search(r"\b(back|backward|backwards|reverse|rear)\b", t):
                direction = "geri"
            elif re.search(r"\b(forward|forwards|ahead|straight|onward)\b|\bto the front\b", t):
                direction = "ileri"
        if not direction and _order(t, r"back up|back off|reverse|go back|move back|backward|backwards"):
            direction = "geri"
        if direction:
            return "move", {"direction": direction, "seconds": _seconds(t), "speed": _speed(t)}

    # --- the robot's own body ---
    sections = [("isi", r"\b(temperature|how hot|hot are you|warm|overheat\w*)\b"),
                ("cpu", r"\b(cpu|processor|load)\b"), ("ram", r"\b(memory usage|ram)\b"),
                ("guc", r"\b(battery|power supply|voltage|undervolt\w*)\b"), ("disk", r"\b(disk|storage)\b"),
                ("ag", r"\b(wifi|wi fi|network|internet connection|signal|ip address)\b")]
    if re.search(r"\b(you|your)\b", t) or re.search(r"\bstatus\b", t):
        for section, pattern in sections:
            if re.search(pattern, t):
                return "body_status", {"section": section}
        if re.search(r"\bhow (are you feeling|do you feel)\b|\b(status report|system status|self check|health check)\b", t):
            return "body_status", {"section": "ozet"}

    # --- face ---
    m = re.search(r"\b(?:look|be|show me (?:a |your )?|make a) (happy|sad|surprised|shocked|angry|mad|sleepy|tired|neutral|normal)\b"
                  r"|^(?:please )?(smile)\b", t)
    if m and not QUESTION.match(t):
        return "express", {"emotion": EMOTIONS[m.group(1) or m.group(2)]}

    # --- memory ---
    m = re.match(POLITE + r"(?:remember|note|keep in mind|don't forget) (?:that )?(.+)", t)
    if m and not re.match(r"(what|when|where|who|me|my name)\b", m.group(1)):
        # Keep the speaker's capitals (names) from the original text
        o = re.search(r"(?:remember|note|keep in mind|don't forget)\s+(?:that\s+)?(.+)", text, re.I)
        return "remember", {"text": _third_person(o.group(1) if o else m.group(1))}
    m = re.match(r"^(?:my (?:name|birthday|favou?rite [a-z]+|[a-z]+'s name|[a-z]+ is called)|i live in|i am from|i'm from|i work as|i work at)\b.*", t)
    if m:
        return "remember", {"text": _third_person(text)}
    m = re.match(POLITE + r"forget (?:that |about |what i said about )?(.+)", t)
    if m:
        return "forget", {"query": m.group(1)}
    m = re.search(r"\bwhat do you (?:remember|know) about (.+)|\bdo you remember (.+)|\bwhere (?:is|are|did i (?:put|leave)) my (.+)"
                  r"|\bwhat(?:'s| is) my (.+)|\bwho am i\b|\bwhat did we (?:talk|speak) about\b", t)
    if m:
        query = next((g for g in m.groups() if g), None) or ("name" if "who am i" in t else "talk")
        return "recall", {"query": query}

    # --- information ---
    if re.search(r"\b(weather|forecast|raining|rain|snow\w*|umbrella|temperature outside|how (hot|cold) is it)\b", t):
        place = re.search(r"\b(?:in|for|at) ([a-z][a-z .'-]+?)(?: today| tomorrow| this week| now)?$", t)
        days = 2 if "tomorrow" in t else 7 if re.search(r"\b(this week|week)\b", t) else 1
        return "weather", {"place": place.group(1).strip() if place else None, "days": days}
    if re.search(r"\bwhat time is it\b|\bwhat('s| is) the time\b|\bwhat (day|date) is (it|today)\b|\bwhat('s| is) (the date|today's date)\b", t):
        return "time", {}
    m = re.match(POLITE + r"(?:look up|search for|search|google|wikipedia|find out about) (.+)", t) \
        or re.match(r"^who (?:is|was) (.+)", t)
    if m:
        return "wiki", {"query": m.group(1)}

    return "chat", {}
