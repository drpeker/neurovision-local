"""Robot actions for the local brain: each returns one or two short German sentences to speak.

The hardware modules are NeuroVision's own (motors, tracker, camera, vision, memory, body,
display) and keep their internal names ("ileri", "kırmızı", "mutlu"); only what is spoken is
German. run() returns None when the request should be answered by the LLM instead.
"""

import asyncio
import logging
import re
import time

from . import web
from .german import number_word, phrase, plural

log = logging.getLogger(__name__)

VOLUME_STEP = 15
DIRECTIONS = {"ileri": "Ich fahre vorwärts.", "geri": "Ich fahre zurück.", "sol": "Ich drehe nach links.",
              "sag": "Ich drehe nach rechts."}
WEEKDAYS = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
MONTHS = ["Januar", "Februar", "März", "April", "Mai", "Juni", "Juli", "August", "September", "Oktober",
          "November", "Dezember"]
# Stored facts quote the user (router.fact); spoken back they are turned to "du"
QUOTE = re.compile(r"^Der Nutzer sagte: „(.*?)\.?“$")
# In a "dass" clause the verb is last: "dass ich einen Hund habe" -> "dass du einen Hund hast"
LAST_VERB = {"habe": "hast", "hab": "hast", "bin": "bist", "kann": "kannst", "will": "willst", "muss": "musst",
             "mag": "magst", "darf": "darfst", "arbeite": "arbeitest", "heiße": "heißt"}
YOU = ((r"\bich bin\b", "du bist"), (r"\bich habe\b", "du hast"), (r"\bich hab\b", "du hast"),
       (r"\bich heiße\b", "du heißt"), (r"\bich arbeite\b", "du arbeitest"),
       (r"\bich (\w+)e\b", r"du \1st"), (r"\bbin ich\b", "bist du"), (r"\bhabe ich\b", "hast du"),
       (r"\bich\b", "du"), (r"\bmich\b", "dich"), (r"\bmir\b", "dir"), (r"\bmein(e|en|em|er|es)?\b", r"dein\1"))


def you(fact: str) -> str:
    """'Der Nutzer sagte: „Meine Schlüssel sind im Flur.“' -> 'Deine Schlüssel sind im Flur.'"""
    m = QUOTE.match(fact.strip())
    text = m.group(1) if m else fact.strip().rstrip(".")
    clause = re.match(r"dass\s+", text, re.I)
    if clause and re.search(r"\bich\b", text, re.I):
        words = text.split()
        words[-1] = LAST_VERB.get(words[-1].lower()) or re.sub(r"e$", "st", words[-1])
        text = " ".join(words)
    for a, b in YOU:
        text = re.sub(a, b, text, flags=re.I)
    if clause:
        return "Du hast gesagt, d" + text[1:] + "."
    return text[:1].upper() + text[1:] + "."


def upper(text: str) -> str:
    return text[:1].upper() + text[1:]


def _where(cx: float) -> str:
    return "links" if cx < 0.33 else "rechts" if cx > 0.67 else "vor mir"


def describe(detections) -> str:
    """'Ich sehe eine Person vor mir und eine rote Tasse links.'"""
    if not detections:
        return "Ich sehe gerade nichts, was ich erkenne."
    groups: dict[tuple[str, str | None, str], int] = {}
    for d in sorted(detections, key=lambda d: -d.conf)[:6]:
        color = d.colors[0] if d.colors and d.name != "person" else None
        key = (d.name, color, _where(d.cx))
        groups[key] = groups.get(key, 0) + 1
    parts = [f"{phrase(name, color) if n == 1 else plural(n, name, color)} {where}"
             for (name, color, where), n in groups.items()]
    text = ", ".join(parts[:-1]) + (" und " if len(parts) > 1 else "") + parts[-1]
    return f"Ich sehe {text}."


class Actions:
    def __init__(self, cfg: dict, camera, memory, body, speaker):
        self._cfg = cfg
        self._camera = camera
        self._memory = memory
        self._body = body
        self._speaker = speaker
        self.face = None
        self.motors = None
        self.tracker = None
        self.detector = None
        self.on_shutdown = None  # robot.py: on_shutdown("robot" | "sistem")
        self.wait_spoken = None  # brain: coroutine that returns when speech has finished
        self.last_seen: tuple[float, str] | None = None  # (time, description) for the LLM's notes

    async def run(self, tool: str, args: dict) -> str | None:
        try:
            return await getattr(self, f"_{tool}")(**args)
        except Exception as e:
            log.warning("action %s failed: %s", tool, e)
            return "Entschuldigung, das hat nicht geklappt."

    # --- movement ---
    async def _stop(self) -> str:
        if self.tracker and self.tracker.active:
            await self.tracker.stop()
        if not self.motors:
            return "Okay."
        await self.motors.stop()
        return "Angehalten."

    async def _move(self, direction: str, seconds: float, speed: str = "normal") -> str:
        if not self.motors:
            return "Meine Motoren sind nicht angeschlossen."
        if self.tracker and self.tracker.active:
            await self.tracker.stop()  # driving by hand ends tracking
        result = await self.motors.move(direction, max(0.25, min(3.0, seconds)), speed)
        return DIRECTIONS[direction] if "error" not in result else "In diese Richtung kann ich nicht fahren."

    async def _track(self, target: str | None, color: str | None = None) -> str:
        if not self.tracker:
            return "Ich kann gerade nichts verfolgen; meine Kamera oder meine Motoren sind nicht bereit."
        if not target:
            return "Ich kann nur Dingen folgen, die ich erkenne, zum Beispiel einer Person, einer Tasse oder einer Flasche."
        result = await self.tracker.start(target, color=color)
        if "error" in result:
            return "Dem kann ich nicht folgen."
        if target == "person":
            return "Okay, ich folge dir."
        return f"Okay, ich folge {phrase(target, color, 'dat', 'def')}."

    async def _turn_to(self, target: str | None, color: str | None = None) -> str:
        if not self.tracker:
            return "Ich kann mich gerade nicht zu Dingen drehen; meine Kamera oder meine Motoren sind nicht bereit."
        if not target:
            return "Ich weiß nicht, welches Ding du meinst."
        await self.tracker.stop()
        hits = await self._detect(target, color)
        if hits is None:
            return "Meine Kamera liefert kein Bild."
        if not hits:
            return f"Ich sehe gerade {phrase(target, color, negative=True)}."
        e = max(hits, key=lambda d: d.conf).cx - 0.5
        if abs(e) <= self._cfg["vision"]["center_tolerance"]:
            return f"{upper(phrase(target, color, 'nom', 'def'))} ist direkt vor mir."
        await self.tracker.turn_toward(e)
        return f"Ich drehe mich zu {phrase(target, color, 'dat', 'def')}."

    # --- seeing ---
    async def _detect(self, target: str | None = None, color: str | None = None):
        jpeg = self._camera.frame() if self._camera else None
        if not jpeg or not self.detector:
            return None
        if self.face:
            self.face.flash()
            self.face.preview(jpeg)
        if target == "colored object":
            return await self.detector.find_color(jpeg, color)
        found = await self.detector.detect(jpeg)
        if target:
            found = [d for d in found if d.name == target and (not color or color in d.colors)]
        return found

    async def _look(self, find: str | None = None, color: str | None = None) -> str:
        if not self._camera:
            return "Ich habe keine Kamera."
        if not self.detector:
            return "Meine Objekterkennung ist aus."
        found = await self._detect()
        if found is None:
            return "Meine Kamera liefert kein Bild."
        seen = describe(found)
        self.last_seen = (time.monotonic(), seen)
        if not find:
            return seen
        hits = [d for d in found if d.name == find and (not color or color in d.colors)]
        if hits:
            return f"Ja, ich sehe {phrase(find, color) if len(hits) == 1 else plural(len(hits), find, color)} {_where(hits[0].cx)}."
        return f"Nein, ich sehe gerade {phrase(find, color, negative=True)}."

    # --- face and levels ---
    async def _express(self, emotion: str) -> str:
        if not self.face:
            return "Ich habe keinen Bildschirm für mein Gesicht."
        self.face.set_emotion(emotion)
        return "So?"

    async def _volume(self, action: str, level: int | None = None) -> str:
        v = self._speaker.volume
        if action == "sor":
            return f"Meine Lautstärke ist {v} Prozent."
        if action == "ayarla" and level is not None:
            v = level
        else:
            v += VOLUME_STEP if action == "artır" else -VOLUME_STEP
        self._speaker.volume = max(10, v)  # fully silent, you could not hear the robot again
        self._memory.set("volume", str(self._speaker.volume))
        return f"Die Lautstärke ist jetzt {self._speaker.volume} Prozent."

    async def _motor_speed(self, action: str, level: int | None = None) -> str:
        if not self.motors:
            return "Meine Motoren sind nicht angeschlossen."
        if action == "ayarla" and level is not None:
            self.motors.speed_level = level
        elif action in ("artır", "azalt"):
            self.motors.speed_level += 10 if action == "artır" else -10
        if action != "sor":
            self._memory.set("motor_speed", str(self.motors.speed_level))
        return f"Meine Motorgeschwindigkeit ist {self.motors.speed_level} Prozent."

    # --- body ---
    async def _body_status(self, section: str = "ozet") -> str:
        s = await self._body.status(section)
        temp = s.get("isi", {}).get("işlemci_°C")
        cpu = s.get("cpu", {}).get("toplam_%")
        ram = s.get("ram", {})
        disk = s.get("disk", {})
        net = s.get("ag", {})
        power = s.get("guc", {})
        low_now = "düşük voltaj" in power.get("şu_an", [])
        low_before = "düşük voltaj" in power.get("açılıştan_beri", [])
        hot = temp is not None and temp >= self._cfg["body"]["warn_temp_c"]
        if section == "isi":
            return f"Mein Prozessor hat {temp:.0f} Grad." + (" Das ist heiß." if hot else "")
        if section == "cpu":
            return f"Mein Prozessor ist zu {cpu} Prozent ausgelastet."
        if section == "ram":
            return f"Mein Arbeitsspeicher ist zu {ram.get('kullanım_%')} Prozent belegt, {ram.get('boş_MB')} Megabyte sind frei."
        if section == "disk":
            return f"Mein Speicher ist zu {disk.get('kullanım_%')} Prozent voll, {disk.get('boş_GB')} Gigabyte sind frei."
        if section == "ag":
            dbm = net.get("wifi_sinyal_dBm")
            if dbm is None:
                return "Ich bin nicht mit dem WLAN verbunden."
            quality = "gut" if dbm > -60 else "mittel" if dbm > -72 else "schwach"
            return f"Mein WLAN-Signal ist {quality}, {dbm} dBm. Meine Adresse ist {net.get('ip')}."
        if section == "guc":
            if low_now:
                return "Meine Stromversorgung ist gerade zu schwach: Ich messe Unterspannung."
            return "Die Stromversorgung ist in Ordnung." + (" Seit dem Start gab es einmal Unterspannung." if low_before else "")
        parts = [f"Prozessor bei {temp:.0f} Grad" if temp is not None else None,
                 f"{cpu} Prozent Auslastung" if cpu is not None else None,
                 f"Arbeitsspeicher zu {ram.get('kullanım_%')} Prozent belegt" if ram else None]
        warnings = []
        if hot:
            warnings.append("mir ist heiß")
        if low_now or low_before:
            warnings.append("ich hatte Unterspannung" if not low_now else "mein Strom ist zu schwach")
        head = "Mir geht es gut" if not warnings else "Es geht, aber " + " und ".join(warnings)
        return f"{head}: " + ", ".join(p for p in parts if p) + "."

    # --- memory ---
    async def _remember(self, text: str) -> str:
        category = "kişi" if re.search(r"\b(name|heiße|heißt|tochter|sohn|frau|mann|mutter|vater|freund\w*)\b", text, re.I) \
            else "ev" if re.search(r"\b(wohne|lebe) in\b|\b(komme|bin) aus\b", text, re.I) else "diğer"
        place = re.search(r"\b(?:wohne in|lebe in|komme aus|bin aus) ([A-ZÄÖÜ][\w-]+)", text)
        if place:
            self._memory.set("location", place.group(1))  # default place for the weather
        fid = self._memory.remember(text, category)
        return "Okay, das merke ich mir." if fid else "Das weiß ich schon."

    async def _recall(self, query: str) -> str:
        facts = [t for _, _, t in self._memory.recall(query, limit=3)]
        if facts:
            return " ".join(you(f) for f in facts[:2])
        turns = self._memory.search_turns(query, limit=2)
        if turns:
            ts, role, text = turns[0]
            who = "Du hast" if role == "user" else "Ich habe"
            day = time.localtime(ts)
            return f"{who} am {day.tm_mday}. {MONTHS[day.tm_mon - 1]} gesagt: {text}"
        return "Daran erinnere ich mich nicht."

    async def _forget(self, query: str) -> str:
        removed = self._memory.forget(query)
        return f"Okay, ich habe vergessen: {you(removed[0])}" if removed else "Das habe ich nicht in meinem Gedächtnis gefunden."

    # --- information ---
    async def _time(self) -> str:
        now = time.localtime()
        return (f"Es ist {now.tm_hour} Uhr {now.tm_min}, {WEEKDAYS[now.tm_wday]}, "
                f"der {now.tm_mday}. {MONTHS[now.tm_mon - 1]}.").replace(" Uhr 0,", " Uhr,")

    async def _weather(self, place: str | None, days: int = 1) -> str:
        place = place or self._memory.get("location") or self._cfg["local"].get("home_city")
        if not place:
            return "Für welche Stadt soll ich nachsehen?"
        try:
            w = await web.weather(place, max(days, 2))
        except Exception as e:
            log.info("weather failed: %s", e)
            return "Ich erreiche den Wetterdienst gerade nicht."
        if "error" in w:
            return f"Ich habe keinen Ort namens {place} gefunden."
        now, d = w["now"], w["days"]
        city = w["place"].split(",")[0]
        if days == 2 and len(d) > 1:
            t = d[1]
            return (f"Morgen in {city}: {t['condition']}, {t['min_c']:.0f} bis {t['max_c']:.0f} Grad, "
                    f"Regenwahrscheinlichkeit {t['rain_chance']} Prozent.")
        today = d[0]
        return (f"In {city} sind es {now['temp_c']:.0f} Grad, {now['condition']}. "
                f"Heute {today['min_c']:.0f} bis {today['max_c']:.0f} Grad, Regenwahrscheinlichkeit {today['rain_chance']} Prozent.")

    async def _wiki(self, query: str) -> str | None:
        try:
            r = await web.wiki(query)
        except Exception as e:
            log.info("wiki failed, the LLM answers: %s", e)
            return None
        results = r.get("results") or []
        if not results:
            return None
        text = re.sub(r"\s*\([^)]*\)", "", results[0]["summary"])  # drop pronunciations and dates in brackets
        sentences = re.split(r"(?<=[.!?])\s+", text)
        return " ".join(sentences[:2])

    # --- power ---
    async def _shutdown(self, target: str) -> str:
        if not self.on_shutdown:
            return "Von hier aus kann ich mich nicht ausschalten."
        asyncio.create_task(self._shutdown_after_speech(target), name="shutdown")
        return "Auf Wiedersehen. Ich fahre herunter." if target == "sistem" else "Auf Wiedersehen. Ich beende mein Programm."

    async def _shutdown_after_speech(self, target: str) -> None:
        await asyncio.sleep(0.5)
        if self.wait_spoken:
            await self.wait_spoken()
        log.info("shutdown: %s", target)
        self.on_shutdown(target)

    async def _chat(self) -> None:
        return None
