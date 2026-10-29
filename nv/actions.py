"""Robot actions for the local brain: each returns one or two short English sentences to speak.

The hardware modules are NeuroVision's own (motors, tracker, camera, vision, memory, body,
display) and keep their internal names ("ileri", "kırmızı", "mutlu"); only what is spoken is
English. run() returns None when the request should be answered by the LLM instead.
"""

import asyncio
import logging
import re
import time

from . import web
from .audio import set_mic_boost
from .router import COLORS

log = logging.getLogger(__name__)

VOLUME_STEP = 15
MIC_STEP_DB = 4
DIRECTIONS = {"ileri": "Moving forward.", "geri": "Backing up.", "sol": "Turning left.", "sag": "Turning right."}


def _decibels(db: int) -> str:
    """Spoken by Piper: "plus 12 decibels" (it would read "+12 dB" letter by letter)."""
    return f"{'minus' if db < 0 else 'plus'} {abs(db)} decibels" if db else "0 decibels"


COLOR_WORDS = {v: k for k, v in COLORS.items() if k not in ("violet", "grey")}
# Speaking about stored facts: they are written about "the user"
YOU = ((r"\bthe user's\b", "your"), (r"\bthe user is\b", "you are"), (r"\bthe user has\b", "you have"),
       (r"\bthe user\b", "you"))


def you(text: str) -> str:
    for a, b in YOU:
        text = re.sub(a, b, text, flags=re.I)
    return text[:1].upper() + text[1:]


def _where(cx: float) -> str:
    return "on the left" if cx < 0.33 else "on the right" if cx > 0.67 else "in front of me"


def _name(d) -> str:
    color = COLOR_WORDS.get(d.colors[0]) if d.colors else None
    return f"{color} {d.name}" if color and d.name != "person" else d.name


def describe(detections) -> str:
    """'I can see a person in front of me and a red cup on the left.'"""
    if not detections:
        return "I don't see anything I recognise right now."
    groups: dict[tuple[str, str], int] = {}
    for d in sorted(detections, key=lambda d: -d.conf)[:6]:
        key = (_name(d), _where(d.cx))
        groups[key] = groups.get(key, 0) + 1
    parts = []
    for (name, where), n in groups.items():
        if n == 1:
            article = "an" if name[0] in "aeiou" else "a"
            parts.append(f"{article} {name} {where}")
        else:
            plural = "people" if name == "person" else name + "s"
            parts.append(f"{n} {plural} {where}")
    text = ", ".join(parts[:-1]) + (" and " if len(parts) > 1 else "") + parts[-1]
    return f"I can see {text}."


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
        self.turner = None  # nv.imu.AngleTurner (MPU6050): turns by angle instead of by time
        self._turn_task: asyncio.Task | None = None
        self.on_shutdown = None  # robot.py: on_shutdown("robot" | "sistem")
        self.wait_spoken = None  # brain: coroutine that returns when speech has finished
        self.last_seen: tuple[float, str] | None = None  # (time, description) for the LLM's notes
        self.gate = None  # brain's SpeechGate: its noise floor follows a microphone gain change
        self.mic_db = int(memory.get("mic_boost_db") or cfg["audio"]["mic_boost_db"])

    async def run(self, tool: str, args: dict) -> str | None:
        try:
            return await getattr(self, f"_{tool}")(**args)
        except Exception as e:
            log.warning("action %s failed: %s", tool, e)
            return "Sorry, that did not work."

    # --- movement ---
    async def _cancel_turn(self) -> None:
        if self._turn_task and not self._turn_task.done():
            self._turn_task.cancel()
            try:
                await self._turn_task
            except asyncio.CancelledError:
                pass

    async def _stop(self) -> str:
        await self._cancel_turn()
        if self.tracker and self.tracker.active:
            await self.tracker.stop()
        if not self.motors:
            return "Okay."
        await self.motors.stop()
        return "Stopped."

    async def _move(self, direction: str, seconds: float, speed: str = "normal", degrees: float | None = None) -> str:
        if not self.motors:
            return "My motors are not connected."
        if self.tracker and self.tracker.active:
            await self.tracker.stop()  # driving by hand ends tracking
        await self._cancel_turn()
        if self.turner and direction in ("sol", "sag") and degrees:
            # By the gyro: the turn runs on while the robot answers; "stop" cancels it
            sign = 1 if direction == "sol" else -1
            self._turn_task = asyncio.create_task(self.turner.turn(sign * degrees), name="turn")
            return DIRECTIONS[direction]
        result = await self.motors.move(direction, max(0.25, min(3.0, seconds)), speed)
        return DIRECTIONS[direction] if "error" not in result else "I can't move that way."

    async def _track(self, target: str | None, color: str | None = None, side: str | None = None) -> str:
        if not self.tracker:
            return "I can't follow things right now; my camera or motors are not ready."
        if not target:
            return "I can only follow things I recognise, like a person, a cup or a bottle."
        await self._cancel_turn()  # one owner of the motors at a time
        result = await self.tracker.start(target, color=color, side=side)
        if "error" in result:
            return "I can't follow that."
        if target == "person":
            return "Okay, I'm following you."
        label = f"{COLOR_WORDS.get(color, '')} {'object' if target == 'colored object' else target}".strip()
        if side:
            return f"Okay, I'll look to my {'left' if side == 'sol' else 'right'} for the {label}."
        return f"Okay, I'm following the {label}."

    async def _turn_to(self, target: str | None, color: str | None = None) -> str:
        if not self.tracker:
            return "I can't turn to things right now; my camera or motors are not ready."
        if not target:
            return "I don't know which thing you mean."
        await self.tracker.stop()
        hits = await self._detect(target, color)
        if hits is None:
            return "My camera is not giving me a picture."
        if not hits:
            return f"I don't see a {target} right now."
        e = max(hits, key=lambda d: d.conf).cx - 0.5
        if abs(e) <= self._cfg["vision"]["center_tolerance"]:
            return f"The {target} is right in front of me."
        if self.turner:
            await self._cancel_turn()
            self._turn_task = asyncio.create_task(self.turner.turn(-e * self._cfg["vision"]["hfov_deg"]),
                                                  name="turn")
        else:
            await self.tracker.turn_toward(e)
        return f"Turning to the {target}."

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

    async def _look(self, find: str | None = None, color: str | None = None, me: bool = False) -> str:
        if not self._camera:
            return "I don't have a camera."
        if not self.detector:
            return "My object recognition is off."
        found = await self._detect()
        if found is None:
            return "My camera is not giving me a picture."
        seen = describe(found)
        self.last_seen = (time.monotonic(), seen)
        if not find:
            return seen
        hits = [d for d in found if d.name == find and (not color or color in d.colors)]
        if me:
            # The detector knows objects and colours, not faces or clothes: say what it really sees
            others = describe([d for d in found if d.name != "person"])
            if not hits:
                return "No, I don't see you right now."
            you = max(hits, key=lambda d: d.conf)
            rest = f" {others[0].upper()}{others[1:]}" if not others.startswith("I don't") else ""
            return f"I can see you {_where(you.cx)}.{rest} I recognise objects and colours, not faces or clothes."
        if hits:
            return f"Yes, I can see {'it' if len(hits) == 1 else 'them'} {_where(hits[0].cx)}."
        return f"No, I don't see a {COLOR_WORDS.get(color, '')} {find} right now.".replace("  ", " ")

    # --- face and levels ---
    async def _express(self, emotion: str) -> str:
        if not self.face:
            return "I don't have a face screen connected."
        self.face.set_emotion(emotion)
        return "Like this?"

    async def _volume(self, action: str, level: int | None = None) -> str:
        v = self._speaker.volume
        if action == "sor":
            return f"My volume is {v} percent."
        if action == "ayarla" and level is not None:
            v = level
        else:
            v += VOLUME_STEP if action == "artır" else -VOLUME_STEP
        self._speaker.volume = max(10, v)  # fully silent, you could not hear the robot again
        self._memory.set("volume", str(self._speaker.volume))
        return f"Volume is now {self._speaker.volume} percent."

    async def _mic(self, action: str, level: int | None = None) -> str:
        """Microphone gain as a percent of the "Mic Boost" control (0 % = -10 dB, 100 % = +30 dB)."""
        db = self.mic_db
        if action == "sor":
            return f"My microphone is at {(db + 10) * 100 // 40} percent, {_decibels(db)}."
        if action == "ayarla" and level is not None:
            new = round(level * 40 / 100) - 10
        else:
            new = db + (MIC_STEP_DB if action == "artır" else -MIC_STEP_DB)
        # Below 0 dB speech hardly reaches the speech threshold; above +24 dB loud speech clips
        new = max(0, min(24, new))
        set_mic_boost(new, self._cfg["audio"]["card"])
        if self.gate:
            self.gate.floor_db += new - db  # room noise rises and falls with the gain
        self.mic_db = new
        self._memory.set("mic_boost_db", str(new))
        return f"My microphone is now at {(new + 10) * 100 // 40} percent, {_decibels(new)}."

    async def _levels(self) -> str:
        return f"My speaker is at {self._speaker.volume} percent. {await self._mic('sor')}"

    async def _motor_speed(self, action: str, level: int | None = None) -> str:
        if not self.motors:
            return "My motors are not connected."
        if action == "ayarla" and level is not None:
            self.motors.speed_level = level
        elif action in ("artır", "azalt"):
            self.motors.speed_level += 10 if action == "artır" else -10
        if action != "sor":
            self._memory.set("motor_speed", str(self.motors.speed_level))
        return f"My motor speed is {self.motors.speed_level} percent."

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
            return f"My processor is at {temp:.0f} degrees." + (" That is hot." if hot else "")
        if section == "cpu":
            return f"My processor is {cpu} percent busy."
        if section == "ram":
            return f"My memory is {ram.get('kullanım_%')} percent used, {ram.get('boş_MB')} megabytes are free."
        if section == "disk":
            return f"My disk is {disk.get('kullanım_%')} percent full, {disk.get('boş_GB')} gigabytes are free."
        if section == "ag":
            dbm = net.get("wifi_sinyal_dBm")
            if dbm is None:
                return "I'm not connected to Wi-Fi."
            quality = "good" if dbm > -60 else "fair" if dbm > -72 else "weak"
            return f"My Wi-Fi signal is {quality}, {dbm} dBm. My address is {net.get('ip')}."
        if section == "guc":
            if low_now:
                return "My power supply is too weak right now: I'm seeing low voltage."
            return "Power is fine." + (" There was a low voltage moment since I started." if low_before else "")
        parts = [f"processor at {temp:.0f} degrees" if temp is not None else None,
                 f"{cpu} percent busy" if cpu is not None else None,
                 f"memory {ram.get('kullanım_%')} percent used" if ram else None]
        warnings = []
        if hot:
            warnings.append("I'm running hot")
        if low_now or low_before:
            warnings.append("I had low voltage" if not low_now else "my power is too low")
        head = "I feel fine" if not warnings else "I'm okay, but " + " and ".join(warnings)
        return f"{head}: " + ", ".join(p for p in parts if p) + "."

    # --- memory ---
    async def _remember(self, text: str) -> str:
        category = "kişi" if re.search(r"\b(name|daughter|son|wife|husband|mother|father|friend)\b", text, re.I) \
            else "ev" if re.search(r"\blives? in\b|\bfrom\b", text, re.I) else "diğer"
        place = re.search(r"\b(?:lives in|is from) ([A-Z][\w-]+)", text)
        if place:
            self._memory.set("location", place.group(1))  # default place for the weather
        fid = self._memory.remember(text, category)
        return "Okay, I'll remember that." if fid else "I already know that."

    async def _recall(self, query: str) -> str:
        facts = [t for _, _, t in self._memory.recall(query, limit=3)]
        if facts:
            return " ".join(you(f) for f in facts[:2])
        turns = self._memory.search_turns(query, limit=2)
        if turns:
            ts, role, text = turns[0]
            who = "You" if role == "user" else "I"
            return f"{who} said on {time.strftime('%B %d', time.localtime(ts))}: {text}"
        return "I don't remember anything about that."

    async def _forget(self, query: str) -> str:
        removed = self._memory.forget(query)
        return f"Okay, I forgot that {you(removed[0]).rstrip('.')[0].lower()}{you(removed[0]).rstrip('.')[1:]}." \
            if removed else "I couldn't find that in my memory."

    # --- information ---
    async def _time(self) -> str:
        now = time.localtime()
        return time.strftime("It's %H:%M, %A, %B %d.", now).replace(" 0", " ")

    async def _weather(self, place: str | None, days: int = 1) -> str:
        place = place or self._memory.get("location") or self._cfg["local"].get("home_city")
        if not place:
            return "Which city should I check?"
        try:
            w = await web.weather(place, max(days, 2))
        except Exception as e:
            log.info("weather failed: %s", e)
            return "I can't reach the weather service right now."
        if "error" in w:
            return f"I couldn't find a place called {place}."
        now, d = w["now"], w["days"]
        city = w["place"].split(",")[0]
        if days == 2 and len(d) > 1:
            t = d[1]
            return (f"Tomorrow in {city}: {t['condition']}, {t['min_c']:.0f} to {t['max_c']:.0f} degrees, "
                    f"{t['rain_chance']} percent chance of rain.")
        today = d[0]
        return (f"In {city} it's {now['temp_c']:.0f} degrees and {now['condition']}. "
                f"Today {today['min_c']:.0f} to {today['max_c']:.0f} degrees, {today['rain_chance']} percent chance of rain.")

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
            return "I can't switch myself off from here."
        asyncio.create_task(self._shutdown_after_speech(target), name="shutdown")
        return "Goodbye. I'm shutting down." if target == "sistem" else "Goodbye. I'm closing my program."

    async def _shutdown_after_speech(self, target: str) -> None:
        await asyncio.sleep(0.5)
        if self.wait_spoken:
            await self.wait_spoken()
        log.info("shutdown: %s", target)
        self.on_shutdown(target)

    async def _chat(self) -> None:
        return None
