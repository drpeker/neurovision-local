"""Nesne takibi: robot yalnizca yerinde donerek hedefi kadrajin ortasinda tutar.

Dongu: hareket bittikten sonra cekilen yeni kare -> YOLO -> hedefin yatay konumu ->
ortadan sapma kadar kisa donus darbesi. Hareket sirasindaki bulanik karelere bakilmaz.
Hedef kaybolursa son goruldugu yone dogru adim adim aranir. Gemini dongude degil;
yalnizca "buldum" / "kaybettim" anlarinda kisa bir bildirim alir.
Takilma: her darbeden sonra goruntu kaymasi olculur; art arda stall_pulses darbede robot
beklenenin cok altinda donduyse (batarya bitti, koruma kesti, palet sikisti) takip durur.
"""

import asyncio
import logging
import time

from .vision import COCO80, COLORED, COLORS, Detection, Detector, turkish

log = logging.getLogger(__name__)


class Tracker:
    def __init__(self, cfg: dict, camera, motors, detector: Detector):
        self._cfg = cfg
        self._camera, self._motors, self._detector = camera, motors, detector
        self._task: asyncio.Task | None = None
        self.target: str | None = None
        self.color: str | None = None  # istege bagli: "turuncu" kase, "siyah" fare
        self.last_seen = 0.0
        self._boost = 1.0  # zemine gore guc carpani (takip boyunca ogrenilir)
        self._stalls = 0  # art arda donmeyen darbe sayisi
        self.face = None  # gozler hedefe baksin
        self.notify = None  # LiveBrain: kisa olay bildirimi (metin)

    @property
    def active(self) -> bool:
        return bool(self._task and not self._task.done())

    async def start(self, target: str, seconds: float | None = None, color: str | None = None) -> dict:
        if target not in COCO80 and target != COLORED:
            return {"error": f"'{target}' tanınan nesneler arasında yok; renkliyse colored object kullan."}
        if color and color not in COLORS:
            color = None
        if target == COLORED and not color:
            return {"error": "Renkle takip için color gerekli (örneğin kırmızı)."}
        await self.stop()
        self.color = color
        c = self._cfg
        seconds = max(5.0, min(float(seconds or c["timeout_s"]), c["max_timeout_s"]))
        self.target = target
        self._task = asyncio.create_task(self._run(target, seconds), name="tracker")
        log.info("takip basladi: %s%s (%.0f sn)", target, f" ({color})" if color else "", seconds)
        return {"result": f"{self._label()} takibi başladı ({seconds:.0f} sn). Bulunca ya da "
                          "kaybedince sana kısa bir bildirim gelecek; o zamana kadar konuşma."}

    async def stop(self) -> dict:
        was = self.target if self.active else None
        if self.active:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        return {"result": f"{turkish(was)} takibi durdu." if was else "Takip zaten kapalı."}

    def _label(self) -> str:
        if self.target == COLORED:
            return f"{self.color} nesne"
        return f"{self.color} {turkish(self.target)}" if self.color else turkish(self.target)

    def status(self) -> dict:
        seen = [f"{'/'.join(d.colors[:2])} {turkish(d.name)} {d.conf:.0%}".strip()
                for d in self._detector.last[:6]]
        return {
            "takip": f"{self._label()} takip ediliyor" if self.active else "kapalı",
            "hedef_son_görülme_sn": round(time.monotonic() - self.last_seen, 1)
            if self.active and self.last_seen else None,
            "son_karede_görülenler": seen or "yok",
            "model": "YOLO11n 320 (yerel, COCO 80 nesne)",
        }

    # --- dongu ---
    async def _settle(self) -> float:
        """Motor dursun (mikrofon sessiz suresi dahil); durdugu ani dondur."""
        while self._motors.busy:
            await asyncio.sleep(0.05)
        return time.monotonic()

    async def _frame_after(self, t: float) -> bytes | None:
        """t'den sonra cekilmis kare (hareket bulaniksiz)."""
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            if self._camera.latest_at > t + 0.05:
                return self._camera.frame()
            await asyncio.sleep(0.05)
        return None

    def plan(self, e: float) -> tuple[float, float, float]:
        """Sapma (goruntu genisliginin orani) -> (guc %, sure sn, beklenen derece).

        turn_table yerde olculdu: her guc icin 0.25 sn darbede kac derece donuyor.
        Kucuk sapma dusuk guc, buyuk sapma yuksek guc ve daha uzun darbe.
        """
        c = self._cfg
        angle = abs(e) * c["hfov_deg"] * self._boost
        for duty, deg in c["turn_table"]:
            if deg >= angle:
                return duty, 0.25, deg
        duty, deg = c["turn_table"][-1]
        rate = c.get("turn_rate_dps")
        if rate:
            # Ilk 0.25 sn kalkista gider; sonra donus bu hizla surer (yerde olculdu)
            seconds = min(c["pulse_max"], 0.25 + (angle - deg) / rate)
            return duty, seconds, deg + (seconds - 0.25) * rate
        seconds = min(c["pulse_max"], 0.25 * angle / deg)
        return duty, seconds, deg * seconds / 0.25

    async def _moved(self, before: bytes, after: bytes, expected: float,
                     target_deg: float | None = None) -> bool:
        """Darbe sonrasi donuldu mu? Art arda takilmada False (takip durmali).

        target_deg: hedefin kadrajdaki kaymasi (derece); yoksa goruntu kaymasi olculur.
        """
        if target_deg is not None:
            deg = target_deg
        elif expected > self._cfg["hfov_deg"] / 2:
            return True  # iki kare ortusmez, goruntu kaymasi olculemez: karar verme
        else:
            deg = await self._detector.shift(before, after, self._cfg["hfov_deg"])
        if deg < self._cfg["stall_ratio"] * expected:
            self._stalls += 1
            log.info("takip: donus beklenenden az (%.1f / %.0f derece), %d. kez", deg, expected, self._stalls)
        else:
            self._stalls = 0
        return self._stalls < self._cfg["stall_pulses"]

    async def turn_toward(self, e: float) -> float:
        """Sapma kadar don (yumusak rampasiz, kalibre guc); beklenen derece."""
        duty, seconds, deg = self.plan(e)
        await self._motors.move("sol" if e < 0 else "sag", seconds, duty=duty, soft=False)
        log.debug("takip: e=%+.2f -> %%%d %.2f sn (~%.0f derece, carpan %.2f)", e, duty, seconds, deg, self._boost)
        return deg

    def _adapt(self, prev_e: float, e: float, expected_deg: float) -> None:
        """Zemin farki: beklenenden az donduyse gucu artir, cok donduyse azalt."""
        moved = abs(prev_e - e) * self._cfg["hfov_deg"]
        if moved < 0.4 * expected_deg:
            self._boost = min(2.0, self._boost * 1.3)
        elif moved > 1.6 * expected_deg:
            self._boost = max(0.6, self._boost * 0.8)

    def _pick(self, found: list[Detection], prev_cx: float | None) -> Detection | None:
        th = 0.0 if self.target == COLORED else self._cfg["target_threshold"]
        cand = [d for d in found if d.name == self.target and d.conf >= th]
        if self.color:
            # Renk istendiyse: o renk kutunun ortasinda baskin renklerden biri olmali
            cand = [d for d in cand if self.color in d.colors]
        if not cand:
            return None
        if prev_cx is None:
            return cand[0]  # en yuksek guven
        # Birden fazla varsa onceki konuma en yakini: kisiden kisiye atlamasin
        return min(cand, key=lambda d: abs(d.cx - prev_cx) - d.conf * 0.1)

    async def _say(self, text: str) -> None:
        if self.notify:
            await self.notify(text)

    async def _run(self, target: str, seconds: float) -> None:
        c = self._cfg
        name = self._label()
        start = time.monotonic()
        prev_cx, last_dir, searched = None, 1, 0
        turned = None  # (donusten onceki sapma, beklenen derece)
        pulse = None  # (donusten onceki kare, beklenen derece): takilma kontrolu
        self._boost, self._stalls = 1.0, 0
        self.last_seen = 0.0
        announced = lost = False  # lost: "kayboldu" denildi, bulununca "tekrar bulundu" denir
        try:
            settled = await self._settle()
            # Sure hedefin son goruldugu andan sayilir: gordukce takip surer (en fazla max_timeout_s)
            while (time.monotonic() - max(start, self.last_seen) < seconds
                   and time.monotonic() - start < c["max_timeout_s"]):
                jpeg = await self._frame_after(settled)
                if not jpeg:
                    log.warning("takip: kamera karesi yok")
                    await self._say("Takip durdu: kameradan görüntü alınamadı. Kısaca söyle.")
                    return
                found = (await self._detector.find_color(jpeg, self.color) if target == COLORED
                         else await self._detector.detect(jpeg))
                hit = self._pick(found, prev_cx)
                if pulse:
                    # Hedef donusten once ve sonra gorunduyse kadrajdaki kaymasi donusu gosterir;
                    # buyuk donuste iki kare ortusmez, goruntu kaymasi ~0 olcup "takildi" diyordu
                    target_deg = (abs(turned[0] - (hit.cx - 0.5)) * c["hfov_deg"]
                                  if hit and turned else None)
                    moved = await self._moved(pulse[0], jpeg, pulse[1], target_deg)
                    pulse = None
                    if not moved:
                        log.warning("takip durdu: motorlar donmuyor (batarya/koruma/sikisma?)")
                        check = await self._motors.signal_check()
                        if check["pwm_sinyali"] == "var":
                            await self._say("Takip durdu: motor sinyalleri gidiyor ama dönemiyorsun; motor "
                                            "bataryası bitmiş ya da korumaya girmiş olabilir. Kullanıcıya "
                                            "kısaca söyle, bataryayı kontrol etmesini iste.")
                        else:
                            await self._say("Takip durdu: motor pinlerine sinyal gitmiyor, bu bir yazılım "
                                            "sorunu. Kullanıcıya kısaca söyle; yeniden başlatma gerekebilir.")
                        return
                now = time.monotonic()
                if hit:
                    self.last_seen, searched = now, 0
                    e = hit.cx - 0.5  # <0: hedef goruntunun solunda
                    prev_cx = hit.cx
                    if turned:
                        self._adapt(turned[0], e, turned[1])
                        turned = None
                    last_dir = -1 if e < 0 else 1
                    if self.face:
                        self.face.gaze(e)
                    if not announced:
                        announced = True
                        await self._say(f"Takip: {name} görüldü, ona dönüyorsun. Bunu tek kısa cümleyle söyle.")
                    elif lost:
                        await self._say(f"Takip: {name} tekrar bulundu, ona dönüyorsun. "
                                        "Bunu tek kısa cümleyle söyle.")
                    lost = False
                    if abs(e) > c["center_tolerance"]:
                        turned = (e, await self.turn_toward(e))
                        pulse = (jpeg, turned[1])
                        settled = await self._settle()
                        # Dinleme araligi: mikrofon acik kalsin, "dur" duyulsun
                        await asyncio.sleep(c["listen_s"])
                    else:
                        settled = now  # ortada: donme, yeni kareyi bekle
                        await asyncio.sleep(0.2)
                    continue
                if self.last_seen and now - self.last_seen < c["lost_s"]:
                    settled = now  # kisa kayip: bir kare daha bak
                    continue
                if searched >= c["search_steps"]:
                    log.info("takip: %s bulunamadi, birakildi", target)
                    await self._say(f"Takip bitti: {name} bulunamadı. Bunu tek kısa cümleyle söyle.")
                    return
                if searched == 0 and announced:
                    lost = True
                    await self._say(f"Takip: {name} kayboldu, arıyorsun. Bunu tek kısa cümleyle söyle.")
                searched += 1
                duty = c["search_duty"]
                # Aramada takilma kontrolu yok: hedef cikinca kamera cogu zaman duz duvara bakar,
                # goruntu kaymasi olculemez (~0 derece) ve donen robot "takildi" sayiliyordu
                pulse = None
                await self._motors.move("sol" if last_dir < 0 else "sag", c["search_pulse"],
                                        duty=duty, soft=False)
                settled = await self._settle()
                await asyncio.sleep(c["listen_s"])
            log.info("takip suresi doldu: %s", target)
            await self._say(f"Takip bitti: süre doldu ({name}). Bunu tek kısa cümleyle söyle.")
        finally:
            if self.face:
                self.face.gaze(None)
