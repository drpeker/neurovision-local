"""L298N motor testi, dusuk hiz, kisa darbeler. Paletler yerden kaldirilmis olmali.

    python3 deploy/motor_test.py idle            pinleri sahiplen, hepsi DUSUK, hareket yok
    python3 deploy/motor_test.py single [hiz]    A ve B motorlarini tek tek iki yonde 1 sn
    python3 deploy/motor_test.py drive  [hiz]    ileri, geri, sola, saga 1'er sn
    python3 deploy/motor_test.py ramp   [A|B]    kalkis esigi: %60/80/100, 1 kHz ve 100 Hz, 0.7 sn

Pinler (pin plani): A: IN1 GPIO23, IN2 GPIO24, ENA GPIO12 | B: IN3 GPIO27, IN4 GPIO22, ENB GPIO13
Her durumda (hata, Ctrl+C) cikista tum pinler DUSUK: motorlar durur.
"""
import signal
import sys
import time

import lgpio

A = {"in1": 23, "in2": 24, "en": 12, "invert": False}   # sol palet
B = {"in1": 27, "in2": 22, "en": 13, "invert": True}    # sag palet: kablo ters, yazilimla duzeltildi
PWM_HZ = 1000
PULSE = 1.0
PAUSE = 2.5

h = lgpio.gpiochip_open(0)
for m in (A, B):
    for key in ("in1", "in2", "en"):
        lgpio.gpio_claim_output(h, m[key], 0)


def run(m: dict, direction: int, speed: int, hz: int = PWM_HZ, cap: int = 70) -> None:
    """direction: 1 / -1 / 0 (dur). speed: % gorev orani (cap ile sinirli)."""
    if direction == 0 or speed <= 0:
        # tx_pwm(..., 0) arka planda calismaya devam edip sonraki gpio_write'i eziyordu:
        # PWM'i tamamen birak, pini yeniden DUSUK cikis olarak al
        lgpio.gpio_free(h, m["en"])
        lgpio.gpio_claim_output(h, m["en"], 0)
        lgpio.gpio_write(h, m["in1"], 0)
        lgpio.gpio_write(h, m["in2"], 0)
        return
    if m["invert"]:
        direction = -direction
    lgpio.gpio_write(h, m["in1"], 1 if direction > 0 else 0)
    lgpio.gpio_write(h, m["in2"], 0 if direction > 0 else 1)
    lgpio.tx_pwm(h, m["en"], hz, min(speed, cap))


def stop_all() -> None:
    for m in (A, B):
        run(m, 0, 0)


def step(label: str, a: int, b: int, speed: int) -> None:
    print(f"{time.strftime('%H:%M:%S')}  {label}", flush=True)
    run(A, a, speed)
    run(B, b, speed)
    time.sleep(PULSE)
    stop_all()
    time.sleep(PAUSE)


def finish(*_):
    stop_all()
    lgpio.gpiochip_close(h)
    print("motorlar durduruldu, pinler serbest", flush=True)
    sys.exit(0)


signal.signal(signal.SIGINT, finish)
signal.signal(signal.SIGTERM, finish)
mode = sys.argv[1] if len(sys.argv) > 1 else "idle"
speed = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 45
try:
    if mode == "idle":
        stop_all()
        print("pinler sahiplenildi, hepsi DUSUK; hareket yok")
    elif mode == "single":
        print(f"hiz %{speed}, her adim {PULSE} sn")
        time.sleep(2)
        step("1) MOTOR A, yon +", 1, 0, speed)
        step("2) MOTOR A, yon -", -1, 0, speed)
        step("3) MOTOR B, yon +", 0, 1, speed)
        step("4) MOTOR B, yon -", 0, -1, speed)
    elif mode == "full":
        # PWM yok: EN sabit 1, IN1=1 IN2=0; once A sonra B, 1'er sn
        time.sleep(2)
        for name, m in (("A", A), ("B", B)):
            print(f"{time.strftime('%H:%M:%S')}  MOTOR {name}: EN=1 (sabit), IN1=1, IN2=0", flush=True)
            lgpio.gpio_write(h, m["in1"], 1)
            lgpio.gpio_write(h, m["in2"], 0)
            lgpio.gpio_write(h, m["en"], 1)
            time.sleep(PULSE)
            stop_all()
            time.sleep(PAUSE)
    elif mode == "verify":
        # EN=1 sabit, her motor her yonde 3 sn; surerken pinlerin gercek seviyesini oku
        import subprocess
        time.sleep(2)
        for name, m in (("A", A), ("B", B)):
            for d, label in ((1, "+"), (-1, "-")):
                lgpio.gpio_write(h, m["in1"], 1 if d > 0 else 0)
                lgpio.gpio_write(h, m["in2"], 0 if d > 0 else 1)
                lgpio.gpio_write(h, m["en"], 1)
                time.sleep(1.0)
                pins = ",".join(str(m[k]) for k in ("in1", "in2", "en"))
                state = subprocess.run(["pinctrl", "get", pins], capture_output=True, text=True).stdout
                levels = " ".join(f"GPIO{l.split(':')[0].strip()}={l.split('|')[1].split()[0]}"
                                  for l in state.splitlines())
                print(f"{time.strftime('%H:%M:%S')}  MOTOR {name} yon {label}, EN=1: {levels}", flush=True)
                time.sleep(2.0)
                stop_all()
                time.sleep(PAUSE)
    elif mode == "ground":
        # Yerde: 150 ms %100 kalkis darbesi, sonra hedef hiz; 200 Hz PWM; 1'er sn
        hz = 200
        time.sleep(3)
        for label, a, b, duty in (("ILERI %45", 1, 1, 45), ("GERI %45", -1, -1, 45),
                                  ("ILERI %60", 1, 1, 60), ("GERI %60", -1, -1, 60),
                                  ("SOLA DON %60", -1, 1, 60), ("SAGA DON %60", 1, -1, 60),
                                  ("SOLA DON %80", -1, 1, 80), ("SAGA DON %80", 1, -1, 80)):
            print(f"{time.strftime('%H:%M:%S')}  {label}", flush=True)
            run(A, a, 100, hz, cap=100)
            run(B, b, 100, hz, cap=100)
            time.sleep(0.15)
            run(A, a, duty, hz, cap=100)
            run(B, b, duty, hz, cap=100)
            time.sleep(0.85)
            stop_all()
            time.sleep(2.5)
    elif mode == "turns":
        # Yalniz donusler, 2'ser sn, arada 3 sn
        time.sleep(1)
        for label, a, b in (("1) SOLA DON: sol geri, sag ileri", -1, 1),
                            ("2) SAGA DON: sol ileri, sag geri", 1, -1)):
            print(f"{time.strftime('%H:%M:%S')}  {label}", flush=True)
            run(A, a, speed)
            run(B, b, speed)
            time.sleep(2)
            stop_all()
            time.sleep(3)
    elif mode == "ident":
        # Yon/palet eslestirme: once yalniz A +, sonra yalniz B +, 3'er sn, arada 5 sn
        time.sleep(1)
        for i, (name, m) in enumerate((("A", A), ("B", B)), 1):
            print(f"{time.strftime('%H:%M:%S')}  TEST {i}: sadece MOTOR {name}, yon +, %{speed}, 3 sn", flush=True)
            run(m, 1, speed)
            time.sleep(3)
            stop_all()
            time.sleep(5 if i == 1 else 0)
    elif mode == "turn":
        # 3 sn saga don (sol A ileri, sag B geri); ortada pin seviyelerini oku
        import subprocess
        print(f"hiz %{speed}, 3 sn SAGA DON: beklenen sol(A) ileri, sag(B) geri", flush=True)
        time.sleep(1)
        run(A, 1, speed)
        run(B, -1, speed)
        time.sleep(1.5)
        state = subprocess.run(["pinctrl", "get", "12,13,22,23,24,27"], capture_output=True, text=True).stdout
        print("  " + " ".join(f"GPIO{l.split(':')[0].strip()}={l.split('|')[1].split()[0]}"
                              for l in state.splitlines()), flush=True)
        print("  beklenen: A IN1(23)=hi IN2(24)=lo, B IN3(27)=lo IN4(22)=hi, EN'ler PWM", flush=True)
        time.sleep(1.5)
        stop_all()
    elif mode == "ramp":
        m = B if (sys.argv[2:3] or ["A"])[0].upper() == "B" else A
        name = "B" if m is B else "A"
        time.sleep(2)
        for hz in (1000, 100):
            for duty in (60, 80, 100):
                print(f"{time.strftime('%H:%M:%S')}  MOTOR {name} yon +  %{duty}  {hz} Hz", flush=True)
                run(m, 1, duty, hz, cap=100)
                time.sleep(0.7)
                stop_all()
                time.sleep(2.0)
    elif mode == "drive":
        print(f"hiz %{speed}, her adim {PULSE} sn")
        time.sleep(2)
        step("1) ILERI", 1, 1, speed)
        step("2) GERI", -1, -1, speed)
        step("3) SOLA DON (yerinde)", -1, 1, speed)
        step("4) SAGA DON (yerinde)", 1, -1, speed)
finally:
    finish()
