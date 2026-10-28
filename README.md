# NeuroVision (Gemini Live)

Paletli ev robotu için kompakt sesli ve görsel asistan. Pi 4'te geliştirildi, Pi Zero 2W hedefleniyor.

- **Ses:** INMP441 I²S mikrofon (16 kHz) → Gemini Live → 3,5 mm jak (24 kHz)
- **Görme:** Pi kamera → `look` aracı ile modele tek kare
- **Uyku modu yok:** yerel VAD sadece konuşmayı gönderir, sessizlikte internete ses gitmez
- **Yarı çift yönlü:** robot konuşurken mikrofon kapalı (AEC gelene kadar)

Kaynak kullanımı (Pi 4): ~%5 CPU, ~90 MB RAM.

## Hafıza
Hepsi Pi'de, `data/memory.db` (SQLite) içinde. Git dışında tutulur, fotoğraflar saklanmaz.

| Katman | İçerik | Nasıl |
|---|---|---|
| Yakın | Şu anki sohbet | Gemini bağlamı + oturum belirteci diskte (yeniden başlayınca 2 saate kadar kaldığı yerden devam eder) |
| Orta | Son sohbetler | Döküm kaydedilir; 2 dk sessizlikten sonra `gemini-3.5-flash-lite` özetler |
| Uzun | Kalıcı bilgiler | `remember`/`recall`/`forget` araçları + özetleyicinin döküm üzerinden otomatik çıkardığı bilgiler |

Yeni oturum açılırken en önemli bilgiler ve son özetler sistem talimatına eklenir. Hafızayı görmek için:
```bash
sqlite3 data/memory.db "select category, text from facts order by importance desc"
sqlite3 data/memory.db "select datetime(ts_start,'unixepoch','localtime'), summary from episodes"
```

## Kurulum
```bash
cd ~/projects/neurovision-updated
uv venv --python /usr/bin/python3 .venv
uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env && chmod 600 .env && nano .env   # GEMINI_API_KEY
```
ALSA cihazları (`mic`, `speaker`) `~/.asoundrc` dosyasında tanımlı. Mikrofon için `config.txt` içinde `dtoverlay=adau7002-simple,card-name=inmp441` gerekir.

## Çalıştırma
```bash
.venv/bin/python robot.py --check   # donanım testi: mikrofon, hoparlör, kamera
.venv/bin/python robot.py           # elle çalıştır (Ctrl+C ile durur)
```

## Servis (açılışta otomatik başlar)
```bash
bash deploy/install-service.sh
systemctl --user status neurovision     # durum
systemctl --user restart neurovision    # yeniden başlat
systemctl --user stop neurovision       # durdur
tail -f logs/robot.log                  # canlı log
```

## Yapı
| Dosya | Görevi |
|---|---|
| `robot.py` | Başlatma, log, `--check` |
| `config.toml` | Model, ses, kişilik, VAD, kamera ayarları |
| `nv/audio.py` | Mikrofon, hoparlör (ses düzeyi, radyo kanalı karıştırma), mikrofon kazancı |
| `nv/vad.py` | Yerel konuşma kapısı |
| `nv/camera.py` | `rpicam-vid` MJPEG → son kare |
| `nv/live.py` | Gemini Live oturumu, yeniden bağlanma |
| `nv/tools.py` | Modelin çağırdığı araçlar (`look`, `remember`, `recall`, `forget`, `body_status`, `move`, `stop`, `express`, `weather`, `wiki`, `volume`) |
| `nv/body.py` | Öz farkındalık (`body_status`): güç, ısı, CPU, RAM, disk, wifi, GPIO, I²C, ışık, ses, alt sistemler, Gemini token kullanımı (`kota`). Yalnızca okur |
| `nv/display.py` | Robotun yüzü: ST7789 ekranda gözler, durum (dinliyor/düşünüyor/konuşuyor/bakıyor), duygular (`express`), altta altyazı, arka ışık |
| `nv/motors.py` | Paletler (L298N): `move`/`stop`, yumuşak kalkış/duruş, 3 sn sınır, bağlantı kopunca durma, hareket sırasında mikrofon susturma |
| `nv/vision.py` | Yerel nesne tanıma: YOLO11n 320 (COCO 80 nesne) ve renk bölgesi bulma. Model: `deploy/get-model.sh` |
| `nv/tracker.py` | Nesne takibi: yalnızca yerinde dönerek ortalar; yerde ölçülmüş dönüş tablosu, takılma algılama |
| `nv/web.py` | Hava durumu (Open-Meteo) ve Vikipedi araması; anahtarsız |
| `nv/radio.py` | İnternet radyosu (ffmpeg). Şimdilik kapalı: `[radio] enabled = false` |
| `nv/debug.py` | Teşhis: son 10 konuşmanın sesi ve oturum talimatı (`logs/debug/`) |
| `nv/memory.py` | SQLite hafıza: döküm, özetler, kalıcı bilgiler |
| `nv/summarizer.py` | Sessizlikte özet + otomatik bilgi çıkarma |

Yeni araç eklemek için `nv/tools.py` içindeki `DECLARATIONS` listesine bir tanım, `handle()` fonksiyonuna da bir dal ekle.
