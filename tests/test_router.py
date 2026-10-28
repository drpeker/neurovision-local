"""Router check: python -m tests.test_router (from the project folder)."""

from nv.router import route

CASES = [
    ("Stopp!", "stop", {}), ("Stop.", "stop", {}), ("Bitte anhalten.", "stop", {}), ("Bleib stehen!", "stop", {}),
    ("Hör auf, mir zu folgen.", "stop", {}), ("Warte.", "stop", {}), ("Top!", "stop", {}), ("Halt!", "stop", {}),
    ("Top Ten Filme?", "chat", {}),
    ("Fahr vorwärts.", "move", {"direction": "ileri", "seconds": 1.0}),
    ("Fahr ein bisschen nach vorne.", "move", {"direction": "ileri", "seconds": 0.3}),
    ("Fahr langsam vorwärts.", "move", {"direction": "ileri", "speed": "yavaş"}),
    ("Fahr langsam zurück.", "move", {"direction": "geri", "speed": "yavaş"}),
    ("Setz ein Stück zurück.", "move", {"direction": "geri", "seconds": 0.3}),
    ("Dreh dich nach links.", "move", {"direction": "sol"}), ("Kannst du bitte nach rechts fahren?", "move", {"direction": "sag"}),
    ("Dreh dich ein bisschen nach links.", "move", {"direction": "sol", "seconds": 0.3}),
    ("Dreh dich zwei Sekunden nach rechts.", "move", {"direction": "sag", "seconds": 2}),
    ("Fahr ganz langsam 1,5 Sekunden vorwärts.", "move", {"direction": "ileri", "seconds": 1.5, "speed": "çok yavaş"}),
    ("Nach links!", "move", {"direction": "sol"}), ("Schau nach rechts.", "move", {"direction": "sag"}),
    ("Wie komme ich an der nächsten Straße nach links?", "chat", {}), ("Was ist richtig und falsch?", "chat", {}),
    ("Ich gehe jetzt nach rechts.", "chat", {}),
    # Heard on the robot (Sie form, lost verbs, English)
    ("Kommen Sie zu mir.", "move", {"direction": "ileri"}), ("Kannst du steuern sie vorne bitte?", "move", {"direction": "ileri"}),
    ("Far forwards.", "move", {"direction": "ileri"}), ("Rehen Sie nach links wieder.", "move", {"direction": "sol"}),
    ("Richtig nach links.", "move", {"direction": "sol"}), ("Drehen Sie sich nach rechts.", "move", {"direction": "sag"}),
    ("Folgen Sie mir.", "track", {"target": "person"}), ("Halten Sie an.", "stop", {}),
    ("Schau dich um und sagen sie, was.", "look", {}), ("Wie heißen Sie?", "chat", {}),
    ("Folge mir.", "track", {"target": "person"}), ("Verfolge die rote Tasse.", "track", {"target": "cup", "color": "kırmızı"}),
    ("Folge dem blauen Ding.", "track", {"target": "colored object", "color": "mavi"}),
    ("Beobachte die Katze.", "track", {"target": "cat"}), ("Schau mich an.", "track", {"target": "person"}),
    ("Kannst du mir folgen?", "track", {"target": "person"}),
    ("Dreh dich zur Flasche.", "turn_to", {"target": "bottle"}), ("Schau auf den Laptop.", "turn_to", {"target": "laptop"}),
    ("Was siehst du?", "look", {}), ("Was kannst du gerade sehen?", "look", {}),
    ("Sag mir, was du siehst.", "look", {}), ("Schau dich um.", "look", {}), ("Was ist vor dir?", "look", {}),
    ("Komm zurück zu mir.", "move", {"direction": "ileri"}), ("Komm her.", "move", {"direction": "ileri"}),
    ("Siehst du ein Handy?", "look", {"find": "cell phone"}), ("Siehst du die Orange?", "look", {"find": "orange"}),
    ("Lauter bitte.", "volume", {"action": "artır"}), ("Stell die Lautstärke auf 40.", "volume", {"action": "ayarla", "level": 40}),
    ("Stell die Lautstärke auf sechzig.", "volume", {"action": "ayarla", "level": 60}),
    ("Leiser bitte.", "volume", {"action": "azalt"}), ("Wie laut bist du?", "volume", {"action": "sor"}),
    ("Setz die Motorgeschwindigkeit auf 60.", "motor_speed", {"level": 60}), ("Fahr ab jetzt langsamer.", "motor_speed", {"action": "azalt"}),
    ("Wie warm bist du?", "body_status", {"section": "isi"}), ("Wie hoch ist deine CPU-Auslastung?", "body_status", {"section": "cpu"}),
    ("Wie geht es dir?", "body_status", {"section": "ozet"}), ("Wie ist dein WLAN-Signal?", "body_status", {"section": "ag"}),
    ("Lächle!", "express", {"emotion": "mutlu"}), ("Schau überrascht.", "express", {"emotion": "şaşkın"}),
    ("Sei traurig.", "express", {"emotion": "üzgün"}),
    ("Merk dir, dass meine Schlüssel in der blauen Schale sind.", "remember",
     {"text": "Der Nutzer sagte: „Dass meine Schlüssel in der blauen Schale sind.“"}),
    ("Meine Tochter heißt Elif.", "remember", {"text": "Der Nutzer sagte: „Meine Tochter heißt Elif.“"}),
    ("Ich wohne in München.", "remember", {"text": "Der Nutzer sagte: „Ich wohne in München.“"}),
    ("Ich heiße Mustafa.", "remember", {}),
    ("Vergiss meine Schlüssel.", "forget", {"query": "meine schlüssel"}),
    ("Wo sind meine Schlüssel?", "recall", {"query": "schlüssel"}), ("Wie heißt meine Tochter?", "recall", {}),
    ("Was weißt du über Elif?", "recall", {"query": "elif"}), ("Wer bin ich?", "recall", {}),
    ("Wie ist das Wetter?", "weather", {"place": None}), ("Regnet es morgen in Hamburg?", "weather", {"days": 2, "place": "hamburg"}),
    ("Wie wird das Wetter morgen in Berlin?", "weather", {"place": "berlin", "days": 2}),
    ("Wie spät ist es?", "time", {}), ("Wie viel Uhr haben wir?", "time", {}),
    ("Wer war Albert Einstein?", "wiki", {"query": "albert einstein"}),
    ("Schlag schwarze Löcher nach.", "wiki", {"query": "schwarze löcher"}),
    ("Fahr das System herunter.", "shutdown", {"target": "sistem"}), ("Schalte dich aus.", "shutdown", {"target": "robot"}),
    ("Mach weiter so.", "chat", {}), ("Ich habe einen Hund namens Max.", "chat", {}),
    ("Merk dir, dass ich einen Hund namens Max habe.", "remember", {"text": "Der Nutzer sagte: „Dass ich einen Hund namens Max habe.“"}),
    ("Was ist die Hauptstadt von Frankreich?", "chat", {}), ("Erzähl mir einen Witz.", "chat", {}),
    ("Warum ist der Himmel blau?", "chat", {}), ("Wie heißt du?", "chat", {}), ("Wer bist du?", "chat", {}),
    ("Schalte das Licht aus.", "chat", {}), ("Mach das Licht an.", "chat", {}),
    ("Ich bin heute glücklich.", "chat", {}), ("Die richtige Antwort ist vier.", "chat", {}),
    ("Was ist zwei plus drei?", "chat", {}), ("Danke, NeuroVision.", "chat", {}),
]

bad = 0
for text, tool, expect in CASES:
    got_tool, args = route(text)
    ok = got_tool == tool and all(args.get(k) == v for k, v in expect.items())
    bad += not ok
    if not ok:
        print(f"BAD {text!r:48} -> {got_tool} {args}   (want {tool} {expect})")
print(f"{len(CASES) - bad}/{len(CASES)} correct")
