"""Sohbet bitince dokumu ozetler ve kalici bilgileri cikarir (orta + uzun hafiza)."""

import json
import logging
import time

from google import genai
from google.genai import types

from .memory import CATEGORIES, Memory

log = logging.getLogger(__name__)

PROMPT = """Aşağıda bir ev robotu ile kullanıcının konuşma dökümü var.
Döküm otomatik ses tanımadan geliyor; anlamsız, yabancı dilde ya da bağlamsız parçalar
gürültüdür, onları yok say. Robotun konuyla ilgisiz ya da kalıp cevaplarını özete taşıma.
1) "summary": Sohbetin 1-3 cümlelik Türkçe özeti (kim ne istedi, ne konuşuldu, yarım kalan işler).
2) "facts": Kullanıcı veya ev hakkında İLERİDE İŞE YARAYACAK kalıcı bilgiler (isim, tercih, aile,
   eşyaların yeri, alışkanlıklar). Geçici şeyleri (hava, o anki ruh hali, selamlaşma) ekleme.
   Aşağıdaki "bilinenler" listesinde zaten olan bilgiyi tekrar ekleme.
   Sağlık, şifre, banka gibi hassas bilgileri ekleme.
   Her bilgi: {{"text": kısa Türkçe cümle, "category": {cats}, "importance": 1-5}}.

Bilinenler:
{known}

Döküm:
{transcript}
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "facts": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "category": {"type": "string", "enum": CATEGORIES},
                "importance": {"type": "integer"},
            },
            "required": ["text", "category", "importance"],
        }},
    },
    "required": ["summary", "facts"],
}


async def consolidate(memory: Memory, client: genai.Client, model: str) -> bool:
    turns = memory.pending_turns()
    if not turns:
        return False
    transcript = "\n".join(
        f"[{time.strftime('%H:%M', time.localtime(ts))}] {'Kullanıcı' if role == 'user' else 'Robot'}: {text}"
        for _, ts, role, text in turns)
    known = "\n".join(f"- {t}" for _, _, t in memory.top_facts(60)) or "(henüz yok)"
    resp = await client.aio.models.generate_content(
        model=model,
        contents=PROMPT.format(cats="|".join(CATEGORIES), known=known, transcript=transcript),
        config=types.GenerateContentConfig(
            response_mime_type="application/json", response_json_schema=SCHEMA, temperature=0.2),
    )
    memory.add_usage("ozet", resp.usage_metadata)
    data = json.loads(resp.text)
    memory.add_episode(data["summary"], turns)
    added = [f["text"] for f in data.get("facts", [])
             if memory.remember(f["text"], f["category"], f["importance"], source="auto")]
    log.info("hafiza: %d satir ozetlendi, %d yeni bilgi %s", len(turns), len(added), added or "")
    return True
