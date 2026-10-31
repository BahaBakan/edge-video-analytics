import ollama
import config
from datetime import datetime

# RAG'ın en basit biçimi: Modele cevap vermeden önce bağlam (context) sağlamak. Bu bağlam, veritabanındaki olaylar olabilir. Modelin soruyu anlaması ve doğru SQL sorgusunu üretmesi için gerekli bilgileri içerir.
SEMA = """
Veritabani: SQLite
Tablo: events
Sütunlar:
  id         INTEGER  - otomatik artan birincil anahtar
  nesne_id   INTEGER  - takip edilen nesnenin kimliği
  olay       TEXT     - olay tipi, örn: 'Bölgeye giriş'
  timestamp  TEXT  - ISO 8601 format, e.g. 'YYYY-MM-DDTHH:MM:SS.ffffff'
"""



def sql_uret(soru):
    
    sistem_prompt = f"""You are a SQL expert. Convert the user's question into a single SQLite query.

{SEMA}

Current date and time: {datetime.now().isoformat()}

Rules:
- Output ONLY the SQL query, nothing else.
- No explanation, no markdown code fences.
- Use strftime() for date/time filtering.
- Never use INSERT, UPDATE, DELETE or DROP. Only SELECT.
- If the question does not mention a date, do NOT filter by date
- If the question does not mention a date or time, do NOT add any date/time filter.
- Return only the columns the question asks for.
- 'nesne_id' is the tracked object's identity. 'id' is just the row number. Use nesne_id when the question refers to objects.

Examples:

Question: Kaç farklı nesne var?
SQL: SELECT COUNT(DISTINCT nesne_id) FROM events

Question: Saatlik dağılım nedir?
SQL: SELECT strftime('%Y-%m-%d %H', timestamp) AS saat, COUNT(*) FROM events GROUP BY saat ORDER BY saat

Question: En çok olay hangi saatte oldu?
SQL: SELECT strftime('%Y-%m-%d %H', timestamp) AS saat FROM events GROUP BY saat ORDER BY COUNT(*) DESC LIMIT 1

Question: İlk üç nesnenin kimliği nedir?
SQL: SELECT nesne_id FROM events ORDER BY id ASC LIMIT 3

Question: Olaylarin tum sutunlarini tarihe gore listele.
SQL: SELECT * FROM events ORDER BY timestamp ASC

Question: Son giren nesnenin kimligi nedir?
SQL: SELECT nesne_id FROM events ORDER BY id DESC LIMIT 1
"""
    cevap = ollama.chat(
        model= config.LLM_MODEL,
        messages= [
            {"role": "system", "content": sistem_prompt},
            {"role": "user", "content": soru}
        ],
    )
    ham= cevap["message"]["content"]
    return temizle(ham)

def temizle(ham):
    metin = ham.strip()
    if metin.startswith("```"):
        # ilk satırı tamamen at (``` veya ```sql olabilir)
        satirlar = metin.split("\n")
        satirlar = satirlar[1:]
        # son satır ``` ise onu da at
        if satirlar and satirlar[-1].strip() == "```":
            satirlar = satirlar[:-1]
        metin = "\n".join(satirlar)
    return metin.strip()

if __name__ == "__main__":
    print(sql_uret("Saat 21'de kaç olay oldu?"))