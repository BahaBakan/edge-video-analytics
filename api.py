from fastapi import FastAPI 
from storage import Storage
from pydantic import BaseModel
import llm

app = FastAPI(title= "Video Analytics API")
storage = Storage()

@app.get("/events")
def son_olaylar(limit: int = 10):
    return storage.son_olaylar(limit)

@app.get("/events/hourly")
def saatlik_dagilim():
    return storage.saatlik_dagilim()

@app.get("/stats")
def istatistik():
    return {
        "toplam_olay": len(storage.tum_olaylar())
    }

class Soru(BaseModel):
    metin: str  

@app.post("/ask")
def sor(soru: Soru):
    sql = llm.sql_uret(soru.metin)

    # Güvenlik: Sadece SELECT sorgularına izin ver
    if not sql.strip().upper().startswith("SELECT"):
        return {"error" : "Sadece okuma (SELECT) sorgularına izin verilir.", "uretilen_sql": sql}

    try:
        storage.cur.execute(sql)
        sonuc= storage.cur.fetchall()
    except Exception as e:
        return {"error": str(e), "uretilen_sql": sql}

    return {"soru": soru.metin, "uretilen_sql": sql, "sonuc": sonuc}