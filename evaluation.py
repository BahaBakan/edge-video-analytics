import llm
from storage import Storage

TESTLER = [
        {
        "soru" : "Toplam kaç olay var?",
        "referans" : "SELECT COUNT(*) FROM events",
    },

        {
        "soru" : "Kaç farklı nesne bölgeye girdi?",
        "referans" : "SELECT COUNT(DISTINCT nesne_id) FROM events",
    },    

        {
        "soru" : "En son olay ne zaman gerçekleşti?",
        "referans" : "SELECT MAX(timestamp) FROM events",
    },

        {
        "soru" : "İlk olay ne zaman gerçekleşti?",
        "referans" : "SELECT MIN(timestamp) FROM events",
    },

        {
        "soru" : "7 numaralı nesnenin kaç kaydı var?",
        "referans" : "SELECT COUNT(*) FROM events WHERE nesne_id = 7",
    },

        {
        "soru" : "'Bölgeye giriş' olayından kaç tane var?",
        "referans" : "SELECT COUNT(*) FROM events WHERE olay = 'Bölgeye giriş'",
    },

        {"soru": "Son 5 olayın tüm sütunlarını listele.",
        "referans": "SELECT * FROM events ORDER BY id DESC LIMIT 5"
    },

        {
        "soru" : "Saat 21'de kaç olay oldu?",
        "referans" : "SELECT COUNT(*) FROM events WHERE strftime('%H', timestamp) = '21'",
    },

        {
        "soru" : "Saatlik dağılım nedir?",
        "referans" : "SELECT strftime('%Y-%m-%d %H', timestamp) AS saat, COUNT(*) FROM events GROUP BY saat ORDER BY saat",
    },

        {"soru": "Hangi tarih ve saatte en çok olay oldu? 'YYYY-MM-DD HH' biçiminde ver.",
        "referans": "SELECT strftime('%Y-%m-%d %H', timestamp) AS saat FROM events GROUP BY saat ORDER BY COUNT(*) DESC LIMIT 1"
    },

       
        {
        "soru" : "nesne_id'si  20'den büyük kaç kayıt var?",
        "referans" : "SELECT COUNT(*) FROM events WHERE nesne_id > 20",
    },

        {
        "soru" : "Bugün kaç olay oldu?",
        "referans" : "SELECT COUNT(*) FROM events WHERE date(timestamp) = date('now')",
    },

        {
        "soru" : "En büyük nesne_id kaç?",
        "referans" : "SELECT MAX(nesne_id) FROM events",
    },

        {
        "soru" : "Olayların tüm sütunlarını nesne_id'ye göre sıralı listele.",
        "referans" : "SELECT * FROM events ORDER BY nesne_id ASC",
    },

        {
        "soru" : "10 ile 20 arasındaki nesne_id'lere ait kaç olay var?",
        "referans" : "SELECT COUNT(*) FROM events WHERE nesne_id BETWEEN 10 AND 20",
    },

        {"soru": "Son bir saatte kaç olay oldu?",
        "referans": "SELECT COUNT(*) FROM events WHERE timestamp >= datetime('now','-1 hour')"
    },

        {
        "soru" : "Ortalama olarak saatte kaç olay düşüyor?",
        "referans" : "SELECT COUNT(*) * 1.0 / COUNT(DISTINCT strftime('%Y-%m-%d %H', timestamp)) FROM events",
    },

        {
        "soru" : "Dün kaç olay oldu?",
        "referans" : "SELECT COUNT(*) FROM events WHERE date(timestamp) = date('now','-1 day')",
    },

        {
        "soru" : "En sık görülen olay tipi nedir?",
        "referans" : "SELECT olay FROM events GROUP BY olay ORDER BY COUNT(*) DESC LIMIT 1",
    },

        {
        "soru" : "Bölgeye giren ilk üç nesnenin id'si nedir?",
        "referans" : "SELECT nesne_id FROM events ORDER BY id ASC LIMIT 3",
    },


]

def calistir(storage, sql):
    try:
        storage.cur.execute(sql)
        return storage.cur.fetchall()
    except Exception as e:
        return f"HATA: {e}"


def main():
    storage = Storage()
    dogru_sayisi = 0

    for i,test in enumerate(TESTLER,1):
        uretilen = llm.sql_uret(test["soru"])
        sonuc_model = calistir(storage, uretilen)
        sonuc_referans = calistir(storage, test["referans"])

        gecti = sonuc_model == sonuc_referans

        if gecti:
            dogru_sayisi += 1


        durum = "OK" if gecti else "FAIL"
        print(f"{durum} {i}. {test['soru']}")
        if not gecti:
            print(f"    uretilen : {uretilen}")
            print(f"    sonuc    : {sonuc_model}")
            print(f"    beklenen : {sonuc_referans}")

    oran = dogru_sayisi / len(TESTLER) * 100
    print(f"\nDogruluk: {dogru_sayisi}/{len(TESTLER)} = %{oran:.0f}")        


if __name__ == "__main__":
    main()
    
