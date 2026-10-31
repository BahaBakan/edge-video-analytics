import sqlite3
from datetime import datetime
import config

class Storage:
    def __init__(self):
        self.conn = sqlite3.connect(config.DB_PATH, check_same_thread= False)
        self.cur = self.conn.cursor()
        self._create_table()

    def _create_table(self):
        self.cur.execute('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                nesne_id INTEGER,
                olay TEXT,
                timestamp TEXT
            )
        ''')
        self.conn.commit()

    def kaydet(self, nesne_id, olay):
        self.cur.execute(
            "INSERT INTO events (nesne_id, olay, timestamp) VALUES (?, ?, ?)",
            (nesne_id, olay, datetime.now().isoformat())
        )
        self.conn.commit()

    def tum_olaylar(self):
        self.cur.execute("SELECT * FROM events")
        return self.cur.fetchall()

    def son_olaylar(self, limit=10):
        self.cur.execute("SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        return self.cur.fetchall()

    def saat_araligi(self, baslangic, bitis):
        self.cur.execute(
            "SELECT * FROM events WHERE timestamp BETWEEN ? AND ? ORDER BY timestamp ASC",(baslangic, bitis)
        )
        return self.cur.fetchall()

    def saatlik_dagilim(self):
        self.cur.execute(
            "SELECT strftime('%Y-%m-%d %H', timestamp) AS saat, COUNT(*) FROM events GROUP BY saat ORDER BY saat ASC"
        )
        return self.cur.fetchall()
    
    def kapat(self):
        self.conn.close()




    
