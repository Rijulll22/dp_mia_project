import sqlite3
import json

def fetch_all():
    conn = sqlite3.connect('backend/privacyguard.db')
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT * FROM experiments")
    rows = cur.fetchall()
    
    data = []
    for r in rows:
        data.append(dict(r))
        
    with open('db_dump.json', 'w') as f:
        json.dump(data, f, indent=2)

if __name__ == '__main__':
    fetch_all()
