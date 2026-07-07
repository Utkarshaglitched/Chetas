import sqlite3
from datetime import datetime, UTC
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "database" / "memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("""
    CREATE TABLE IF NOT EXISTS memory (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        people TEXT,
        sentence TEXT,
        embedding TEXT,
        date DATETIME
    )
    """)

def add(people,sentence,embedding):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            INSERT INTO memory(people,sentence,embedding,date)
            VALUES (?,?,?,?)
            """,
            (people,sentence,str(embedding),datetime.now(UTC))
        )
        conn.commit()

        return True
    except Exception as e:
        print(e)
        return False
    
    finally:
        conn.close()


def update(id,sentance,emb):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        cursor.execute(
            """
            UPDATE memory SET 
            sentence=?,embedding=? 
            WHERE id=?
            """,(sentance,str(emb),id)
        )
        conn.commit()
        return True

    except Exception as e:
        print(e)
        return False
    
    finally:
        conn.close()

def retrive(ppl):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT * FROM memory WHERE people=?
        """,(ppl,)
    )

    result=cursor.fetchall()
    
    conn.close()
    return list(result)


