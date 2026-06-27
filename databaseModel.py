import sqlite3
import time
from datetime import datetime, UTC

conn=sqlite3.connect("database/memory.db")

cursor=conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    people TEXT,
    embedding TEXT,
    date DATETIME
)
""")

def add(people,embedding):
    try:
        cursor.execute(
            """
            INSERT INTO memory(people,embedding,date)
            VALUES (?,?,?)
            """,
            (people,embedding,datetime.now(UTC))
        )
        conn.commit()

        return True
    except Exception as e:
        print(e)
        return False
    
def retrive(ppl):
    cursor.execute(
        """
        SELECT * FROM memory WHERE people=?
        """,(ppl,)
    )

    result=cursor.fetchall()
    return list(result)

