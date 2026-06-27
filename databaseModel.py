import sqlite3
from datetime import datetime, UTC

conn=sqlite3.connect("database/memory.db")

cursor=conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memory (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    people TEXT,
    sentence TEXT,
    embedding TEXT,
    date DATETIME
)
""")

def add(people,sentence,embedding):
    try:
        cursor.execute(
            """
            INSERT INTO memory(people,sentence,embedding,date)
            VALUES (?,?,?,?)
            """,
            (people,sentence,embedding,datetime.now(UTC))
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

for i in retrive("Utkarsha"):
    print(i[3])
    print()