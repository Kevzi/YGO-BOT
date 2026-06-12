import sqlite3
import os

try:
    conn = sqlite3.connect('data/ygo.db')
    conn.execute("INSERT INTO alembic_version (version_num) VALUES ('3a99862fb5bf')")
    conn.commit()
    print("Inserted successfully")
except Exception as e:
    print(e)
