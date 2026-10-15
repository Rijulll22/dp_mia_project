from sqlalchemy import text
from backend.database import engine

with engine.connect() as conn:
    result = conn.execute(text("PRAGMA table_info('experiments');"))
    for row in result:
        print(row)
