"""Long-term memory: every note the agent has ever taken, in SQLite, searchable by meaning."""
import sqlite3
from datetime import datetime, timezone

import numpy as np

from agent.config import settings
from agent.embeddings import embed
from agent.state import Note

SCHEMA = """CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY, question TEXT, claim TEXT, quote TEXT,
    url TEXT, title TEXT, created_at TEXT, embedding BLOB, UNIQUE(claim, url))"""


class Memory:
    def __init__(self, path: str = settings.memory_db) -> None:
        self.db = sqlite3.connect(path)
        self.db.execute(SCHEMA)

    def save(self, question: str, notes: list[Note]) -> int:
        vecs = embed([n["claim"] for n in notes]) if notes else []
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        cur = self.db.executemany(
            "INSERT OR IGNORE INTO notes (question, claim, quote, url, title, created_at, embedding) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(question, n["claim"], n["quote"], n["url"], n["title"], now, v.astype(np.float32).tobytes())
             for n, v in zip(notes, vecs)])
        self.db.commit()
        return cur.rowcount

    def recall(self, query: str, k: int = 8) -> list[Note]:
        """Past notes whose claim is close in meaning to the query."""
        rows = self.db.execute("SELECT claim, quote, url, title, embedding FROM notes").fetchall()
        if not rows:
            return []
        vecs = np.stack([np.frombuffer(r[4], dtype=np.float32) for r in rows])
        scores = vecs @ embed([query])[0]
        best = [i for i in scores.argsort()[::-1][:k] if scores[i] >= settings.recall_threshold]
        return [Note(claim=rows[i][0], quote=rows[i][1], url=rows[i][2], title=rows[i][3], from_memory=True)
                for i in best]

    def count(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM notes").fetchone()[0]
