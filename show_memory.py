"""Look inside long-term memory: what's stored, and what a new question would recall."""
import sys

from agent.memory import Memory

mem = Memory()
print(f"{mem.count()} notes in memory\n")
for question, n in mem.db.execute("SELECT question, COUNT(*) FROM notes GROUP BY question"):
    print(f"{n:3} notes from: {question}")
if len(sys.argv) > 1:
    query = " ".join(sys.argv[1:])
    print(f"\nrecall for: {query}")
    for note in mem.recall(query):
        print(f" - {note['claim'][:100]}\n   {note['url']}")
