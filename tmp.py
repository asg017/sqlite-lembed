import sqlite3
import time
from sentence_transformers import SentenceTransformer

db = sqlite3.connect("bench/headlines-2024.db")

sentences = [
    row[0] for row in db.execute("select headline from articles limit 1000").fetchall()
]

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
t0 = time.time()
print(t0)
embeddings = model.encode(sentences)
print(time.time() - t0)
print(embeddings[0][0:8])
