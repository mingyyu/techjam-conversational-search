"""Train a corpus-local semantic layer and export a compact neighbour table.

Latent Semantic Analysis over the frozen catalog. Terms that occur in similar
product contexts land near each other, which is exactly the synonymy that BM25
cannot see ("waterproof" / "water-resistant", "tee" / "t-shirt").

Output is a term -> neighbours table, not dense document vectors: it is two
orders of magnitude smaller, needs no matrix multiply at query time, and plugs
into the existing BM25 path as query expansion rather than replacing ranking.
"""
import json, time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from src.catalog import searchable_text, tokenize

t0 = time.time()
docs = []
with open('data/catalog.jsonl', encoding='utf-8') as fh:
    for line in fh:
        if line.strip():
            docs.append(" ".join(tokenize(searchable_text(json.loads(line)))))
print(f"docs={len(docs)} load={time.time()-t0:.1f}s", flush=True)

vec = TfidfVectorizer(min_df=8, max_df=0.4, sublinear_tf=True, token_pattern=r"\S+")
X = vec.fit_transform(docs)
vocab = np.array(vec.get_feature_names_out())
print(f"vocab={len(vocab)} nnz={X.nnz} tfidf={time.time()-t0:.1f}s", flush=True)

svd = TruncatedSVD(n_components=192, random_state=0, algorithm="randomized")
svd.fit(X)
# Term representation: component loadings scaled by singular values.
terms = (svd.components_.T * svd.singular_values_).astype(np.float32)
terms /= (np.linalg.norm(terms, axis=1, keepdims=True) + 1e-9)
print(f"svd done var={svd.explained_variance_ratio_.sum():.3f} t={time.time()-t0:.1f}s", flush=True)

TOP = 6
MIN_SIM = 0.55
table = {}
B = 2000
for start in range(0, len(vocab), B):
    block = terms[start:start+B]
    sims = block @ terms.T
    for i in range(block.shape[0]):
        row = sims[i]
        row[start + i] = -1.0
        idx = np.argpartition(-row, TOP)[:TOP]
        idx = idx[np.argsort(-row[idx])]
        pairs = [(str(vocab[j]), round(float(row[j]), 4)) for j in idx if row[j] >= MIN_SIM]
        if pairs:
            table[str(vocab[start + i])] = pairs
print(f"terms_with_neighbours={len(table)} t={time.time()-t0:.1f}s", flush=True)
json.dump(table, open('src/semantic_neighbours.json', 'w'))
import os; print(f"artifact={os.path.getsize('src/semantic_neighbours.json')/1e6:.1f} MB")
