from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from openai import OpenAI
client = OpenAI()

def get_embeddings(titles, model="text-embedding-3-small"):
    try:
        response = client.embeddings.create(
            input=titles,
            model=model,
            dimensions=25
        )
        embeddings = [item.embedding for item in response.data]
    except Exception as e:
        print(f"An error occurred: {e}")
        embeddings = [None] * len(titles)
    return embeddings

def get_all_embeddings(titles):
    num_batches = np.ceil(len(titles) / 1000).astype(int)
    all_embeddings = []
    for i in range(num_batches):
        batch_start = i * 1000
        batch_end = min((i + 1) * 1000, len(titles))
        batch_titles = titles[batch_start:batch_end]
        batch_embeddings = get_embeddings(batch_titles)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings
