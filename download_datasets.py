import json
import pandas as pd
corpus = []

with open("data/corpus.jsonl", "r") as f:
    for line in f:
        corpus.append(json.loads(line))

c_df = pd.DataFrame(corpus)

# print("Total papers:", len(c_df))
# print(c_df.head())

c_df['abs_text'] = c_df['abstract'].apply(lambda x: " ".join(x))
c_df["text"] = c_df["title"] + " " + c_df["abs_text"]
print(c_df['text'])