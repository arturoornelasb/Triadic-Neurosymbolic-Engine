import os
import pandas as pd
import random
import nltk
from nltk.corpus import wordnet as wn

nltk.download('wordnet')
nltk.download('omw-1.4')

# Extract nouns from WordNet
words = list(set([lemma.name().replace('_', ' ') for synset in wn.all_synsets('n') for lemma in synset.lemmas()]))

# Filter valid single words (no symbols)
words = [w for w in words if w.isalpha()]

# Sample 2000 random concepts
random.seed(42)
sampled_words = random.sample(words, min(2000, len(words)))

# Save to CSV
os.makedirs("examples/data", exist_ok=True)
df = pd.DataFrame({"concept": sampled_words})
output_path = "examples/data/wordnet_2k.csv"
df.to_csv(output_path, index=False)

print(f"✅ Generated Real Dataset: {output_path} with {len(sampled_words)} concepts from WordNet.")
