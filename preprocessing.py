# Updated installation: pip install pandas nltk emoji beautifulsoup4 stopwordsiso
# =========================
# 1. IMPORTS
# =========================
import pandas as pd
import re
import string
import nltk
import emoji
from bs4 import BeautifulSoup
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import SequenceMatcher
import stopwordsiso 

# =========================
# 2. LOAD DATASET
# =========================
df = pd.read_csv('data/panagbenga2013-2026_cleaned=9013.csv')
df = df[['id','text', 'source', 'timestamp', 'engagement']]
df['text'] = df['text'].astype(str)

print("Dataset loaded:", df.shape)

# =========================
# 3. LOAD NLP RESOURCES
# =========================
nltk.download('punkt_tab')
nltk.download('stopwords')

# English Stopwords
stop_words_english = set(stopwords.words('english'))

# Tagalog Stopwords (Fetching from stopwordsiso instead of calamancy)
tagalog_stopwords = stopwordsiso.stopwords("tl")

# Combine
all_stop_words = stop_words_english.union(tagalog_stopwords)

DOMAIN_STOPWORDS = {
    "panagbenga",
    "panagbengafestival",
    "panagbengafestival2024",
    "panagbenga2024"
}

print(f"Stopwords loaded. Total count: {len(all_stop_words)}")

# =========================
# 4. HELPER FUNCTIONS
# =========================
def is_noise_token(token):
    if len(token) > 20: return True
    if re.search(r'[^\x00-\x7F]', token): return True
    if len(re.findall(r'[aeiou]', token)) <= 1 and len(token) > 5: return True
    return False

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]

def clean_text(text):
    text = str(text).lower()
    emojis = extract_emojis(text)
    hashtags = re.findall(r"#\w+", text)
    hashtags = [tag.replace("#", "HASHTAG_") for tag in hashtags]

    text = re.sub(r'https\S+|@\w+', '', text)
    text = re.sub(r'(.)\1{4,}', r'\1\1\1', text)
    text = re.sub(r'\d+', '', text)
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text, emojis, hashtags

def is_similar(a, b, threshold=0.75):
    return SequenceMatcher(None, a, b).ratio() > threshold

def split_simple_compounds(token):
    splits = re.findall(r'[a-z]+', token)
    if len(token) > 15 and len(splits) > 1:
        return splits
    return [token]

def dynamic_token_cleanup(tokens, hashtags):
    hashtag_words = [tag.replace("HASHTAG_", "") for tag in hashtags]
    combined = tokens + hashtag_words
    cleaned = []

    for token in combined:
        if is_noise_token(token): continue
        duplicate = False
        for existing in cleaned:
            if is_similar(token, existing):
                duplicate = True
                break
        if not duplicate:
            cleaned.append(token)
    return cleaned

def clean_hashtags(hashtags):
    cleaned = []
    for tag in hashtags:
        base = tag.replace("HASHTAG_", "")
        if base not in DOMAIN_STOPWORDS:
            cleaned.append(tag)
    return cleaned

# =========================
# 5. EXECUTION PIPELINE
# =========================

# Step 5 & 6: Clean and Tokenize
cleaned = df['text'].apply(clean_text)
df['clean_text'] = cleaned.apply(lambda x: x[0])
df['emojis'] = cleaned.apply(lambda x: x[1])
df['hashtags'] = cleaned.apply(lambda x: x[2])
df['hashtags'] = df['hashtags'].apply(clean_hashtags)
df['tokenized'] = df['clean_text'].apply(word_tokenize)

# Step 7: Stopword Filtering
df['filtered'] = df['tokenized'].apply(
    lambda tokens: [
        word for word in tokens
        if word not in all_stop_words and word not in DOMAIN_STOPWORDS
    ]
)

# Step 8: Dynamic Normalization
df['normalized_tokens'] = df.apply(
    lambda row: dynamic_token_cleanup(row['filtered'], row['hashtags']),
    axis=1
)

# Step 9: Final Processed Text
df['processed'] = df.apply(
    lambda row: " ".join(row['normalized_tokens'] + row['emojis']),
    axis=1
)

# =========================
# 10. SAVE RESULTS
# =========================
df_final = df[['id', 'text', 'processed', 'clean_text', 'emojis', 'hashtags', 'source', 'timestamp', 'engagement']]
df_final.to_csv('data/prep_dataset_v3.csv', index=False)
print("Processing complete. Saved to data/prep_dataset_v3.csv")