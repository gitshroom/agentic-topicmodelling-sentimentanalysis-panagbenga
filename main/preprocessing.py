# =========================
# preprocessing.py
# Optimized multilingual preprocessing
# =========================

import re
import string
import unicodedata

import emoji
import nltk
import pandas as pd
import stopwordsiso

from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

import config

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("punkt_tab", quiet=True) #v2

# =========================================================
# STOPWORDS
# =========================================================

STOPWORDS_EN = set(stopwords.words("english"))
STOPWORDS_TL = stopwordsiso.stopwords("tl")

# Ilocano conversational fillers
STOPWORDS_ILO = {
    "wen",
    "haan",
    "nga",
    "gayam",
    "adi",
    "met",
    "man",
    # Additions v2
    "ken",
    "ti",
    "iti",
    "ni",
    "amin",
    "ditoy",
    "daytoy",
}

STOPWORDS_TL_EXTRA = { # Additions v2
    "sa",       
    "na",       
    "ang",      
    "ng",       
    "mga",      
    "pa",       
    "lang",     
    "ka",       
    "ko",       
    "mo",       
    "po",       
    "si",       
    "yung",     
    "naman",    
    "dito",     
    "din",      
    "pero",     
    "para",     
    "tayo",     
    "kayo",     
    "ito",      
    "rin",      
    "kung",
    "talaga",   
    "pag",      
    "nag",      
    "mag",      
    "muna",     
    "kita",     
    "kasama",   
    "tara",     
    "maraming",  
    "salamat",  
    "batang",   
    "kasi",     
    "daw",      
    "pala",     
    "natin",    
    "nang",
    "buong",     
    "ngayong",   
    "bulaklak",         
}

SOCIAL_MEDIA_STOPWORDS = {

    # Instagram / TikTok junk
    "instagood",
    "photooftheday",
    "picoftheday",
    "igdaily",
    "vscocam",
    "lateupload",
    "followme",
    "selfie",
    "repost",
    "trending",
    "viral",

    # TikTok algorithm bait
    "fyp", 'fypシ', 'viralシ',
    "fypp",
    "fyppp",
    "fypppp",
    "fyppppp",
    "foryou",
    "foryoupage",
    "foryouu",
    "xyzbca",
    "viralvideo",

    # Facebook
    "fb",
    "fbreelsvideo",
    "fbreelsfypシ",
    "fbreels",

    # Generic spam
    "lol",
    "lmao",
    "haha",
    "hahaha",
    "hehe",
    "omg",

    # Generic low-information fillers
    "post",
    "share",
    "comment",
    "like",
    "video",
    "tiktok",

    # Additions v2
    # Instagram repost / engagement junk (corpus-confirmed)
    "regrann",          
    "igers",            
    "reels",            
    "reelsinstagram",   

    # v2 Generic travel/photo hashtags (no topical signal)
    "wanderlust",       
    "travelphotography",
    "travelph",         
    "photography",      
    "photo",            

    # v2 — Instagram story/highlight noise
    "highlights",      
    "highlight",        

    # v2 — URL/platform fragments surviving hashtag stripping
    "com",              
    "www",              
    "app",              
    "via",              
    "instagram",        
    "facebook",         
    "ph",

    # v2 TikTok/FB Reels compound tags
    "fypsi",                       
    "fbreelsfypsi",                
    "fbreels",                     
    "fbreelsvideo",                
    "viralsi",                     
    "reelsvideo",                  
    "reelsvideoシ",                 
    "reelsviralシfb",               
    "viralシfypシ",                  
    "fypageシ",                     
    "fypviralシ",                   
    "trendingreels",               

    # v2  media/celebrity engagement tags
    "gmaregionaltv",               
    "starseverywhere",             
    "highlightseveryone",          
    "highlightseveryonefollowers", 

    # v2 additions — generic photo/travel tags
    "travelgram",                  
    "streetphotography",
    "travelgoals",           

    # v2 additions — CTA and engagement prompt tokens
    "thank",                       
    "thanks",                      
    "everyone",                    
    "follow",                      
    "followers",                   
    "inquiries",                   
    "contact",                     
    "link",                        
    "available",                   
    "please",                      
    "book",                        
    "message",                     
    "official",                    
    "check",                       
    "join",                        
    "supportlocal",
    
    # v2 — new stripped Reels compound forms and generic tags
    "fbreelsfyp",       
    "reelsviral",       
    "fypage",           
    "reelsfb",          
    "fypviral",         
    "reelsfacebook",    
    "travelphilippines",
    "instatravel",      

    # v2 — generic English filler verbs/adverbs not in NLTK stopwords
    "see",      
    "let",      
    "every",    
    "also",     
    "make",     
    "take",     
    "watch",    
    "come",     
    "get",      
    "much",     
    "still",    
    "truly",    
    "always",   
    "keep",     
    "look",

    # v5 — brand fragments, price tags, generic fillers
    "inc",
    "php",
    "bio",
    "kfc",
    "pics",
    "shots",
    "mega",
    "mas",      
    "saya",     
    "tao",      
    "sir",           
}

DOMAIN_STOPWORDS = {
    "panagbenga",
    "baguio",
    "baguiocity",
    "festival",
    "flowerfestival", 
    "philippines",
    
    # Additions v2
    # v2 — Compound hashtag variants of domain terms
    "panagbengafestival", "panagbenga festival",        
    "baguioflowerfestival", "baguio flower festival",
    "baguioph", "baguiocityphilippines", "baguiocityph",

    # v2 — Baguio descriptor tags (zero discriminative value)
    "wheninbaguio",             
    "breathebaguio",            
    "sabaguio",                 
    "summercapital",            
    "cityofpines",         

    # v2 — Venue/event sub-tags synonymous with festival
    "sessionroad", "session road",
    "sessionroadinbloom", "session road in bloom",
    "sessioninbloom", "session in bloom",          
    "floatparade", "float parade",              
    "grandfloatparade",  "grand float parade",
    "bloominmotion", 
    "bikesinbloom",
    "baguioevents", 
    "streetdance",

    # v2 — Tourism campaign slogan (DOT Philippines boilerplate)
    "itsmorefuninthephilippines",
    "baguiotrip",  
    "baguioeats",


    # v2 — Recurring brand/sponsor names drowning out organic topics
    "kapamilyakaravan",         
    "kapamilya",                
    "themalatree",
    "arenaissanceofwonderandbeauty",  # Panagbenga 2023 theme hashtag
    "axisdancestudio",
    
    # v2 additions — compound variants 
    "summercapitalofthephilippines",   
    "baguiofeels",                     
    "baguiobased",                     
    "sessionroadbaguiocity",           
    "sessionroadbaguio",               
    "baguiostaycation",               
    "baguiofoodtrip",                  
    "baguiotour",                      
    "lovethephilippines",              
    "choosephilippines",               
    "findyourselfinthecordilleras",    
    "karavan",

    # v2 generic
    "foodtrip",
    "foodie",

    # v5
    "quiapo",
    "axissolstice",
    "bloomingwithoutend",
    "cokestudioph",
    "pilipinas",
}

ALL_STOPWORDS = (
    STOPWORDS_EN
    | STOPWORDS_TL
    | STOPWORDS_ILO
    | STOPWORDS_TL_EXTRA
    | SOCIAL_MEDIA_STOPWORDS
    | DOMAIN_STOPWORDS
)

# =========== V2 Blocked Accounts ==================
# Each entry is a string that uniquely identifies an account's posts
# in the text column (e.g. their hashtag or website domain).
# Rows where text contains ANY of these signals are dropped entirely
# before preprocessing begins.
#
# Manually Verified accounts:
#   themalatree — crystal bracelet shop; 98 posts confirmed as pure ads,
#                 zero relation to Panagbenga Festival.
#                 Verified by: manual post inspection (May 12 2026)
BLOCKED_ACCOUNT_SIGNALS = {
    "#themalatree",       # crystal bracelet ads — 98 posts, manually verified
    "themalatree.com",    # same account, via URL variant in their posts
}

# =========================================================
# HELPERS
# =========================================================

def normalize_repeated_chars(text: str) -> str:
    return re.sub(r"(.)\1{2,}", r"\1\1", text)

def remove_urls_mentions(text: str) -> str:
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    return text

def clean_hashtags(text: str) -> str:
    text = re.sub(r"#(\w+?)(20\d{2})", r"\1", text)
    text = re.sub(r"#", "", text)
    return text

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFKD", text)

def strip_non_ascii(text: str) -> str: 
    # v2 FIX: Remove all non-ASCII characters.
    # Python's \w matches Unicode letters including Japanese katakana (シ),
    # so the existing [^\w\s] step does NOT remove them.
    # This caused TikTok tags like fypシ, fbreelsfypシ, viralシ to survive
    # preprocessing. Running after emoji removal catches remaining
    # non-Latin Unicode embedded in hashtag compounds.
    return re.sub(r"[^\x00-\x7F]+", " ", text)

def extract_year(ts):
    if pd.isna(ts):
        return None

    ts = str(ts)

    match = re.match(r"(\d{4})", ts)

    if match:
        year = int(match.group(1))
        if config.YEAR_START <= year <= config.YEAR_END:
            return year

    return None

# v2
def is_blocked_account(text: str) -> bool:
    t = str(text).lower()
    return any(signal in t for signal in BLOCKED_ACCOUNT_SIGNALS)

# =========================================================
# MAIN CLEANING
# =========================================================

def preprocess_text(text: str) -> str:

    # v2 FIX: normalize_unicode MUST run before lower().
    # Unicode Mathematical Bold/Italic letters (U+1D400 range) used in
    # stylised Instagram text (e.g. 𝙋𝙖𝙣𝙖𝙜𝙗𝙚𝙣𝙜𝙖) do not respond to
    # lower() — they stay uppercase. NFKD converts them to ASCII
    # equivalents but preserves their case. If lower() runs first, NFKD
    # then produces 'Panagbenga' (capital P), which escapes stopword
    # matching. Running NFKD first, then lower(), produces 'panagbenga'.
    text = unicodedata.normalize("NFKD", text)

    text = text.lower()

    text = remove_urls_mentions(text)

    text = clean_hashtags(text)

    text = normalize_repeated_chars(text)

    text = BeautifulSoup(text, "html.parser").get_text()

    text = emoji.replace_emoji(text, replace=" ")
    
    text = strip_non_ascii(text) #v2 katakana removal

    text = re.sub(r"\d+", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    tokens = word_tokenize(text)

    cleaned_tokens = []

    for token in tokens:

        if len(token) <= 2:
            continue

        if token in ALL_STOPWORDS:
            continue

        if token.isnumeric():
            continue

        if re.fullmatch(r"[a-z]*\d+[a-z]*", token):
            continue

        cleaned_tokens.append(token)

    # Remove duplicates while preserving order
    cleaned_tokens = list(dict.fromkeys(cleaned_tokens))

    return " ".join(cleaned_tokens)

# =========================================================
# MAIN
# =========================================================

def main():

    print("[preprocessing] Loading dataset...")

    df = pd.read_csv(config.RAW_FILE)

    df = df[["id", "text", "source", "timestamp", "engagement"]]

    # v2 blocked account filter
    df["text"] = df["text"].fillna("")
    before = len(df)
    df = df[~df["text"].apply(is_blocked_account)]
    print(f"[preprocessing] Blocked account filter removed {before - len(df)} rows")

    df["year"] = df["timestamp"].apply(extract_year)

    df = df.dropna(subset=["year"])

    print("[preprocessing] Cleaning text...")

    df["processed"] = df["text"].apply(preprocess_text)

    # Remove empty docs AFTER preprocessing
    df = df[df["processed"].str.strip().astype(bool)]

    # Remove tiny docs
    df = df[df["processed"].str.split().apply(len) >= 3]

    df.to_csv(config.PREPROCESSED_FILE, index=False)

    print(f"[preprocessing] Saved {len(df)} rows")

    return df

if __name__ == "__main__":
    main()