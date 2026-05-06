# =========================
# data_collection_agent.py
# Agent: collects posts/comments from Facebook, Twitter/X, TikTok, and Instagram
# using Apify actors, then normalizes them into the pipeline schema:
#   id, text, source, timestamp, engagement
#
# Usage:
#   python data_collection_agent.py --query "panagbenga 2025" --max_items 200
#   python data_collection_agent.py --query "panagbenga" --platforms facebook tiktok
#   python data_collection_agent.py --query "panagbenga" --platforms all
# =========================

import argparse
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from apify_client import ApifyClient

from utils import get_logger, save_json, timestamp, log_banner

logger = get_logger("data_collection")

# ---------------------------------------------------------------------------
# Apify Actor IDs (verified 2025-2026)
# ---------------------------------------------------------------------------
ACTORS = {
    "facebook":  "apify/facebook-posts-scraper",
    "tiktok":    "clockworks/tiktok-scraper",
    "twitter":   "apidojo/tweet-scraper",
    "instagram": "apify/instagram-scraper",
}

OUTPUT_CSV  = "panagbenga-dataset.csv"
OUTPUT_JSON = "collection_report.json"


# ---------------------------------------------------------------------------
# Normalizers — one per platform
# Each returns a dict with: id, text, source, timestamp, engagement
# ---------------------------------------------------------------------------

def _make_id(platform: str, raw_id) -> str:
    """Stable unique id: platform prefix + hash of the raw post id."""
    h = hashlib.md5(str(raw_id).encode()).hexdigest()[:10]
    return f"{platform[:2].upper()}-{h}"


def _parse_ts(value) -> str:
    """Best-effort ISO timestamp string from various raw formats."""
    if not value:
        return ""
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()
        except Exception:
            return str(value)
    return str(value)


def normalize_facebook(item: dict) -> dict | None:
    text = (
        item.get("message") or
        item.get("text") or
        item.get("postText") or ""
    ).strip()
    if not text:
        return None
    reactions = (
        item.get("reactionsCount") or
        item.get("likes") or
        item.get("likesCount") or 0
    )
    return {
        "id":         _make_id("facebook", item.get("postId") or item.get("id") or text[:30]),
        "text":       text,
        "source":     item.get("url") or item.get("postUrl") or "",
        "timestamp":  _parse_ts(item.get("time") or item.get("timestamp")),
        "engagement": int(reactions),
    }


def normalize_tiktok(item: dict) -> dict | None:
    text = (
        item.get("text") or
        item.get("description") or
        item.get("desc") or ""
    ).strip()
    if not text:
        return None
    likes = (
        item.get("diggCount") or
        item.get("likes") or
        (item.get("statsV2") or {}).get("diggCount") or
        (item.get("stats") or {}).get("diggCount") or 0
    )
    video_id = item.get("id") or item.get("videoId") or text[:30]
    author   = (item.get("authorMeta") or {}).get("name") or item.get("author") or ""
    source   = (
        item.get("webVideoUrl") or
        item.get("url") or
        (f"https://www.tiktok.com/@{author}/video/{video_id}" if author else "")
    )
    return {
        "id":         _make_id("tiktok", video_id),
        "text":       text,
        "source":     source,
        "timestamp":  _parse_ts(item.get("createTime") or item.get("timestamp")),
        "engagement": int(likes),
    }


def normalize_twitter(item: dict) -> dict | None:
    text = (
        item.get("full_text") or
        item.get("text") or
        item.get("rawContent") or ""
    ).strip()
    if not text:
        return None
    likes = (
        item.get("likeCount") or
        item.get("favoriteCount") or
        item.get("favorite_count") or 0
    )
    tweet_id  = item.get("id") or item.get("tweetId") or item.get("id_str") or ""
    author    = (
        (item.get("author") or {}).get("userName") or
        (item.get("user") or {}).get("screen_name") or ""
    )
    source    = (
        item.get("url") or
        (f"https://twitter.com/{author}/status/{tweet_id}" if author and tweet_id else "")
    )
    return {
        "id":         _make_id("twitter", tweet_id or text[:30]),
        "text":       text,
        "source":     source,
        "timestamp":  _parse_ts(item.get("createdAt") or item.get("created_at")),
        "engagement": int(likes),
    }


def normalize_instagram(item: dict) -> dict | None:
    text = (
        item.get("caption") or
        item.get("text") or
        item.get("alt") or ""
    ).strip()
    if not text:
        return None
    likes = (
        item.get("likesCount") or
        item.get("likes") or
        (item.get("edge_media_preview_like") or {}).get("count") or 0
    )
    return {
        "id":         _make_id("instagram", item.get("id") or item.get("shortCode") or text[:30]),
        "text":       text,
        "source":     item.get("url") or item.get("displayUrl") or "",
        "timestamp":  _parse_ts(item.get("timestamp") or item.get("taken_at_timestamp")),
        "engagement": int(likes),
    }


NORMALIZERS = {
    "facebook":  normalize_facebook,
    "tiktok":    normalize_tiktok,
    "twitter":   normalize_twitter,
    "instagram": normalize_instagram,
}


# ---------------------------------------------------------------------------
# Actor input builders
# ---------------------------------------------------------------------------

def build_facebook_input(query: str, max_items: int) -> dict:
    return {
        "startUrls":    [],
        "searchQuery":  query,
        "maxPosts":     max_items,
        "commentsMode": "RANKED_THREADED",
        "maxComments":  max_items,
        "scrapeComments": True,
    }


def build_tiktok_input(query: str, max_items: int) -> dict:
    return {
        "hashtags":        [query.replace(" ", "")],
        "searchQueries":   [query],
        "resultsPerPage":  max_items,
        "maxResults":      max_items,
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
    }


def build_twitter_input(query: str, max_items: int) -> dict:
    return {
        "searchTerms": [query],
        "maxTweets":   max_items,
        "queryType":   "Latest",
    }


def build_instagram_input(query: str, max_items: int) -> dict:
    return {
        "hashtags":   [query.replace(" ", "")],
        "resultsType": "posts",
        "resultsLimit": max_items,
        "addParentData": False,
    }


INPUT_BUILDERS = {
    "facebook":  build_facebook_input,
    "tiktok":    build_tiktok_input,
    "twitter":   build_twitter_input,
    "instagram": build_instagram_input,
}


# ---------------------------------------------------------------------------
# Core scraping function
# ---------------------------------------------------------------------------

def scrape_platform(
    client: ApifyClient,
    platform: str,
    query: str,
    max_items: int,
) -> list[dict]:
    """Run the Apify actor for one platform and return normalized rows."""
    actor_id    = ACTORS[platform]
    actor_input = INPUT_BUILDERS[platform](query, max_items)
    normalizer  = NORMALIZERS[platform]

    logger.info(f"[{platform}] Starting actor: {actor_id}")
    logger.info(f"[{platform}] Input: {actor_input}")

    try:
        run = client.actor(actor_id).call(run_input=actor_input)
    except Exception as e:
        logger.error(f"[{platform}] Actor failed to start: {e}")
        return []

    dataset_id = run.get("defaultDatasetId")
    if not dataset_id:
        logger.error(f"[{platform}] No dataset returned.")
        return []

    logger.info(f"[{platform}] Actor finished. Fetching dataset {dataset_id}...")

    raw_items = list(client.dataset(dataset_id).iterate_items())
    logger.info(f"[{platform}] Raw items: {len(raw_items)}")

    rows = []
    for item in raw_items:
        normalized = normalizer(item)
        if normalized:
            rows.append(normalized)

    logger.info(f"[{platform}] Normalized rows: {len(rows)}")
    return rows


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def deduplicate(rows: list[dict]) -> list[dict]:
    """Remove exact duplicate texts across platforms."""
    seen_texts = set()
    seen_ids   = set()
    unique     = []
    for row in rows:
        key = row["text"].lower().strip()
        if key in seen_texts or row["id"] in seen_ids:
            continue
        seen_texts.add(key)
        seen_ids.add(row["id"])
        unique.append(row)
    return unique


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def main(
    api_token:  str,
    query:      str,
    platforms:  list[str],
    max_items:  int,
    output_csv: str = OUTPUT_CSV,
):
    log_banner(logger, "Data Collection Agent")
    logger.info(f"Query    : {query}")
    logger.info(f"Platforms: {platforms}")
    logger.info(f"Max items: {max_items} per platform")

    client = ApifyClient(api_token)
    all_rows = []
    report   = {
        "generated_at": timestamp(),
        "query":        query,
        "platforms":    {},
    }

    for platform in platforms:
        if platform not in ACTORS:
            logger.warning(f"Unknown platform '{platform}', skipping.")
            continue

        rows = scrape_platform(client, platform, query, max_items)
        all_rows.extend(rows)
        report["platforms"][platform] = {
            "actor":       ACTORS[platform],
            "items_collected": len(rows),
        }
        logger.info(f"[{platform}] Done. {len(rows)} rows collected.")

    logger.info(f"Total raw rows before dedup: {len(all_rows)}")
    all_rows = deduplicate(all_rows)
    logger.info(f"Total after dedup: {len(all_rows)}")

    if not all_rows:
        logger.warning("No data collected. Check your API token, query, and platform inputs.")
        return

    df = pd.DataFrame(all_rows, columns=["id", "text", "source", "timestamp", "engagement"])
    df.to_csv(output_csv, index=False, encoding="utf-8")
    logger.info(f"Dataset saved to {output_csv}  ({len(df)} rows)")

    report["total_rows"] = len(df)
    report["output_file"] = output_csv
    save_json(report, OUTPUT_JSON)
    logger.info(f"Collection report saved to {OUTPUT_JSON}")

    # Print quick preview
    print("\n--- Sample rows ---")
    print(df.head(5).to_string(index=False))

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Collect social media data via Apify into the pipeline schema."
    )
    parser.add_argument(
        "--api_token", type=str,
        default=None,
        help="Apify API token. Or set APIFY_API_TOKEN env variable.",
    )
    parser.add_argument(
        "--query", type=str, required=True,
        help='Search query / hashtag. E.g. "panagbenga 2025"',
    )
    parser.add_argument(
        "--platforms", nargs="+",
        default=["facebook", "tiktok", "twitter", "instagram"],
        choices=["facebook", "tiktok", "twitter", "instagram", "all"],
        help="Platforms to collect from. Use 'all' for every platform.",
    )
    parser.add_argument(
        "--max_items", type=int, default=200,
        help="Max items to collect per platform.",
    )
    parser.add_argument(
        "--output", type=str, default=OUTPUT_CSV,
        help=f"Output CSV filename (default: {OUTPUT_CSV})",
    )

    args = parser.parse_args()

    # Resolve 'all'
    platforms = (
        list(ACTORS.keys())
        if "all" in args.platforms
        else args.platforms
    )

    # Resolve API token (arg > env var)
    import os
    token = args.api_token or os.environ.get("APIFY_API_TOKEN")
    if not token:
        parser.error(
            "Apify API token required. Pass --api_token or set APIFY_API_TOKEN env variable.\n"
            "Get yours at: https://console.apify.com/account/integrations"
        )

    main(
        api_token  = token,
        query      = args.query,
        platforms  = platforms,
        max_items  = args.max_items,
        output_csv = args.output,
    )
