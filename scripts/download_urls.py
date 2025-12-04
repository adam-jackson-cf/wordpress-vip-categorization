#!/usr/bin/env python3
"""
Download HTML content from URLs in CSV file and save with metadata.
"""
import csv
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple
import urllib.request
import urllib.error
from datetime import datetime

# Configuration
CSV_PATH = "/Users/adamjackson/Projects/wordpress-vip-categorization/data/examples/sample_matches.csv"
OUTPUT_DIR = Path("/Users/adamjackson/Projects/wordpress-vip-categorization/data/examples/content")
ERROR_LOG = OUTPUT_DIR / "download_errors.log"
SUMMARY_FILE = OUTPUT_DIR / "download_summary.json"

# User agent to avoid being blocked
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

def download_url(url: str, timeout: int = 30) -> Tuple[bool, str, int]:
    """
    Download content from URL.

    Returns:
        Tuple of (success, content_or_error, size)
    """
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as response:
            content = response.read()
            return True, content.decode('utf-8', errors='ignore'), len(content)
    except urllib.error.HTTPError as e:
        return False, f"HTTP Error {e.code}: {e.reason}", 0
    except urllib.error.URLError as e:
        return False, f"URL Error: {e.reason}", 0
    except TimeoutError:
        return False, "Timeout: Request took too long", 0
    except Exception as e:
        return False, f"Unexpected error: {str(e)}", 0

def main():
    """Main download function."""
    print(f"Starting download process at {datetime.now().isoformat()}")
    print(f"Reading CSV from: {CSV_PATH}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("-" * 80)

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Statistics
    stats = {
        "total_urls": 0,
        "successful": 0,
        "failed": 0,
        "total_size_bytes": 0,
        "start_time": datetime.now().isoformat(),
        "errors": []
    }

    # Open error log
    error_log = open(ERROR_LOG, 'w')
    error_log.write(f"Download Error Log - {datetime.now().isoformat()}\n")
    error_log.write("=" * 80 + "\n\n")

    try:
        # Read CSV and process each row
        with open(CSV_PATH, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)

            for row_num, row in enumerate(reader, start=1):
                stats["total_urls"] += 1
                source_url = row.get('source_url', '').strip()

                if not source_url:
                    print(f"Row {row_num}: Skipping empty URL")
                    continue

                print(f"\nRow {row_num}: {source_url}")

                # Download content
                success, content_or_error, size = download_url(source_url)

                if success:
                    # Save HTML content
                    html_file = OUTPUT_DIR / f"page_{row_num}.html"
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(content_or_error)

                    # Save metadata
                    metadata = {
                        "row_number": row_num,
                        "source_url": row.get('source_url', ''),
                        "target_url": row.get('target_url', ''),
                        "category": row.get('category', ''),
                        "similarity_score": row.get('similarity_score', ''),
                        "download_timestamp": datetime.now().isoformat(),
                        "content_size_bytes": size,
                        "html_file": str(html_file.name)
                    }

                    meta_file = OUTPUT_DIR / f"page_{row_num}_meta.json"
                    with open(meta_file, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)

                    stats["successful"] += 1
                    stats["total_size_bytes"] += size
                    print(f"  ✓ Success ({size:,} bytes)")

                    # Be polite - add small delay between requests
                    time.sleep(0.5)

                else:
                    # Log error
                    error_info = {
                        "row_number": row_num,
                        "url": source_url,
                        "error": content_or_error,
                        "timestamp": datetime.now().isoformat()
                    }
                    stats["errors"].append(error_info)
                    stats["failed"] += 1

                    error_log.write(f"Row {row_num}: {source_url}\n")
                    error_log.write(f"Error: {content_or_error}\n\n")

                    print(f"  ✗ Failed: {content_or_error}")

    finally:
        error_log.close()

    # Add end time to stats
    stats["end_time"] = datetime.now().isoformat()
    stats["total_size_mb"] = round(stats["total_size_bytes"] / (1024 * 1024), 2)

    # Save summary
    with open(SUMMARY_FILE, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Total URLs processed: {stats['total_urls']}")
    print(f"Successful downloads: {stats['successful']}")
    print(f"Failed downloads: {stats['failed']}")
    print(f"Total content size: {stats['total_size_mb']} MB ({stats['total_size_bytes']:,} bytes)")
    print(f"\nFiles saved to: {OUTPUT_DIR}")
    print(f"Error log: {ERROR_LOG}")
    print(f"Summary report: {SUMMARY_FILE}")

    if stats["errors"]:
        print(f"\nFailed URLs ({len(stats['errors'])}):")
        for error in stats["errors"]:
            print(f"  • Row {error['row_number']}: {error['url']}")
            print(f"    Error: {error['error']}")

    return stats

if __name__ == "__main__":
    main()
