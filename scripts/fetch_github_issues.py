# scripts/fetch_github_issues.py

import os
import json
import time
from github import Github
from tqdm import tqdm
from dotenv import load_dotenv

# Load GitHub token from .env or environment
load_dotenv()
TOKEN = os.getenv("GITHUB_TOKEN")
if not TOKEN:
    raise RuntimeError("Please set your GITHUB_TOKEN in the environment or .env file")

# Initialize GitHub client
gh = Github(TOKEN)

# Repositories to ingest
REPOS = ["apache/solr", "apache/lucene-solr"]
OUTPUT_DIR = "data/github_issues"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CRAWL_DELAY = 1  # seconds

def fetch_and_save_issues(repo_name: str):
    repo = gh.get_repo(repo_name)
    issues = repo.get_issues(state="all")
    total = getattr(issues, "totalCount", None)

    loop = tqdm(issues, total=total, desc=f"Fetching {repo_name}", unit="issue")
    for issue in loop:
        # Fetch comments list
        comments_data = []
        for comment in issue.get_comments():
            comments_data.append({
                "id":          comment.id,
                "author":      comment.user.login,
                "body":        comment.body,
                "created_at":  comment.created_at.isoformat()
            })
            time.sleep(0.1)  # slight delay per-comment to avoid secondary rate limits

        data = {
            "id":           issue.id,
            "number":       issue.number,
            "title":        issue.title,
            "body":         issue.body or "",
            "state":        issue.state,
            "labels":       [l.name for l in issue.labels],
            "comments":     comments_data,          # now the full list of comment objects
            "created_at":   issue.created_at.isoformat(),
            "updated_at":   issue.updated_at.isoformat(),
            "url":          issue.html_url
        }

        safe_name = repo_name.replace("/", "_")
        fn = os.path.join(OUTPUT_DIR, f"{safe_name}_issue_{issue.number}.json")
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)

        time.sleep(CRAWL_DELAY)

def main():
    for repo in REPOS:
        fetch_and_save_issues(repo)

if __name__ == "__main__":
    main()

