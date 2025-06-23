from github import Github
import json
import os

# Authenticate (use a PAT with repo scope)
gh = Github("")

# Repos to ingest
repos = ["apache/solr", "apache/lucene-solr"]
output_dir = "data/github_issues"
os.makedirs(output_dir, exist_ok=True)

for repo_name in repos:
    repo = gh.get_repo(repo_name)
    issues = repo.get_issues(state="all")  # open+closed
    for issue in issues:
        data = {
            "id": issue.id,
            "number": issue.number,
            "title": issue.title,
            "body": issue.body or "",
            "state": issue.state,
            "labels": [l.name for l in issue.labels],
            "comments": issue.comments,
            "created_at": issue.created_at.isoformat(),
            "updated_at": issue.updated_at.isoformat(),
            "url": issue.html_url
        }
        fn = os.path.join(output_dir, f"{repo_name.replace('/', '_')}_issue_{issue.number}.json")
        with open(fn, "w") as f:
            json.dump(data, f, indent=2)

