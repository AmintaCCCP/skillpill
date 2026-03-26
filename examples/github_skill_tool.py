from typing import Literal, Optional


def get_github_issue(
    owner: str,
    repo: str,
    issue_number: int,
    include_comments: bool = False,
    response_format: Literal["summary", "full"] = "summary",
    state_filter: Optional[Literal["open", "closed", "all"]] = None,
) -> dict:
    """Fetch a GitHub issue and optionally include comments."""
    return {
        "owner": owner,
        "repo": repo,
        "issue_number": issue_number,
        "include_comments": include_comments,
        "response_format": response_format,
        "state_filter": state_filter,
        "title": "Example issue",
        "state": "open",
    }
