#!/usr/bin/env python

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from dataclasses_json import DataClassJsonMixin
from git import Commit, Repo
from tqdm import tqdm

#######################################################################################

log = logging.getLogger(__name__)

#######################################################################################


def clone_repo(host_url: str, repo_owner_name: str, dest_dir: str) -> Repo:
    repo_full_url = f"{host_url}/{repo_owner_name}.git"
    log.info(f"Cloning {repo_full_url} to {dest_dir}")
    return Repo.clone_from(repo_full_url, dest_dir)


@dataclass
class CommitDetails(DataClassJsonMixin):
    author_name: str
    author_email: str
    commit_message: str
    commit_datetime: datetime
    commit_hash: str
    positive_lines_changed: int
    negative_lines_changed: int
    abs_lines_changed: int
    n_files_changed: int
    per_file_lines_changed: str


def get_commit_details(commit: Commit) -> CommitDetails:
    # Construct a "per-file-lines-changed" string
    # that looks like:
    # "file1: +10 -5; file2: +2 -0; file3: +3 -3"
    per_file_lines_changed = []
    for file in commit.stats.files:
        per_file_lines_changed.append(
            f"{file}: "
            f"+{commit.stats.files[file]['insertions']} "
            f"-{commit.stats.files[file]['deletions']}"
        )

    per_file_lines_changed_str = "; ".join(per_file_lines_changed)

    # Sum positive lines changed, negative lines changed, and number of files changed
    positive_lines_changed = sum(
        commit.stats.files[file]["insertions"] for file in commit.stats.files
    )
    negative_lines_changed = sum(
        commit.stats.files[file]["deletions"] for file in commit.stats.files
    )
    abs_lines_changed = positive_lines_changed + negative_lines_changed
    n_files_changed = len(commit.stats.files)

    # Process the rest and return
    return CommitDetails(
        author_name=commit.author.name,
        author_email=commit.author.email,
        commit_message=commit.message,
        commit_datetime=commit.authored_datetime,
        commit_hash=commit.hexsha,
        positive_lines_changed=positive_lines_changed,
        negative_lines_changed=negative_lines_changed,
        abs_lines_changed=abs_lines_changed,
        n_files_changed=n_files_changed,
        per_file_lines_changed=per_file_lines_changed_str,
    )


def process_commits(
    repo: Repo, tqdm_kwargs: dict[str, str] | None
) -> list[CommitDetails]:
    # Get count of commits
    commit_count = int(repo.git.rev_list("--count", "HEAD"))

    return [
        get_commit_details(commit)
        for commit in tqdm(
            repo.iter_commits(),
            desc="Processing commits",
            total=commit_count,
            **(tqdm_kwargs or {}),
        )
    ]
