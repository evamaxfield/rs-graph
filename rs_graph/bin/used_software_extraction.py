#!/usr/bin/env python

import json
import subprocess
import tempfile
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import typer
from eil import Extractor
from git import GitCommandError, Repo
from git.remote import RemoteProgress
from nb_to_src import convert_directory as convert_nb_to_src_in_dir
from prefect import flow
from sqlmodel import Session, col, select
from tqdm import tqdm

from rs_graph.bin.pipeline_utils import (
    _get_small_cpu_api_cluster,
    _wrap_func_with_coiled_prefect_task,
)
from rs_graph.db import models as db_models
from rs_graph.db import utils as db_utils
from rs_graph.utils.identifier_normalization import normalize_name

###############################################################################

app = typer.Typer(rich_markup_mode=None, pretty_exceptions_enable=False)

DEFAULT_LANGUAGE_FILTER = ["Python", "Jupyter Notebook", "R"]

###############################################################################


@dataclass
class ImportRecord:
    software_name: str
    file_path: str | None


@dataclass
class DependencyRecord:
    software_name: str
    version_spec: str | None


@dataclass
class RepoExtractionResult:
    repository_id: int
    owner: str
    name: str
    imports: list[ImportRecord] = field(default_factory=list)
    dependencies: list[DependencyRecord] = field(default_factory=list)
    error: str | None = None


###############################################################################


class _TqdmProgress(RemoteProgress):
    """Progress handler for git clone operations with tqdm integration."""

    def __init__(self, desc: str):
        super().__init__()
        self._tqdm = tqdm(unit="objects", desc=desc, leave=False)

    def update(
        self,
        op_code: int,
        cur_count: str | float,
        max_count: str | float | None = None,
        message: str = "",
    ) -> None:
        try:
            cur = int(cur_count or 0)
            if max_count and int(max_count) > 0:
                total = int(max_count)
                if self._tqdm.total is None or self._tqdm.total < total:
                    self._tqdm.total = total
                self._tqdm.update(cur - self._tqdm.n)
            else:
                self._tqdm.update(cur - self._tqdm.n)
            if message:
                self._tqdm.set_postfix_str(str(message))
        except Exception:  # noqa: BLE001
            pass

    def __enter__(self) -> "_TqdmProgress":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> bool:
        self._tqdm.clear()
        self._tqdm.close()
        return False


###############################################################################


def _query_unprocessed_repositories(
    use_prod: bool,
    language_filter: list[str],
) -> list[tuple[int, str, str]]:
    """Return (repo_id, owner, name) for repos not yet extracted, filtered by language."""
    engine = db_utils.get_engine(use_prod=use_prod)
    with Session(engine) as session:
        # Collect repo IDs that already have imports or dependencies
        imported_ids = set(
            session.exec(
                select(db_models.RepositoryImport.repository_id).distinct()
            ).all()
        )
        dep_ids = set(
            session.exec(
                select(db_models.RepositoryDependency.repository_id).distinct()
            ).all()
        )
        processed_ids = imported_ids | dep_ids

        # Query unique repos linked to at least one document with a target language
        stmt = (
            select(db_models.Repository)
            .join(
                db_models.DocumentRepositoryLink,
                db_models.DocumentRepositoryLink.repository_id == db_models.Repository.id,
            )
            .where(col(db_models.Repository.primary_language).in_(language_filter))
            .distinct()
        )
        all_repos = session.exec(stmt).all()

    return [
        (r.id, r.owner, r.name)
        for r in all_repos
        if r.id is not None and r.id not in processed_ids
    ]


###############################################################################


def _clone_repo(repo_path: Path, repo_full_name: str) -> str | None:
    """Clone a GitHub repo to repo_path. Returns an error string on failure, None on success."""
    try:
        with _TqdmProgress(desc=f"Cloning {repo_full_name}") as progress:
            Repo.clone_from(
                f"https://github.com/{repo_full_name}.git",
                str(repo_path),
                progress=progress,
            )
        return None
    except GitCommandError as e:
        return f"Clone failed: {e}"


def _convert_notebooks(repo_path: Path, repo_full_name: str) -> None:
    """Convert Jupyter notebooks to Python scripts in place."""
    try:
        convert_nb_to_src_in_dir(repo_path, recursive=True, progress_leave=False)
    except Exception as e:  # noqa: BLE001
        print(f"Notebook conversion warning for {repo_full_name}: {e}")


def _get_imports(repo_path: Path, repo_full_name: str) -> list[ImportRecord]:
    """Extract third-party imports via eil."""
    try:
        extractor = Extractor()
        extracted = extractor.extract_from_directory(
            repo_path,
            recursive=True,
            progress_leave=False,
        )
        lib_to_files: dict[str, list[str]] = {}
        for file_path, result in extracted.extracted.items():
            for lib in result.third_party:
                if lib not in lib_to_files:
                    lib_to_files[lib] = []
                try:
                    rel = str(Path(str(file_path)).relative_to(repo_path))
                except ValueError:
                    rel = str(file_path)
                lib_to_files[lib].append(rel)
        return [
            ImportRecord(
                software_name=lib,
                file_path=";".join(paths) if paths else None,
            )
            for lib, paths in lib_to_files.items()
        ]
    except Exception as e:  # noqa: BLE001
        print(f"Import extraction warning for {repo_full_name}: {e}")
        return []


def _get_dependencies(repo_path: Path, repo_full_name: str) -> list[DependencyRecord]:
    """Extract declared dependencies via git-pkgs CLI."""
    try:
        subprocess.run(
            ["git-pkgs", "init", "-q"],
            cwd=str(repo_path),
            capture_output=True,
            check=False,
        )
        dep_proc = subprocess.run(
            ["git-pkgs", "list", "--format", "json", "-q"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if dep_proc.returncode == 0 and dep_proc.stdout.strip():
            raw_deps: list[dict] = json.loads(dep_proc.stdout)
            seen_names: set[str] = set()
            dependencies = []
            for dep in raw_deps:
                dep_name = dep.get("name", "")
                if not dep_name or dep_name in seen_names:
                    continue
                seen_names.add(dep_name)
                dependencies.append(
                    DependencyRecord(
                        software_name=dep_name,
                        version_spec=dep.get("requirement"),
                    )
                )
            return dependencies
    except Exception as e:  # noqa: BLE001
        print(f"Dependency extraction warning for {repo_full_name}: {e}")
    return []


def _extract_repo_imports_and_deps(
    repository_id: int,
    owner: str,
    name: str,
) -> RepoExtractionResult:
    """Clone a repo and extract software imports (eil) and dependencies (git-pkgs)."""
    repo_full_name = f"{owner}/{name}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir) / "repo"
        clone_error = _clone_repo(repo_path, repo_full_name)
        if clone_error:
            return RepoExtractionResult(
                repository_id=repository_id,
                owner=owner,
                name=name,
                error=clone_error,
            )
        
        # Convert, extract imports, and extract deps
        _convert_notebooks(repo_path, repo_full_name)
        imports = _get_imports(repo_path, repo_full_name)
        dependencies = _get_dependencies(repo_path, repo_full_name)
    return RepoExtractionResult(
        repository_id=repository_id,
        owner=owner,
        name=name,
        imports=imports,
        dependencies=dependencies,
    )


def _store_repo_result(result: RepoExtractionResult, use_prod: bool) -> None:
    """Store extraction results to RepositoryImport and RepositoryDependency tables."""
    engine = db_utils.get_engine(use_prod=use_prod)
    with Session(engine) as session:
        for record in result.imports:
            db_utils._get_or_add_and_flush(
                db_models.RepositoryImport(
                    repository_id=result.repository_id,
                    software_name=record.software_name,
                    software_name_normalized=normalize_name(record.software_name),
                    file_path=record.file_path,
                ),
                session,
            )
        for record in result.dependencies:
            db_utils._get_or_add_and_flush(
                db_models.RepositoryDependency(
                    repository_id=result.repository_id,
                    software_name=record.software_name,
                    software_name_normalized=normalize_name(record.software_name),
                    version_spec=record.version_spec,
                ),
                session,
            )
        session.commit()


###############################################################################


@flow(log_prints=True)
def _used_software_extraction_flow(
    use_prod: bool,
    language_filter: list[str],
    use_coiled: bool,
    coiled_region: str,
    limit: int | None,
) -> None:
    repos = _query_unprocessed_repositories(
        use_prod=use_prod,
        language_filter=language_filter,
    )
    print(f"Found {len(repos)} repositories to process")

    if not repos:
        return

    if limit is not None:
        repos = repos[:limit]
        print(f"Limited to {limit} repositories")

    extract_task = _wrap_func_with_coiled_prefect_task(
        _extract_repo_imports_and_deps,
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=10,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
        timeout_seconds=600,
    )

    errors = 0
    for repo_id, owner, name in tqdm(repos, desc="Processing repositories"):
        try:
            future = extract_task.submit(
                repository_id=repo_id,
                owner=owner,
                name=name,
            )
            result: RepoExtractionResult = future.result()
            if result.error:
                print(f"Extraction error for {owner}/{name}: {result.error}")
                errors += 1
                continue
            _store_repo_result(result=result, use_prod=use_prod)
        except Exception as e:  # noqa: BLE001
            print(f"Unexpected error processing {owner}/{name}: {e}")
            print(traceback.format_exc())
            errors += 1

    print(f"Done. {len(repos) - errors} succeeded, {errors} failed.")


@app.command()
def used_software_extraction(
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    language_filter: list[str] | None = None,
    limit: int | None = None,
) -> None:
    """Extract software imports and dependencies from document-linked repositories."""
    _used_software_extraction_flow(
        use_prod=use_prod,
        language_filter=language_filter if language_filter is not None else DEFAULT_LANGUAGE_FILTER,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        limit=limit,
    )


def main() -> None:
    app()
