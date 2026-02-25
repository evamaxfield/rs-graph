#!/usr/bin/env python

import json
import platform
import shutil
import subprocess
import tarfile
import tempfile
import traceback
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
import typer
from eil import Extractor
from git import GitCommandError, Repo
from git.remote import RemoteProgress
from nb_to_src import convert_directory as convert_nb_to_src_in_dir
from prefect import flow
from sqlmodel import Session, col, select
from tqdm import tqdm

from rs_graph import types
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
_GIT_PKGS_VERSION = "0.14.0"
_GIT_PKGS_CACHE = Path("/tmp/git-pkgs-bin/git-pkgs")


def _ensure_git_pkgs() -> str:
    """Return path to git-pkgs binary, downloading and caching if not on PATH."""
    # Check for git-pkgs on PATH first
    print("Checking for git-pkgs binary...")
    git_pkgs_path = shutil.which("git-pkgs")
    if git_pkgs_path:
        print("Found git-pkgs on PATH")
        return git_pkgs_path
    if _GIT_PKGS_CACHE.exists():
        print(f"Found cached git-pkgs binary at {_GIT_PKGS_CACHE}")
        return str(_GIT_PKGS_CACHE)

    # No git-pkgs, get machine details
    print("git-pkgs not found on PATH or cache, downloading...")
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("aarch64", "arm64") else "amd64"
    url = (
        f"https://github.com/git-pkgs/git-pkgs/releases/download/"
        f"v{_GIT_PKGS_VERSION}/git-pkgs_{_GIT_PKGS_VERSION}_linux_{arch}.tar.gz"
    )

    # Make cache dir and download
    _GIT_PKGS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    tarball = _GIT_PKGS_CACHE.parent / "git-pkgs.tar.gz"
    urllib.request.urlretrieve(url, tarball)

    # Decompress and set permissions
    with tarfile.open(tarball) as tf:
        tf.extract("git-pkgs", path=_GIT_PKGS_CACHE.parent, filter="data")
    tarball.unlink()
    _GIT_PKGS_CACHE.chmod(0o755)

    # Return the path
    print(f"git-pkgs stored to {_GIT_PKGS_CACHE}")
    return str(_GIT_PKGS_CACHE)


###############################################################################

# TODO: we want to change import records to record file_paths instead of file_path
# this should still be a string
# but it can contain multiple paths separated by semicolons.
# TODO: we want to add fields for ecosystem (e.g., PyPI, CRAN, conda),
# dependency type (e.g., runtime vs dev),
# and manifest paths to the DependencyRecord as well.


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
        except Exception:
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


@dataclass
class UnprocessedRepository:
    repo_id: int
    owner: str
    name: str


def _query_unprocessed_repositories(
    use_prod: bool,
    language_filter: list[str],
) -> list[UnprocessedRepository]:
    """Return (repo_id, owner, name) for repos not yet extracted, filtered by language."""
    engine = db_utils.get_engine(use_prod=use_prod)
    with Session(engine) as session:
        # Collect repo IDs that already have imports or dependencies
        imported_ids = set(
            session.exec(select(db_models.RepositoryImport.repository_id).distinct()).all()
        )
        dep_ids = set(
            session.exec(select(db_models.RepositoryDependency.repository_id).distinct()).all()
        )
        processed_ids = imported_ids | dep_ids

        # Query unique repos linked to at least one document with a target language
        stmt = (
            select(db_models.Repository)
            .join(
                db_models.DocumentRepositoryLink,
                col(db_models.DocumentRepositoryLink.repository_id)
                == col(db_models.Repository.id),
            )
            .where(col(db_models.Repository.primary_language).in_(language_filter))
            .distinct()
        )
        all_repos = session.exec(stmt).all()

    return [
        UnprocessedRepository(repo_id=r.id, owner=r.owner, name=r.name)
        for r in all_repos
        if r.id is not None and r.id not in processed_ids
    ]


###############################################################################


def _clone_repo(repo_path: Path, repo_full_name: str) -> None | types.ErrorResult:
    """Clone a GitHub repo to repo_path. Returns an error string on failure, None on success."""
    try:
        with _TqdmProgress(desc=f"Cloning {repo_full_name}") as progress:
            Repo.clone_from(
                f"https://github.com/{repo_full_name}.git",
                str(repo_path),
                progress=progress,  # type: ignore[arg-type]
            )
        return None
    except GitCommandError as e:
        return types.ErrorResult(
            source="used-software-extraction",
            step="clone_repo",
            identifier=repo_full_name,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _convert_notebooks(repo_path: Path, repo_full_name: str) -> types.ErrorResult | None:
    """Convert Jupyter notebooks to Python scripts in place."""
    try:
        convert_nb_to_src_in_dir(repo_path, recursive=True, progress_leave=False)
    except Exception as e:
        return types.ErrorResult(
            source="used-software-extraction",
            step="convert_notebooks",
            identifier=repo_full_name,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _get_imports(
    repo_path: Path, repo_full_name: str
) -> list[ImportRecord] | types.ErrorResult:
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
    except Exception as e:
        return types.ErrorResult(
            source="used-software-extraction",
            step="get_imports",
            identifier=repo_full_name,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _get_dependencies(
    repo_path: Path, repo_full_name: str
) -> list[DependencyRecord] | types.ErrorResult:
    """Extract declared dependencies via git-pkgs CLI."""
    try:
        # TODO: we want to also store the ecosystem (e.g., PyPI, CRAN, conda)
        # the dep type, and manifest paths
        git_pkgs_bin = _ensure_git_pkgs()
        dep_proc = subprocess.run(
            [git_pkgs_bin, "list", "--format", "json", "-q"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            check=False,
        )
        if dep_proc.returncode == 0 and dep_proc.stdout.strip():
            raw_deps: list[dict] = json.loads(dep_proc.stdout) or []
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
        else:
            print(
                f"git-pkgs failed for {repo_full_name} "
                f"(returncode={dep_proc.returncode}): {dep_proc.stderr.strip()}"
            )
            raise RuntimeError(
                f"git-pkgs failed with code {dep_proc.returncode}: {dep_proc.stderr}"
            )
    except Exception as e:
        return types.ErrorResult(
            source="used-software-extraction",
            step="get_dependencies",
            identifier=repo_full_name,
            error=str(e),
            traceback=traceback.format_exc(),
        )


def _extract_repo_imports_and_deps(
    repository_id: int,
    owner: str,
    name: str,
) -> RepoExtractionResult | types.ErrorResult:
    """Clone a repo and extract software imports (eil) and dependencies (git-pkgs)."""
    repo_full_name = f"{owner}/{name}"
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_path = Path(tmp_dir) / "repo"
        clone_error = _clone_repo(repo_path, repo_full_name)
        if clone_error:
            return clone_error

        # Convert, extract imports, and extract deps
        convert_result = _convert_notebooks(repo_path, repo_full_name)
        if isinstance(convert_result, types.ErrorResult):
            return convert_result
        imports = _get_imports(repo_path, repo_full_name)
        if isinstance(imports, types.ErrorResult):
            return imports
        dependencies = _get_dependencies(repo_path, repo_full_name)
        if isinstance(dependencies, types.ErrorResult):
            return dependencies

    # Made it through, we have results
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
    coiled_workers: int,
    batch_size: int,
    limit: int | None,
    errors_cache_file: str,
) -> None:
    # Construct the actual full path of the errors cache file
    errors_cache_path = Path(errors_cache_file).resolve()

    # Print dataset and coiled status
    print("-" * 80)
    print("Pipeline Options:")
    print(f"Use Prod Database: {use_prod}")
    print(f"Language Filter: {language_filter}")
    print(f"Use Coiled: {use_coiled}")
    print(f"Coiled Region: {coiled_region}")
    print(f"Coiled Workers: {coiled_workers}")
    # print(f"Batch Size: {batch_size}")
    print(f"Total Processing Limit: {limit}")
    print(f"Errors Cache File: {errors_cache_path}")
    print("-" * 80)

    # Get list of repos to process
    print("Retrieving list of repositories to process...")
    repos = _query_unprocessed_repositories(
        use_prod=use_prod,
        language_filter=language_filter,
    )
    repos = repos[:limit] if limit is not None else repos
    print(f"Retrieved {len(repos)} repositories to process")

    # Handle no repos to process
    if not repos:
        print("No repositories to process. Exiting.")
        return

    # Construct the extract task mappable function
    extract_task = _wrap_func_with_coiled_prefect_task(
        _extract_repo_imports_and_deps,
        coiled_func_name="extract_repo_imports_and_deps",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=coiled_workers,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
        timeout_seconds=120,
    )

    # Process in batches
    batches = [repos[i : i + batch_size] for i in range(0, len(repos), batch_size)]
    for batch in tqdm(batches, desc="Processing batches", total=len(batches)):
        batch_futures = extract_task.map(
            repository_id=[r.repo_id for r in batch],
            owner=[r.owner for r in batch],
            name=[r.name for r in batch],
        )
        batch_results = []
        for future, repo in zip(batch_futures, batch, strict=False):
            try:
                batch_results.append(future.result())
            except TimeoutError:
                repo_full_name = f"{repo.owner}/{repo.name}"
                print(f"Timeout reached for {repo_full_name}, skipping.")
                batch_results.append(
                    types.ErrorResult(
                        source="used-software-extraction",
                        step="task_timeout",
                        identifier=repo_full_name,
                        error="Task timed out",
                        traceback="",
                    )
                )

        # Split out successful results vs errors
        batch_success_results: list[RepoExtractionResult] = []
        batch_errors: list[types.ErrorResult] = []
        for result in batch_results:
            if isinstance(result, types.ErrorResult):
                batch_errors.append(result)
            else:
                batch_success_results.append(result)

        # Log counts for this batch
        print(
            f"Batch completed with {len(batch_success_results)} successes "
            f"and {len(batch_errors)} errors"
        )

        # Store successful results after each batch to avoid
        # losing everything if something goes wrong at the end
        print("Storing results for batch")
        for result in batch_success_results:
            _store_repo_result(result=result, use_prod=use_prod)

        # Store errors to parquet file
        if batch_errors:
            # Read in existing errors if the file already exists,
            # otherwise start with an empty list
            if errors_cache_path.exists():
                existing_errors = pl.read_parquet(errors_cache_path)
            else:
                existing_errors = pl.DataFrame()

            # Convert new errors to a DataFrame and concatenate with existing errors
            new_errors_df = pl.DataFrame([e.to_dict() for e in batch_errors])
            combined_errors = pl.concat([existing_errors, new_errors_df])

            # Write combined errors back to the parquet file
            combined_errors.write_parquet(errors_cache_path)


@app.command()
def used_software_extraction(
    use_prod: bool = False,
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    coiled_workers: int = 24,
    language_filter: list[str] | None = None,
    batch_size: int = 48,
    limit: int | None = None,
    errors_cache_file: str = "used-software-extraction-errors.parquet",
) -> None:
    """
    Extract software imports and dependencies from document-linked repositories.

    Default language filter is Python, Jupyter Notebook, and R.
    """
    _used_software_extraction_flow(
        use_prod=use_prod,
        language_filter=language_filter
        if language_filter is not None
        else DEFAULT_LANGUAGE_FILTER,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        coiled_workers=coiled_workers,
        batch_size=batch_size,
        limit=limit,
        errors_cache_file=errors_cache_file,
    )


def main() -> None:
    app()
