#!/usr/bin/env python

import json
import platform
import random
import shutil
import subprocess
import tarfile
import tempfile
import time
import traceback
import urllib.request
from collections import Counter
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
from rs_graph.sources.softcite_2025 import SOFTCITE_MENTIONS_NAME, SOFTCITE_PAPERS_NAME
from rs_graph.utils.identifier_normalization import (
    normalize_doi_col,
    normalize_name,
)

###############################################################################

app = typer.Typer(rich_markup_mode=None, pretty_exceptions_enable=False)

DEFAULT_LANGUAGE_FILTER = ["Python", "Jupyter Notebook", "R"]
_GIT_PKGS_VERSION = "0.15.0"
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


@dataclass
class ImportRecord:
    software_name: str
    file_paths: str | None


@dataclass
class DependencyRecord:
    software_name: str
    version_spec: str | None
    ecosystem: str | None = None
    dependency_type: str | None = None
    manifest_paths: str | None = None


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
                self._tqdm.set_postfix_str(message)
        except Exception:
            pass

    def __enter__(self) -> "_TqdmProgress":
        return self

    def __exit__(
        self,
        exc_type: type | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        self._tqdm.clear()
        self._tqdm.close()


###############################################################################


@dataclass
class UnprocessedRepository:
    repo_id: int
    owner: str
    name: str
    language: str


def _query_unprocessed_repositories(
    database_path: str,
    language_filter: list[str],
) -> tuple[list[UnprocessedRepository], Counter[str]]:
    """Return unprocessed repos and a count of already-processed repos by language."""
    engine = db_utils.get_engine(database_path=database_path)
    with Session(engine) as session:
        # Collect repo IDs that already have imports or dependencies
        imported_ids = set(
            session.exec(select(col(db_models.RepositoryImport.repository_id)).distinct()).all()
        )
        dep_ids = set(
            session.exec(
                select(col(db_models.RepositoryDependency.repository_id)).distinct()
            ).all()
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

    processed_by_language: Counter[str] = Counter()
    unprocessed: list[UnprocessedRepository] = []
    for r in all_repos:
        if r.id is None:
            continue
        lang = r.primary_language or "Unknown"
        if r.id in processed_ids:
            processed_by_language[lang] += 1
        else:
            unprocessed.append(
                UnprocessedRepository(repo_id=r.id, owner=r.owner, name=r.name, language=lang)
            )
    return unprocessed, processed_by_language


###############################################################################


def _clone_repo(repo_path: Path, repo_full_name: str) -> None | types.ErrorResult:
    """Clone a GitHub repo to repo_path. Returns an error string on failure, None on success."""
    try:
        with _TqdmProgress(desc=f"Cloning {repo_full_name}") as progress:
            Repo.clone_from(
                f"https://github.com/{repo_full_name}.git",
                str(repo_path),
                progress=progress,  # type: ignore[arg-type]
                multi_options=["--depth=1", "--single-branch"],
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
                file_paths=";".join(paths) if paths else None,
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
            timeout=90,
        )
        if dep_proc.returncode == 0 and dep_proc.stdout.strip():
            raw_deps: list[dict] = json.loads(dep_proc.stdout) or []
            seen_names: set[str] = set()
            dependencies = {}
            for dep in raw_deps:
                dep_name = dep.get("name", "")
                if not dep_name or dep_name in seen_names:
                    continue
                seen_names.add(dep_name)
                if "purl" not in dep:
                    continue
                dep_purl = dep["purl"]
                if dep_purl not in dependencies:
                    dependencies[dep_purl] = DependencyRecord(
                        software_name=dep_name,
                        version_spec=dep.get("requirement"),
                        ecosystem=dep.get("ecosystem"),
                        dependency_type=dep.get("dependency_type"),
                        manifest_paths=dep.get("manifest_path"),
                    )
                else:
                    # If we see the same dep again with a different manifest path,
                    # we want to combine them
                    existing = dependencies[dep_purl]
                    new_manifest_paths = set(
                        existing.manifest_paths.split(";") if existing.manifest_paths else []
                    )
                    new_manifest_paths.add(dep.get("manifest_path", ""))
                    existing.manifest_paths = ";".join(sorted(new_manifest_paths))

            # Return the list of unique dependencies
            return list(dependencies.values())
        else:
            print(
                f"git-pkgs failed for {repo_full_name} "
                f"(returncode={dep_proc.returncode}): {dep_proc.stderr.strip()}"
            )
            raise RuntimeError(
                f"git-pkgs failed with code {dep_proc.returncode}: {dep_proc.stderr}"
            )
    except subprocess.TimeoutExpired:
        return types.ErrorResult(
            source="used-software-extraction",
            step="get_dependencies",
            identifier=repo_full_name,
            error="git-pkgs timed out after 90 seconds",
            traceback=traceback.format_exc(),
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


def _bulk_get_or_add(
    model_cls: type[db_models.RepositoryImport] | type[db_models.RepositoryDependency],
    repository_id: int,
    candidates: list[db_models.RepositoryImport] | list[db_models.RepositoryDependency],
    session: Session,
) -> None:
    """Add only the candidates whose (repository_id, software_name) isn't already stored."""
    if not candidates:
        return

    # Dedup within the batch itself, keeping first occurrence per software_name
    deduped: dict[str, db_models.RepositoryImport | db_models.RepositoryDependency] = {}
    for candidate in candidates:
        deduped.setdefault(candidate.software_name, candidate)

    # Single bulk lookup for names already stored for this repository
    existing_names = set(
        session.exec(
            select(col(model_cls.software_name)).where(
                col(model_cls.repository_id) == repository_id,
                col(model_cls.software_name).in_(list(deduped)),
            )
        ).all()
    )

    session.add_all(
        [candidate for name, candidate in deduped.items() if name not in existing_names]
    )


def _store_repo_result(result: RepoExtractionResult, database_path: str) -> None:
    """Store extraction results to RepositoryImport and RepositoryDependency tables."""
    engine = db_utils.get_engine(database_path=database_path)
    with Session(engine) as session:
        import_candidates = [
            db_models.RepositoryImport(
                repository_id=result.repository_id,
                software_name=record.software_name,
                software_name_normalized=normalize_name(record.software_name),
                file_paths=record.file_paths,
            )
            for record in result.imports
        ]
        dependency_candidates = [
            db_models.RepositoryDependency(
                repository_id=result.repository_id,
                software_name=record.software_name,
                software_name_normalized=normalize_name(record.software_name),
                version_spec=record.version_spec,
                ecosystem=record.ecosystem,
                dependency_type=record.dependency_type,
                manifest_paths=record.manifest_paths,
            )
            for record in result.dependencies
        ]

        _bulk_get_or_add(
            db_models.RepositoryImport, result.repository_id, import_candidates, session
        )
        _bulk_get_or_add(
            db_models.RepositoryDependency,
            result.repository_id,
            dependency_candidates,
            session,
        )
        session.commit()
        # Give the (very large) SQLite file a moment to settle after each repo's write
        time.sleep(0.02)


###############################################################################


def _bulk_lookup_document_ids_by_doi(dois: list[str], session: Session) -> dict[str, int]:
    """Bulk look up document IDs for a batch of DOIs (Document, then DocumentAlternateDOI)."""
    doc_id_by_doi: dict[str, int] = {
        doi: doc_id
        for doi, doc_id in session.exec(
            select(db_models.Document.doi, db_models.Document.id).where(
                col(db_models.Document.doi).in_(dois)
            )
        ).all()
        if doc_id is not None
    }

    remaining = [doi for doi in dois if doi not in doc_id_by_doi]
    if remaining:
        doc_id_by_doi.update(
            session.exec(
                select(
                    db_models.DocumentAlternateDOI.doi,
                    db_models.DocumentAlternateDOI.document_id,
                ).where(col(db_models.DocumentAlternateDOI.doi).in_(remaining))
            ).all()
        )

    return doc_id_by_doi


def _bulk_get_or_add_mentions(
    candidates: list[db_models.DocumentSoftwareMention],
    session: Session,
) -> None:
    """Add only mention candidates not already stored (document_id, name, mention_id)."""
    if not candidates:
        return

    # Dedup within the batch itself, keyed on the model's actual unique constraint
    deduped: dict[tuple[int, str, str], db_models.DocumentSoftwareMention] = {}
    for candidate in candidates:
        assert candidate.document_id is not None
        key = (candidate.document_id, candidate.software_name, candidate.softcite_mention_id)
        deduped.setdefault(key, candidate)

    document_ids = {key[0] for key in deduped}
    mention_ids = {key[2] for key in deduped}
    existing_keys = set(
        session.exec(
            select(
                col(db_models.DocumentSoftwareMention.document_id),
                col(db_models.DocumentSoftwareMention.software_name),
                col(db_models.DocumentSoftwareMention.softcite_mention_id),
            ).where(
                col(db_models.DocumentSoftwareMention.document_id).in_(document_ids),
                col(db_models.DocumentSoftwareMention.softcite_mention_id).in_(mention_ids),
            )
        ).all()
    )

    session.add_all(
        [candidate for key, candidate in deduped.items() if key not in existing_keys]
    )


@app.command()
def ingest_softcite_mentions(
    data_dir: str,
    database_path: str = typer.Argument(
        help="Path to the SQLite database file to use.",
    ),
    batch_size: int = 200,
    reprocess_all: bool = False,
) -> None:
    """
    Ingest SoftCite software mentions into the DocumentSoftwareMention table.

    DATA_DIR should be the path to the raw SoftCite 2025 data directory
    (containing papers.parquet and mentions.pdf.parquet).

    By default, DOIs that already have mentions in the database are skipped so
    new document records can be picked up incrementally on subsequent runs.
    Pass --reprocess-all to re-ingest every DOI regardless.
    """
    data_dir_path = Path(data_dir).resolve()

    # Load and join SoftCite parquet files
    print("Loading SoftCite parquet files...")
    mentions = pl.scan_parquet(data_dir_path / SOFTCITE_MENTIONS_NAME).select(
        "software_mention_id", "paper_id", "software_raw", "context_full_text"
    )
    papers = pl.scan_parquet(data_dir_path / SOFTCITE_PAPERS_NAME).select("paper_id", "doi")
    # This type ignore is caused by the .collect() statement at
    # the end. Polars recently added InProcess execution which
    # causes the type checker to not know which DataFrame type to
    # expect until after execution.
    df: pl.DataFrame = (  # type: ignore
        mentions.join(
            papers,
            on="paper_id",
            how="inner",
        )
        .with_columns(
            normalize_doi_col("doi").alias("doi_normalized"),
            pl.col("software_mention_id").str.strip_chars().alias("software_mention_id"),
            pl.col("context_full_text").str.strip_chars().alias("context_full_text"),
        )
        .collect()
    )
    print(f"Total SoftCite mention rows: {len(df)}")
    print(f"Total unique DOIs in SoftCite mentions: {len(df['doi_normalized'].unique())}")

    # Optionally skip DOIs that already have mentions in the DB
    if not reprocess_all:
        engine = db_utils.get_engine(database_path=database_path)
        with Session(engine) as session:
            processed_doc_ids = set(
                session.exec(
                    select(col(db_models.DocumentSoftwareMention.document_id)).distinct()
                ).all()
            )
            if processed_doc_ids:
                processed_dois = set(
                    session.exec(
                        select(db_models.Document.doi).where(
                            col(db_models.Document.id).in_(processed_doc_ids)
                        )
                    ).all()
                )
                before = len(df)
                df = df.filter(~pl.col("doi_normalized").is_in(processed_dois))
                print(
                    f"Skipping {before - len(df)} rows for {len(processed_dois)} "
                    f"already-processed DOIs; {len(df)} rows remaining"
                )

    unique_dois: list[str] = df["doi_normalized"].unique().to_list()
    if len(unique_dois) == 0:
        print("No new rows to process. Exiting.")
        return

    # Partition once so each DOI's rows are an O(1) dict lookup inside the batch loop
    mentions_by_doi = df.partition_by("doi_normalized", as_dict=True)

    # Process in batches of unique DOIs, with bulk document-id lookups per batch
    engine = db_utils.get_engine(database_path=database_path)
    matched_count = 0
    skipped_count = 0
    doi_batches = [
        unique_dois[i : i + batch_size] for i in range(0, len(unique_dois), batch_size)
    ]
    for doi_batch in tqdm(doi_batches, desc="Ingesting batches", total=len(doi_batches)):
        with Session(engine) as session:
            document_ids_by_doi = _bulk_lookup_document_ids_by_doi(doi_batch, session)

            candidates: list[db_models.DocumentSoftwareMention] = []
            for doi in doi_batch:
                document_id = document_ids_by_doi.get(doi)

                # Handle fast exit
                if document_id is not None:
                    matched_count += 1
                else:
                    skipped_count += 1
                    continue

                # Collect all mentions for this DOI
                for row in mentions_by_doi[(doi,)].iter_rows(named=True):
                    software_raw: str = row["software_raw"]
                    mention_id: str = row["software_mention_id"]
                    mention_context: str = row["context_full_text"]

                    candidates.append(
                        db_models.DocumentSoftwareMention(
                            document_id=document_id,
                            software_name=software_raw,
                            software_name_normalized=normalize_name(software_raw),
                            softcite_mention_id=mention_id,
                            mention_context=mention_context,
                        )
                    )

            _bulk_get_or_add_mentions(candidates, session)
            session.commit()

    print(f"Done. Matched: {matched_count} | Skipped (no DB match): {skipped_count}")


###############################################################################


def _collect_batch_results(
    batch_futures: list,
    batch: list[UnprocessedRepository],
) -> list[RepoExtractionResult | types.ErrorResult]:
    results = []
    for future, repo in zip(batch_futures, batch, strict=False):
        try:
            results.append(future.result())
        except TimeoutError:
            repo_full_name = f"{repo.owner}/{repo.name}"
            print(f"Timeout reached for {repo_full_name}, skipping.")
            results.append(
                types.ErrorResult(
                    source="used-software-extraction",
                    step="task_timeout",
                    identifier=repo_full_name,
                    error="Task timed out",
                    traceback="",
                )
            )
    return results


def _print_language_counts(label: str, counts: "Counter[str]") -> None:
    """Print a language-grouped count table."""
    print(f"\n--- {label} (by language) ---")
    for lang, count in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {lang}: {count}")
    print(f"  TOTAL: {sum(counts.values())}")


def _filter_cached_errors(
    repos: list[UnprocessedRepository],
    errors_cache_path: Path,
    timeout_seconds: int,
) -> tuple[list[UnprocessedRepository], "Counter[str]"]:
    """Filter out repos that already errored, returning filtered list and error counts."""
    errored_by_language: Counter[str] = Counter()
    if not errors_cache_path.exists():
        return repos, errored_by_language

    errors_df = pl.read_parquet(errors_cache_path)
    already_errored = set(
        errors_df.filter(pl.col("timeout_seconds") == timeout_seconds)
        .get_column("identifier")
        .to_list()
    )
    filtered = []
    for r in repos:
        if f"{r.owner}/{r.name}" in already_errored:
            errored_by_language[r.language] += 1
        else:
            filtered.append(r)
    return filtered, errored_by_language


@flow(log_prints=True)
def _used_software_extraction_flow(
    database_path: str,
    language_filter: list[str],
    use_coiled: bool,
    coiled_region: str,
    coiled_workers: int,
    batch_size: int,
    limit: int | None,
    timeout_seconds: int,
    errors_cache_file: str,
) -> None:
    # Construct the actual full path of the errors cache file
    errors_cache_path = Path(errors_cache_file).resolve()

    # Print dataset and coiled status
    print("-" * 80)
    print("Pipeline Options:")
    print(f"Database Path: {database_path}")
    print(f"Language Filter: {language_filter}")
    print(f"Use Coiled: {use_coiled}")
    print(f"Coiled Region: {coiled_region}")
    print(f"Coiled Workers: {coiled_workers}")
    # print(f"Batch Size: {batch_size}")
    print(f"Total Processing Limit: {limit}")
    print(f"Task Timeout (seconds): {timeout_seconds}")
    print(f"Errors Cache File: {errors_cache_path}")
    print("-" * 80)

    # Get list of repos to process
    print("Retrieving list of repositories to process...")
    repos, processed_by_language = _query_unprocessed_repositories(
        database_path=database_path,
        language_filter=language_filter,
    )
    repos = repos[:limit] if limit is not None else repos

    _print_language_counts("Already Processed", processed_by_language)

    # Filter out repos that already errored under the same timeout configuration
    repos, errored_by_language = _filter_cached_errors(
        repos, errors_cache_path, timeout_seconds
    )
    if errored_by_language:
        _print_language_counts(
            f"Cached Errors (timeout={timeout_seconds}s)", errored_by_language
        )

    remaining_by_language: Counter[str] = Counter(r.language for r in repos)
    _print_language_counts("Remaining to Process", remaining_by_language)
    print("-" * 80)

    # Handle no repos to process
    if not repos:
        print("No repositories to process. Exiting.")
        return

    # Shuffle the repos so that we get a mix of languages in each batch
    random.shuffle(repos)

    # Construct extract tasks — warmup task uses a longer timeout to absorb
    # cold-start cluster provisioning (~5-8 min for t4g.large spot instances).
    extract_warmup_task, extract_task = _wrap_func_with_coiled_prefect_task(
        _extract_repo_imports_and_deps,
        coiled_func_name="extract_repo_imports_and_deps",
        coiled_kwargs=_get_small_cpu_api_cluster(
            n_workers=coiled_workers,
            use_coiled=use_coiled,
            coiled_region=coiled_region,
        ),
        timeout_seconds=timeout_seconds,
        warmup_timeout_seconds=max(timeout_seconds, 600),
    )

    # Process in batches
    batches = [repos[i : i + batch_size] for i in range(0, len(repos), batch_size)]
    for batch_idx, batch in enumerate(
        tqdm(batches, desc="Processing batches", total=len(batches))
    ):
        task_fn = extract_warmup_task if batch_idx == 0 else extract_task
        batch_futures = task_fn.map(
            repository_id=[r.repo_id for r in batch],
            owner=[r.owner for r in batch],
            name=[r.name for r in batch],
        )
        batch_results = _collect_batch_results(batch_futures, batch)

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
            try:
                _store_repo_result(result=result, database_path=database_path)
            except UnicodeEncodeError:
                print(f"Skipping repo {result.owner}/{result.name} due to UnicodeEncodeError")

        # Store errors to parquet file
        if batch_errors:
            # Read in existing errors if the file already exists,
            # otherwise start with an empty list
            if errors_cache_path.exists():
                existing_errors = pl.read_parquet(errors_cache_path)
            else:
                existing_errors = pl.DataFrame()

            # Convert new errors to a DataFrame and concatenate with existing errors
            new_errors_df = pl.DataFrame(
                [{**e.to_dict(), "timeout_seconds": timeout_seconds} for e in batch_errors]
            )
            combined_errors = pl.concat([existing_errors, new_errors_df])

            # Write combined errors back to the parquet file
            combined_errors.write_parquet(errors_cache_path)


def _print_extraction_result(result: RepoExtractionResult) -> None:
    """Print extraction results to stdout."""
    print(f"\n{'=' * 60}")
    print(f"IMPORTS ({len(result.imports)} found):")
    print(f"{'=' * 60}")
    for imp in result.imports:
        norm = normalize_name(imp.software_name)
        print(f"  {imp.software_name} (normalized: {norm})")
        if imp.file_paths:
            for fp in imp.file_paths.split(";"):
                print(f"    -> {fp}")

    print(f"\n{'=' * 60}")
    print(f"DEPENDENCIES ({len(result.dependencies)} found):")
    print(f"{'=' * 60}")
    for dep in result.dependencies:
        norm = normalize_name(dep.software_name)
        print(f"  {dep.software_name} (normalized: {norm})")
        print(f"    ecosystem: {dep.ecosystem}, type: {dep.dependency_type}")
        if dep.version_spec:
            print(f"    version: {dep.version_spec}")
        if dep.manifest_paths:
            print(f"    manifests: {dep.manifest_paths}")


def _dry_run_single_repo(repo_spec: str, database_path: str) -> None:
    """Dry-run extraction for a single repo (owner/name). Prints results, saves nothing."""
    owner, name = repo_spec.split("/", 1)

    engine = db_utils.get_engine(database_path=database_path)
    with Session(engine) as session:
        stmt = select(db_models.Repository).where(
            col(db_models.Repository.owner) == owner.lower(),
            col(db_models.Repository.name) == name.lower(),
        )
        repo = session.exec(stmt).first()
        if repo is None:
            print(f"Repository {repo_spec} not found in database")
            return
        if repo.id is None:
            print(f"Repository {repo_spec} has no ID in database")
            return
        repo_id: int = repo.id
        print(f"Found repository: id={repo_id}, language={repo.primary_language}")

    print(f"\nExtracting imports and dependencies for {repo_spec}...")
    result = _extract_repo_imports_and_deps(
        repository_id=repo_id,
        owner=owner,
        name=name,
    )

    if isinstance(result, types.ErrorResult):
        print(f"\nERROR during '{result.step}':")
        print(f"  {result.error}")
        if result.traceback:
            print(f"  Traceback:\n{result.traceback}")
        return

    _print_extraction_result(result)
    print("\nDry run complete. Nothing saved to database.")


@app.command()
def used_software_extraction(
    database_path: str = typer.Argument(
        help="Path to the SQLite database file to use.",
    ),
    use_coiled: bool = False,
    coiled_region: str = "us-west-2",
    coiled_workers: int = 48,
    language_filter: list[str] | None = None,
    batch_size: int = 64,
    limit: int | None = None,
    timeout_seconds: int = 60,
    errors_cache_file: str = "used-software-extraction-errors.parquet",
    dry_run_repo: str | None = None,
) -> None:
    """
    Extract software imports and dependencies from document-linked repositories.

    Default language filter is Python, Jupyter Notebook, and R.
    Repos that exceed --timeout-seconds are skipped and remain unprocessed,
    so a subsequent run with a larger timeout will pick them up automatically.

    Use --dry-run-repo owner/name to extract a single repo without saving results.
    """
    if dry_run_repo:
        _dry_run_single_repo(dry_run_repo, database_path=database_path)
        return

    _used_software_extraction_flow(
        database_path=database_path,
        language_filter=language_filter
        if language_filter is not None
        else DEFAULT_LANGUAGE_FILTER,
        use_coiled=use_coiled,
        coiled_region=coiled_region,
        coiled_workers=coiled_workers,
        batch_size=batch_size,
        limit=limit,
        timeout_seconds=timeout_seconds,
        errors_cache_file=errors_cache_file,
    )


def main() -> None:
    app()
