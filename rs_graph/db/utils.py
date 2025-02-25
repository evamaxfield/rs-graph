#!/usr/bin/env python

import traceback
from pathlib import Path

from prefect import task
from sqlalchemy.engine import Engine
from sqlmodel import Session, SQLModel, UniqueConstraint, create_engine, select
from tqdm import tqdm

from .. import types
from . import models as db_models

###############################################################################


def get_engine(use_prod: bool = False) -> Engine:
    db_path = Path(__file__).parent.parent / "data" / "files"

    if use_prod:
        db_path = db_path / "rs-graph-v1-prod.db"

        return create_engine(f"sqlite:///{db_path}")
    else:
        db_path = db_path / "rs-graph-v1-dev.db"

        return create_engine(f"sqlite:///{db_path}")


def get_unique_first_model(model: SQLModel, session: Session) -> SQLModel | None:
    # Get model class
    model_cls = model.__class__

    # Get constrained fields
    all_table_contraints = model_cls.__table__.constraints
    unique_constraints = [
        constraint
        for constraint in all_table_contraints
        if isinstance(constraint, UniqueConstraint)
    ]

    # Unpack constraints to just a list (converted from set) of all of the field names
    # in each constraint
    required_fields = list(
        {
            field.name
            for constraint in unique_constraints
            for field in constraint.columns
        }
    )

    # Try and select a matching model
    query = select(model_cls)
    for field in required_fields:
        query = query.where(getattr(model_cls, field) == getattr(model, field))

    # Execute the query
    return session.exec(query).first()


def _get_or_add_and_flush(
    model: SQLModel,
    session: Session,
) -> SQLModel:
    # Try getting matching model
    existing_model = get_unique_first_model(model=model, session=session)

    # If model exists, return it
    if existing_model is not None:
        return existing_model

    # Otherwise, add and flush
    session.add(model)
    session.flush()

    return model


def store_full_details(
    pair: types.ExpandedRepositoryDocumentPair,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Create a session
    with Session(engine) as session:
        try:
            assert pair.open_alex_results is not None
            assert pair.github_results is not None
            assert pair.repo_parts is not None

            # Dataset source
            pair.open_alex_results.source_model = _get_or_add_and_flush(
                model=pair.open_alex_results.source_model, session=session
            )
            assert pair.open_alex_results.source_model.id is not None

            # Document (update dataset source)
            pair.open_alex_results.document_model = _get_or_add_and_flush(
                model=pair.open_alex_results.document_model, session=session
            )
            assert pair.open_alex_results.document_model.id is not None

            # Store the abstract
            pair.open_alex_results.document_abstract_model.document_id = (
                pair.open_alex_results.document_model.id
            )
            pair.open_alex_results.document_abstract_model = _get_or_add_and_flush(
                model=pair.open_alex_results.document_abstract_model,
                session=session,
            )

            # Store alternate DOIs
            for alternate_doi in pair.open_alex_results.document_alternate_dois:
                # Create new model
                alternate_doi_model = db_models.DocumentAlternateDOI(
                    document_id=pair.open_alex_results.document_model.id,
                    doi=alternate_doi,
                )

                # Persist
                alternate_doi_model = _get_or_add_and_flush(
                    model=alternate_doi_model,
                    session=session,
                )

            # Topic details
            for topic_detail in pair.open_alex_results.topic_details:
                # Topic model
                topic_detail.topic_model = _get_or_add_and_flush(
                    model=topic_detail.topic_model, session=session
                )
                assert topic_detail.topic_model.id is not None

                # Document topic model
                topic_detail.document_topic_model.document_id = (
                    pair.open_alex_results.document_model.id
                )
                topic_detail.document_topic_model.topic_id = topic_detail.topic_model.id
                topic_detail.document_topic_model = _get_or_add_and_flush(
                    model=topic_detail.document_topic_model, session=session
                )
                assert topic_detail.document_topic_model.id is not None

            # Researcher details
            for researcher_detail in pair.open_alex_results.researcher_details:
                # Researcher model
                researcher_detail.researcher_model = _get_or_add_and_flush(
                    model=researcher_detail.researcher_model, session=session
                )
                assert researcher_detail.researcher_model.id is not None

                # Document contributor model
                researcher_detail.document_contributor_model.document_id = (
                    pair.open_alex_results.document_model.id
                )
                researcher_detail.document_contributor_model.researcher_id = (
                    researcher_detail.researcher_model.id
                )
                researcher_detail.document_contributor_model = _get_or_add_and_flush(
                    model=researcher_detail.document_contributor_model, session=session
                )
                assert researcher_detail.document_contributor_model.id is not None

                # Institution models
                for institution_model in researcher_detail.institution_models:
                    institution_model = _get_or_add_and_flush(
                        model=institution_model, session=session
                    )
                    assert institution_model.id is not None

                    # Create all of the researcher document institution models
                    r_d_i = db_models.DocumentContributorInstitution(
                        document_contributor_id=researcher_detail.document_contributor_model.id,
                        institution_id=institution_model.id,
                    )
                    _get_or_add_and_flush(model=r_d_i, session=session)

            # Funding instance details
            for (
                funding_instance_detail
            ) in pair.open_alex_results.funding_instance_details:
                # Funder model
                funding_instance_detail.funder_model = _get_or_add_and_flush(
                    model=funding_instance_detail.funder_model, session=session
                )
                assert funding_instance_detail.funder_model.id is not None

                # Funding instance model
                funding_instance_detail.funding_instance_model.funder_id = (
                    funding_instance_detail.funder_model.id
                )
                funding_instance_detail.funding_instance_model = _get_or_add_and_flush(
                    model=funding_instance_detail.funding_instance_model,
                    session=session,
                )
                assert funding_instance_detail.funding_instance_model.id is not None

                # Create the document funding instance models
                d_f_i = db_models.DocumentFundingInstance(
                    document_id=pair.open_alex_results.document_model.id,
                    funding_instance_id=funding_instance_detail.funding_instance_model.id,
                )
                _get_or_add_and_flush(model=d_f_i, session=session)

            # Code host
            pair.github_results.code_host_model = _get_or_add_and_flush(
                model=pair.github_results.code_host_model, session=session
            )
            assert pair.github_results.code_host_model.id is not None

            # Repository
            pair.github_results.repository_model.code_host_id = (
                pair.github_results.code_host_model.id
            )
            pair.github_results.repository_model = _get_or_add_and_flush(
                model=pair.github_results.repository_model, session=session
            )
            assert pair.github_results.repository_model.id is not None

            # Store the README
            pair.github_results.repository_readme_model.repository_id = (
                pair.github_results.repository_model.id
            )
            pair.github_results.repository_readme_model = _get_or_add_and_flush(
                model=pair.github_results.repository_readme_model, session=session
            )

            # Repository languages
            for repo_language in pair.github_results.repository_language_models:
                repo_language.repository_id = pair.github_results.repository_model.id
                repo_language = _get_or_add_and_flush(
                    model=repo_language, session=session
                )

            # Repository contributor details
            for (
                repo_contributor_detail
            ) in pair.github_results.repository_contributor_details:
                # Developer account
                repo_contributor_detail.developer_account_model.code_host_id = (
                    pair.github_results.code_host_model.id
                )
                repo_contributor_detail.developer_account_model = _get_or_add_and_flush(
                    model=repo_contributor_detail.developer_account_model,
                    session=session,
                )
                assert repo_contributor_detail.developer_account_model.id is not None

                # Repository contributor
                repo_contributor_detail.repository_contributor_model.repository_id = (
                    pair.github_results.repository_model.id
                )
                repo_contributor_detail.repository_contributor_model.developer_account_id = (  # noqa: E501
                    repo_contributor_detail.developer_account_model.id
                )
                repo_contributor_detail.repository_contributor_model = (
                    _get_or_add_and_flush(
                        model=repo_contributor_detail.repository_contributor_model,
                        session=session,
                    )
                )

            # Create a connection between the document and the repository
            d_r = db_models.DocumentRepositoryLink(
                document_id=pair.open_alex_results.document_model.id,
                repository_id=pair.github_results.repository_model.id,
                dataset_source_id=pair.open_alex_results.source_model.id,
            )
            _get_or_add_and_flush(model=d_r, session=session)

            # Get all the info needed to refresh after commit
            dataset_source_id = pair.open_alex_results.source_model.id
            document_model_id = pair.open_alex_results.document_model.id
            repository_model_id = pair.github_results.repository_model.id
            developer_account_model_ids = [
                repo_contributor_detail.developer_account_model.id
                for repo_contributor_detail in (
                    pair.github_results.repository_contributor_details
                )
            ]
            researcher_model_ids = [
                researcher_detail.researcher_model.id
                for researcher_detail in pair.open_alex_results.researcher_details
            ]

            session.commit()

            # Get all the models after they have been added
            dataset_source_stmt = select(db_models.DatasetSource).where(
                db_models.DatasetSource.id == dataset_source_id
            )
            dataset_source_model = session.exec(dataset_source_stmt).first()
            document_stmt = select(db_models.Document).where(
                db_models.Document.id == document_model_id
            )
            document_model = session.exec(document_stmt).first()
            assert document_model is not None
            repository_stmt = select(db_models.Repository).where(
                db_models.Repository.id == repository_model_id
            )
            repository_model = session.exec(repository_stmt).first()
            assert repository_model is not None
            developer_account_stmt = select(db_models.DeveloperAccount).where(
                db_models.DeveloperAccount.id.in_(  # type: ignore
                    developer_account_model_ids
                )
            )
            developer_account_models = session.exec(developer_account_stmt).all()
            researcher_stmt = select(db_models.Researcher).where(
                db_models.Researcher.id.in_(researcher_model_ids)  # type: ignore
            )
            researcher_models = session.exec(researcher_stmt).all()

            # Create the stored pair
            stored_pair = types.StoredRepositoryDocumentPair(
                dataset_source_model=dataset_source_model,
                document_model=document_model,
                repository_model=repository_model,
                developer_account_models=developer_account_models,
                researcher_models=researcher_models,
            )

            return stored_pair

        except Exception as e:
            session.rollback()
            return types.ErrorResult(
                source=pair.source,
                step="store-full-details",
                identifier=pair.paper_doi,
                error=str(e),
                traceback=traceback.format_exc(),
            )


@task(
    log_prints=True,
    retries=2,
    retry_delay_seconds=2,
)
def store_full_details_task(
    pair: types.ExpandedRepositoryDocumentPair | types.ErrorResult,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    return store_full_details(pair=pair, use_prod=use_prod)


def store_dev_researcher_em_links(
    pair: types.StoredRepositoryDocumentPair,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Create a session
    with Session(engine) as session:
        try:
            assert pair.researcher_developer_links is not None

            # Iter over each linked dev researcher link and add to the database
            link_ids = []
            for linked_dev_researcher_pair in pair.researcher_developer_links:
                linked_dev_researcher_pair = _get_or_add_and_flush(
                    model=linked_dev_researcher_pair, session=session
                )
                link_ids.append(linked_dev_researcher_pair.id)

            # Commit
            session.commit()

            # Get all the models after they have been added
            linked_dev_researcher_stmt = select(
                db_models.ResearcherDeveloperAccountLink
            ).where(
                db_models.ResearcherDeveloperAccountLink.id.in_(  # type: ignore
                    link_ids
                )
            )
            linked_dev_researcher_models = session.exec(
                linked_dev_researcher_stmt
            ).all()

            # Attach to model
            pair.researcher_developer_links = linked_dev_researcher_models

            return pair

        except Exception as e:
            session.rollback()
            return types.ErrorResult(
                source=pair.dataset_source_model.name,
                step="store-dev-research-em-links",
                identifier=pair.document_model.doi,
                error=str(e),
                traceback=traceback.format_exc(),
            )


@task(
    log_prints=True,
    retries=2,
    retry_delay_seconds=2,
)
def store_dev_researcher_em_links_task(
    pair: types.StoredRepositoryDocumentPair | types.ErrorResult,
    use_prod: bool = False,
) -> types.StoredRepositoryDocumentPair | types.ErrorResult:
    if isinstance(pair, types.ErrorResult):
        return pair

    return store_dev_researcher_em_links(pair=pair, use_prod=use_prod)


def check_pair_exists(
    pair: types.ExpandedRepositoryDocumentPair,
    use_prod: bool = False,
) -> bool:
    # Get the engine
    engine = get_engine(use_prod=use_prod)

    # Check we have already processed to repo parts
    assert pair.repo_parts is not None

    # Create a session
    with Session(engine) as session:
        # Five statements
        # One to check and get the document
        # One to check for backup alternate dois
        # One to get the repo host id
        # One to check and get the repository
        # And a last to check and get the link

        # Document
        document_stmt = select(db_models.Document).where(
            db_models.Document.doi == pair.paper_doi
        )
        document_model = session.exec(document_stmt).first()

        # Check for document alternate doi
        if document_model is None:
            document_stmt = select(db_models.DocumentAlternateDOI).where(
                db_models.DocumentAlternateDOI.doi == pair.paper_doi
            )
            document_alternate_doi_model = session.exec(document_stmt).first()

            # Fast fail
            if document_alternate_doi_model is None:
                return False

            # Else get updated document model
            else:
                document_stmt = select(db_models.Document).where(
                    db_models.Document.id == document_alternate_doi_model.document_id
                )
                document_model = session.exec(document_stmt).first()

        # Code Host
        code_host_stmt = select(db_models.CodeHost).where(
            db_models.CodeHost.name == pair.repo_parts.host
        )
        code_host_model = session.exec(code_host_stmt).first()

        # Fast fail
        if code_host_model is None:
            return False

        # Repository
        repository_stmt = select(db_models.Repository).where(
            db_models.Repository.code_host_id == code_host_model.id,
            db_models.Repository.owner == pair.repo_parts.owner,
            db_models.Repository.name == pair.repo_parts.name,
        )
        repository_model = session.exec(repository_stmt).first()

        # Fast fail
        if repository_model is None:
            return False

        # Link
        link_stmt = select(db_models.DocumentRepositoryLink).where(
            db_models.DocumentRepositoryLink.document_id == document_model.id,
            db_models.DocumentRepositoryLink.repository_id == repository_model.id,
        )
        link_model = session.exec(link_stmt).first()

        # Final check
        return link_model is not None


def filter_stored_pairs(
    pairs: list[types.ExpandedRepositoryDocumentPair],
    use_prod: bool = False,
) -> list[types.ExpandedRepositoryDocumentPair]:
    # For each pair, check if it exists in the database
    unprocessed_pairs = [
        pair
        for pair in tqdm(
            pairs,
            desc="Filtering already stored pairs",
        )
        if not check_pair_exists(pair=pair, use_prod=use_prod)
    ]

    # Log remaining pairs and filtered out pairs
    print(f"Filtered out pairs: {len(pairs) - len(unprocessed_pairs)}")
    print(f"Remaining pairs: {len(unprocessed_pairs)}")

    return unprocessed_pairs
