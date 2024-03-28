#!/usr/bin/env python

from __future__ import annotations

import logging
from pathlib import Path

from allofplos.corpus.plos_corpus import get_corpus_dir
from allofplos.update import main as get_latest_plos_corpus

from .proto import DataSource, RepositoryDocumentPair

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class PLOSDataSource(DataSource):
    @staticmethod
    def _get_plos_xmls() -> list[Path]:
        # Download all plos files
        get_latest_plos_corpus()

        # Get the corpus dir
        corpus_dir = get_corpus_dir()

        # Get all files
        return list(Path(corpus_dir).resolve(strict=True).glob("*.xml"))

    @staticmethod
    def get_dataset(
        **kwargs: dict[str, str],
    ) -> list[RepositoryDocumentPair]:
        """Download the PLOS dataset."""
        raise NotImplementedError()
