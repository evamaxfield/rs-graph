#!/usr/bin/env python

from __future__ import annotations

import logging
import random
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: N817

from allofplos.corpus.plos_corpus import get_corpus_dir
from allofplos.update import main as get_latest_plos_corpus
from tqdm import tqdm

from ..utils.dask_functions import process_func
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
    def _get_article_repository_url(root: ET.Element) -> str | None:
        # Before anything else, check for data availability
        # Example:
        # <custom-meta id="data-availability">
        #     <meta-name>Data Availability</meta-name>
        #     <meta-value>A TimeTeller R package is available from GitHub at <ext-link ext-link-type="uri" xlink:href="https://github.com/VadimVasilyev1994/TimeTeller" xlink:type="simple">https://github.com/VadimVasilyev1994/TimeTeller</ext-link>. All data used except that of Bjarnasson et al. is publicly available from the sites given in the Supplementary Information. The Bjarnasson et al. data that is used is available with the R package.</meta-value>  # noqa: E501
        # </custom-meta>
        data_availability = root.find(".//custom-meta[@id='data-availability']")
        if data_availability is None:
            return None

        # Check if data availablity points to code host
        ext_link = data_availability.find(".//ext-link[@ext-link-type='uri']")
        if ext_link is None or ext_link.text is None:
            return None

        # Get the url
        return ext_link.text

    @staticmethod
    def _get_article_doi(root: ET.Element) -> str | None:
        # Get the DOI
        # Example:
        # <article-id pub-id-type="doi">10.1371/journal.pntd.0002114</article-id>
        doi_container = root.find(".//article-id[@pub-id-type='doi']")
        if doi_container is None or doi_container.text is None:
            return None

        return doi_container.text

    @staticmethod
    def _process_xml(jats_xml_filepath: Path) -> RepositoryDocumentPair | None:
        # Load the XML
        try:
            tree = ET.parse(jats_xml_filepath)
            root = tree.getroot()
        except ET.ParseError as e:
            log.error(f"Error parsing XML file: '{jats_xml_filepath}': {e}")
            return None

        # Get the repository URL
        repo_url = PLOSDataSource._get_article_repository_url(root)
        if repo_url is None:
            return None

        # Get the journal info
        paper_doi = PLOSDataSource._get_article_doi(root)
        if paper_doi is None:
            return None

        # Add to successful results
        return RepositoryDocumentPair(
            source="plos",
            repo_url=repo_url,
            paper_doi=paper_doi,
        )

    @staticmethod
    def get_dataset(
        use_dask: bool = False,
        **kwargs: dict[str, str],
    ) -> list[RepositoryDocumentPair]:
        """Download the PLOS dataset."""
        # Get all PLOS XMLs
        plos_xmls = random.sample(PLOSDataSource._get_plos_xmls(), 5000)

        # Parse each XML
        if use_dask:
            results = process_func(
                name="plos-jats-xml-processing",
                func=PLOSDataSource._process_xml,
                func_iterables=[plos_xmls],
                cluster_kwargs={
                    "processes": True,
                    "n_workers": 4,
                    "threads_per_worker": 1,
                },
            )
        else:
            results = [
                PLOSDataSource._process_xml(plos_xml)
                for plos_xml in tqdm(plos_xmls, desc="Processing PLOS XMLs")
            ]

        # Filter out None results
        successful_results = [result for result in results if result is not None]

        # Get count of total processed and errored
        total_errored = len(plos_xmls) - len(successful_results)

        # Log total processed and errored
        log.info(f"Total succeeded: {len(successful_results)}")
        log.info(f"Total errored: {total_errored}")

        # Return filepath
        return successful_results
