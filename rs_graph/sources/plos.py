#!/usr/bin/env python

from __future__ import annotations

import logging
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: N817

from allofplos.corpus.plos_corpus import get_corpus_dir
from allofplos.update import main as get_latest_plos_corpus
from tqdm import tqdm

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
    def get_dataset(
        **kwargs: dict[str, str],
    ) -> list[RepositoryDocumentPair]:
        """Download the PLOS dataset."""
        # Store successful results
        successful_results = []

        # Get all PLOS XMLs
        plos_xmls = PLOSDataSource._get_plos_xmls()

        # Parse each XML
        for jats_xml in tqdm(
            plos_xmls,
            desc="Processing PLOS JATS XMLs",
        ):
            # Load the XML
            try:
                tree = ET.parse(jats_xml)
                root = tree.getroot()
            except ET.ParseError as e:
                log.error(f"Error parsing XML file: '{jats_xml}': {e}")
                continue

            # Get the repository URL
            repo_url = PLOSDataSource._get_article_repository_url(root)
            if repo_url is None:
                continue

            # Get the journal info
            paper_doi = PLOSDataSource._get_article_doi(root)
            if paper_doi is None:
                continue

            # Add to successful results
            successful_results.append(
                RepositoryDocumentPair(
                    source="plos",
                    repo_url=repo_url,
                    paper_doi=paper_doi,
                )
            )

        # Get count of total processed and errored
        total_processed = len(successful_results)
        total_errored = total_processed - len(successful_results)

        # Log total processed and errored
        log.info(f"Total succeeded: {len(successful_results)}")
        log.info(f"Total errored: {total_errored}")

        # Return filepath
        return successful_results
