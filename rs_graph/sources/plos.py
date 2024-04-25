#!/usr/bin/env python

from __future__ import annotations

import traceback
from pathlib import Path
from xml.etree import ElementTree as ET  # noqa: N817

from allofplos.corpus.plos_corpus import get_corpus_dir
from allofplos.update import main as get_latest_plos_corpus
from tqdm import tqdm

from .. import types

###############################################################################


def _get_plos_xmls() -> list[Path]:
    # Download all plos files
    get_latest_plos_corpus()

    # Get the corpus dir
    corpus_dir = get_corpus_dir()

    # Get all files
    return list(Path(corpus_dir).resolve(strict=True).glob("*.xml"))


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


def _get_article_doi(root: ET.Element) -> str | None:
    # Get the DOI
    # Example:
    # <article-id pub-id-type="doi">10.1371/journal.pntd.0002114</article-id>
    doi_container = root.find(".//article-id[@pub-id-type='doi']")
    if doi_container is None or doi_container.text is None:
        return None

    return doi_container.text


def _process_xml(
    jats_xml_filepath: Path,
) -> types.BasicRepositoryDocumentPair | types.ErrorResult:
    # Load the XML
    try:
        tree = ET.parse(jats_xml_filepath)
        root = tree.getroot()
    except ET.ParseError as e:
        return types.ErrorResult(
            source="plos",
            step="plos-xml-processing",
            identifier=jats_xml_filepath.name,
            error=str(e),
            traceback=traceback.format_exc(),
        )

    # Get the repository URL
    repo_url = _get_article_repository_url(root)
    if repo_url is None:
        return types.ErrorResult(
            source="plos",
            step="plos-xml-processing",
            identifier=jats_xml_filepath.name,
            error="No Repository URL",
            traceback="",
        )

    # Get the journal info
    paper_doi = _get_article_doi(root)
    if paper_doi is None:
        return types.ErrorResult(
            source="plos",
            step="plos-xml-processing",
            identifier=jats_xml_filepath.name,
            error="No Paper DOI",
            traceback="",
        )

    # Add to successful results
    return types.BasicRepositoryDocumentPair(
        source="plos",
        repo_url=repo_url,
        paper_doi=paper_doi,
    )


def get_dataset(
    **kwargs: dict[str, str],
) -> types.SuccessAndErroredResultsLists:
    """Download the PLOS dataset."""
    # Get all PLOS XMLs
    plos_xmls = _get_plos_xmls()

    results = [
        _process_xml(plos_xml)
        for plos_xml in tqdm(
            plos_xmls,
            desc="Processing PLOS XMLs",
        )
    ]

    # Create successful results and errored results lists
    successful_results = [
        r for r in results if isinstance(r, types.BasicRepositoryDocumentPair)
    ]
    errored_results = [r for r in results if isinstance(r, types.ErrorResult)]

    # Log total processed and errored
    print(f"Total succeeded: {len(successful_results)}")
    print(f"Total errored: {len(errored_results)}")

    # Return filepath
    return types.SuccessAndErroredResultsLists(
        successful_results=successful_results,
        errored_results=errored_results,
    )
