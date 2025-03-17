#!/usr/bin/env python

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any
import traceback
import logging
import json
import shutil

import docker
import requests
from software_mentions_client.client import software_mentions_client as SoftwareMentionsClient

from .. import types

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

DEFAULT_GROBID_IMAGE = "grobid/software-mentions:0.8.0"
DEFAULT_GROBID_PORT = 8060
DEFAULT_GROBID_CONFIG_PATH = (
    Path(__file__).parent / "support" / "grobid-soft-cite-config.json"
)

###############################################################################

def _get_download_url_from_open_alex(
    doi: str,
) -> str | None:
    pass

def _get_download_url_from_semantic_scholar(
    doi: str,
) -> str | None:
    pass

def setup_or_connect_to_server(  # noqa: C901
    image: str | None = None,
    port: int | None = None,
) -> tuple[SoftwareMentionsClient | None, docker.models.containers.Container]:
    """
    Set up or create a connection to a GROBID server.

    Parameters
    ----------
    image: Optional[str]
        The Docker image name to use for pulling (if needed)
        and creating the container.
        Default: None (check environment variables
        for GROBID_IMAGE or else use default image)
    port: Optional[int]
        The port to make available on the GROBID server and Docker container.
        Default: None (check environment variables
        for GROBID_PORT or else use default default port)

    Returns
    -------
    Optional[GrobidClient]
        If all setup and/or connection was successful, the GROBID client connection.
        None if something went wrong.
    docker.models.containers.Container
        The Docker container object for future management.
    """
    # Handle vars
    if image is None:
        if "GROBID_IMAGE" in os.environ:
            log.debug("Using GROBID_IMAGE from environment vars.")
            image = os.environ["GROBID_IMAGE"]
        else:
            image = DEFAULT_GROBID_IMAGE
    if port is None:
        if "GROBID_PORT" in os.environ:
            log.debug("Using GROBID_PORT from environment vars.")
            port = int(os.environ["GROBID_PORT"])
        else:
            port = DEFAULT_GROBID_PORT

    # Connect to Docker client
    docker_client = docker.from_env()

    # Check if the image is already downloaded
    try:
        docker_client.images.get(image)
    except docker.errors.ImageNotFound:
        log.info(
            f"Pulling GROBID image: '{image}' "
            f"(this will only happen the first time you run Papers without Code)."
        )
        docker_client.images.pull(image)

    # Check for already running image
    # Connect or start again
    found_running_grobid_image = False
    for container in docker_client.containers.list():
        if container.image.tags[0] == DEFAULT_GROBID_IMAGE:
            if container.status == "running":
                found_running_grobid_image = True
                break
            else:
                log.debug(
                    f"Found stopped GROBID container "
                    f"('{container.short_id}') -- Starting."
                )
                container.start()
                time.sleep(5)
                found_running_grobid_image = True
                break

    # Start new container
    if not found_running_grobid_image:
        log.info("Setting up PDF parsing server.")
        log.debug(f"Using GROBID image: '{image}'.")
        container = docker_client.containers.run(
            image,
            ports={"8060/tcp": port},
            detach=True,
        )
        log.debug(f"Started GROBID container: '{container.short_id}'.")
        time.sleep(5)

    # Attempt to connect with client
    try:
        # Make request to check the server is alive
        server_url = f"http://127.0.0.1:{port}"
        response = requests.get(f"{server_url}/api/isalive")
        response.raise_for_status()
        log.debug(f"GROBID API available at: '{server_url}'.")

        # Create a GROBID Client
        software_mentions_client = SoftwareMentionsClient(
            config_path=str(DEFAULT_GROBID_CONFIG_PATH),
        )

        return software_mentions_client, container

    except Exception as e:
        log.error(f"Failed to connect to GROBID server, error: '{e}'.")
        return None, container


def teardown_server(
    container: docker.models.containers.Container,
) -> None:
    """
    Stop and remove a Docker container.

    Parameters
    ----------
    container: docker.models.containers.Container
        The docker container to stop and remove.
    """
    # Stop the container
    container.stop()
    container.remove()
    log.info(f"Stopped and removed Docker container: '{container.short_id}'.")


def process_pdf(
    client: SoftwareMentionsClient,
    document_grant_info: types.DocumentWithGrantInformation,
    pdf_path: str,
) -> dict[str, Any]:
    """
    Process a PDF file using a GROBID client.

    Parameters
    ----------
    client: GrobidClient
        A GROBID client to use for processing.
    pdf_path: str
        The path to the local PDF file to process.

    Returns
    -------
    Dict[str, Any]
        Paper data.

    Raises
    ------
    ValueError
        Something went wrong during processing.
    """
    # Convert to Path
    pdf_path = Path(pdf_path)
    pdf_path = pdf_path.resolve()
    if not pdf_path.exists():
        raise FileNotFoundError(f"Provided file does not exist: '{pdf_path}'")
    if pdf_path.is_dir():
        raise IsADirectoryError(
            f"Parsing currently only supports single files. "
            f"Provided path is a directory: '{pdf_path}'"
        )
    
    # Temporary output path
    tmp_output_path = "tmp-output.json"
    if Path(tmp_output_path).exists():
        Path(tmp_output_path).unlink()

    # Annotate PDF
    client.annotate(
        file_in=pdf_path,
        file_out=tmp_output_path,
        full_record=None,
    )

    # Read the JSON
    with open(tmp_output_path, "r") as f:
        annotations = json.load(f)

    # Parse the annotations
    results = []
    for mention in annotations["mentions"]:
        # Find cls with max value
        grobid_intent_details = mention["mentionContextAttributes"]
        best_cls = "other"
        best_score = 0.0
        for intent_cls in ["used", "created", "shared"]:
            if (
                grobid_intent_details[intent_cls]["value"]
                and grobid_intent_details[intent_cls]["score"] > best_score
            ):
                best_cls = intent_cls
                best_score = grobid_intent_details[intent_cls]["score"]

        # Add to results
        results.append(
            types.DocumentGrantSoftwareTriple(
                grant_id=document_grant_info.grant_id,
                funder=document_grant_info.funder,
                paper_doi=document_grant_info.paper_doi,
                software_statement=mention["context"],
                software_statement_type=best_cls,
                software_statement_confidence=best_score,
                software_name_raw=mention["software-name"]["rawForm"],
                software_name_normalized=mention["software-name"]["normalizedForm"],
                software_url=mention["software-url"],
            )
        )

    return results


def extract_software_statements_from_articles(
    # document_grant_infos: list[types.DocumentWithGrantInformation],
) -> list[types.DocumentGrantSoftwareTriple | types.ErrorResult] | types.ErrorResult:
    try:

        print("Copying file")
        # Hardcoded path for now
        with requests.get(
            "https://evamaxfield.github.io/rs-graph/qss-code-authors.pdf",
            stream=True,
            timeout=1800,
        ) as open_stream:
            open_stream.raise_for_status()
            with open("temp-local-path.pdf", "wb") as open_dst:
                shutil.copyfileobj(
                    open_stream.raw,
                    open_dst,
                    length=64 * 1024 * 1024,  # 64MB chunks
                )

        print("setting up server")

        # Setup GROBID server
        grobid_client, container = setup_or_connect_to_server(
            image=DEFAULT_GROBID_IMAGE,
            port=DEFAULT_GROBID_PORT,
        )
        if grobid_client is None:
            return types.ErrorResult(
                source="grobid_soft_cite",
                step="setup-or-connect-to-server",
                identifier="blah",
                error="Failed to connect to GROBID server.",
                traceback="",
            )
        
        # Process the PDF
        document_grant_infos = [
            types.DocumentWithGrantInformation(
                grant_id="blah",
                funder="NSF",
                paper_doi="10.1234/abcd.efgh",
            )
        ]
        results = []
        for document_grant_info in document_grant_infos:
            print("processing pdf")

            result = process_pdf(
                client=grobid_client,
                document_grant_info=document_grant_info,
                pdf_path="temp-local-path.pdf",
            )
            if isinstance(result, list):
                results.extend(result)
            else:
                results.append(result)

        print(results)

        print("tearing down server")
        
        # Teardown the server
        teardown_server(container)

        return results

    except Exception as e:
        return types.ErrorResult(
            source="grobid_soft_cite",
            step="software-statement-extraction",
            # identifier=";".join(
            #     [doc.paper_doi for doc in document_grant_infos]
            # ),
            identifier="blah",
            error=str(e),
            traceback=traceback.format_exc(),
        )