#!/usr/bin/env python

from __future__ import annotations

from datetime import datetime

import requests
from tqdm import tqdm

from .. import types

###############################################################################

# List of NSF award detail fields to retrieve
NSF_AWARD_FIELDS = [
    # NSF Details
    "agency",
    "fundAgencyCode",
    "awardAgencyCode",
    "cfdaNumber",
    "ueiNumber",
    "parentUeiNumber",
    "poName",
    "poEmail",
    "primaryProgram",
    "transType",
    # Dates and Amounts
    "date",
    "startDate",
    "expDate",
    "estimatedTotalAmt",
    "fundsObligatedAmt",
    "fundProgramName",
    # Awardee and PI
    "awardee",
    "awardeeName",
    "awardeeStateCode",
    "pdPIName",
    "piFirstName",
    "piMiddeInitial",
    "piLastName",
    "piEmail",
    "coPDPI",
    "perfLocation",
    "perfDistrictCode",
    "perfStateCode",
    # Grant Details and Outcomes
    "id",
    "title",
    "abstractText",
    "projectOutComesReport",
    "publicationResearch",
    "publicationConference",
]

NSF_AWARD_FIELDS_STR = ",".join(NSF_AWARD_FIELDS)


class NSFDirectorates:
    Biological_Sciences = "BIO"
    Computer_and_Information_Science_and_Engineering = "CISE"
    Education_and_Human_Resources = "EHR"
    Engineering = "ENG"
    Geosciences = "GEO"
    Integrative_Activities = "OIA"
    International_Science_and_Engineering = "OISE"
    Mathematical_and_Physical_Sciences = "MPS"
    Social_Behavioral_and_Economic_Sciences = "SBE"
    Technology_Innovation_and_Partnerships = "TIP"


ALL_NSF_DIRECTORATES = [
    getattr(NSFDirectorates, a) for a in dir(NSFDirectorates) if "__" not in a
]

CFDA_NUMBER_TO_DIRECTORATE_LUT = {
    "47.041": NSFDirectorates.Engineering,
    "47.049": NSFDirectorates.Mathematical_and_Physical_Sciences,
    "47.050": NSFDirectorates.Geosciences,
    "47.070": NSFDirectorates.Computer_and_Information_Science_and_Engineering,
    "47.074": NSFDirectorates.Biological_Sciences,
    "47.075": NSFDirectorates.Social_Behavioral_and_Economic_Sciences,
    "47.076": NSFDirectorates.Education_and_Human_Resources,
    "47.079": NSFDirectorates.International_Science_and_Engineering,
    "47.083": NSFDirectorates.Integrative_Activities,
    "47.084": NSFDirectorates.Technology_Innovation_and_Partnerships,
}


NSF_DIRECTORATE_TO_CFDA_NUMBER_LUT = {
    code: number for number, code in CFDA_NUMBER_TO_DIRECTORATE_LUT.items()
}

_NSF_API_URL_TEMPLATE = (
    "https://api.nsf.gov/services/v1/awards.json?"
    "&agency={agency}"
    "&dateStart={start_date}"
    "&dateEnd={end_date}"
    "&cfdaNumber={cfda_number}"
    "&transType={transaction_type}"
    "&printFields={dataset_fields}"
    "&offset={offset}"
)

###############################################################################


def _parse_nsf_datetime(dt: str | datetime) -> str:
    if isinstance(dt, str):
        # Assume "/" means MM/DD/YYYY format
        if "/" in dt:
            return dt

        # Assume "-" means isoformat
        if "-" in dt:
            dt = datetime.fromisoformat(dt)
        # Anything else, raise
        else:
            raise ValueError(
                f"Provided value to `start_date` parameter must be provided as "
                f"either MM/DD/YYYY or YYYY-MM-DD format. Received: '{dt}'"
            )

    # Should either be already formated (from "/")
    # or we had isoformat conversion or provided datetime
    return dt.strftime("%m/%d/%Y")


def _get_nsf_chunk(
    start_date: str,
    end_date: str,
    cfda_number: str,
    agency: str,
    transaction_type: str,
    dataset_fields: str,
    offset: int,
) -> list[types.GrantDetails]:
    # Make the request
    response = requests.get(
        _NSF_API_URL_TEMPLATE.format(
            start_date=start_date,
            end_date=end_date,
            cfda_number=cfda_number,
            agency=agency,
            transaction_type=transaction_type,
            dataset_fields=dataset_fields,
            offset=offset,
        )
    )

    # TODO:
    # parse datetimes

    # Parse and return
    response_json = response.json()["response"]
    if "award" in response_json:
        return [
            types.GrantDetails(
                funder="NSF",
                grant_id=award["id"],
                title=award["title"] if "title" in award else None,
                abstract=award["abstractText"] if "abstractText" in award else None,
                outcomes_report=award["projectOutComesReport"]
                if "projectOutComesReport" in award
                else None,
                directorate=CFDA_NUMBER_TO_DIRECTORATE_LUT[cfda_number],
                start_date=award["startDate"] if "startDate" in award else None,
                end_date=award["expDate"] if "expDate" in award else None,
                funded_amount=award["fundsObligatedAmt"]
                if "fundsObligatedAmt" in award
                else None,
                primary_investigator=award["pdPIName"] if "pdPIName" in award else None,
                institution=award["awardeeName"] if "awardeeName" in award else None,
            )
            for award in response_json["award"]
        ]

    return []


def get_dataset(
    start_date: str | datetime,
    end_date: str | datetime | None = None,
    **kwargs: dict[str, str],
) -> list[types.GrantDetails]:
    # Get chunks
    all_directorate_grant_ids: set[str] = set()
    all_directorates_data: list[types.GrantDetails] = []
    for directorate in tqdm(ALL_NSF_DIRECTORATES, desc="Iterating directorates..."):
        # Parse datetimes
        formatted_start_date = _parse_nsf_datetime(start_date)
        if end_date is None:
            end_date = datetime.now()
        formatted_end_date = _parse_nsf_datetime(end_date)

        # Reverse lookup the cfda number from the name
        cfda_number = NSF_DIRECTORATE_TO_CFDA_NUMBER_LUT[directorate]

        # TODO:
        # impl more filtering
        # we don't want to get XYZ grants

        # Run gather
        current_offset = 1
        with tqdm(desc="Getting NSF data in chunks...", leave=False) as pbar:
            while True:
                # Get chunk
                chunk = _get_nsf_chunk(
                    start_date=formatted_start_date,
                    end_date=formatted_end_date,
                    cfda_number=cfda_number,
                    agency="NSF",
                    transaction_type="Standard Grant",
                    dataset_fields=NSF_AWARD_FIELDS_STR,
                    offset=current_offset,
                )

                # Append to all data
                all_directorates_data.extend(
                    [
                        award
                        for award in chunk
                        if award.grant_id not in all_directorate_grant_ids
                    ]
                )
                all_directorate_grant_ids.update([award.grant_id for award in chunk])

                break

                # Check chunk length
                # The default request size for NSF is 25
                # If we received less than 25 results,
                # we can assume we are done.
                if len(chunk) < 25:
                    break

                # Update state
                current_offset += 25
                pbar.update(1)

    return all_directorates_data
