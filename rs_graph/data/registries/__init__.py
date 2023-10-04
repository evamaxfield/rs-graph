"""Preprocessing and availability of different datasets."""

from dataclasses import dataclass
from enum import Enum

###############################################################################


class RegistryEnum(Enum):
    CARGO = "rust"
    COMPOSER = "composer"
    NUGET = "nuget"
    ACTIONS = "actions"
    GO = "go"
    MAVEN = "maven"
    NPM = "npm"
    PIP = "pip"
    PNPM = "pnpm"
    PUB = "pub"
    POETRY = "poetry"
    RUBYGEMS = "rubygems"
    SWIFT = "swift"
    YARN = "yarn"


@dataclass(frozen=True)
class AnalysisRegistries:
    values: list[RegistryEnum]


REGISTRIES_FOR_ANALYSIS = AnalysisRegistries(
    values=[
        RegistryEnum.CARGO,
        RegistryEnum.PIP,
        RegistryEnum.NPM,
    ],
)
