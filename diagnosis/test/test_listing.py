import pytest
from typing import List
from divr_diagnosis import diagnosis_maps

diagnosis_map = diagnosis_maps.USVAC_2025()


@pytest.mark.parametrize(
    "parent_name, expected_diags",
    [
        ("laryngeal_trauma_blunt", ["laryngeal_trauma_blunt"]),
        (
            "organic_trauma_external",
            [
                "arytenoid_dislocation",
                "laryngeal_trauma",
                "laryngeal_trauma_blunt",
                "organic_trauma_external",
            ],
        ),
        (
            "organic_trauma_internal",
            [
                "intubation_damage",
                "intubation_granuloma",
                "intubation_trauma",
                "laryngeal_mucosa_trauma_chemical_and_thermal",
                "organic_trauma_internal",
            ],
        ),
        (
            "organic_trauma",
            [
                "arytenoid_dislocation",
                "intubation_damage",
                "intubation_granuloma",
                "intubation_trauma",
                "laryngeal_mucosa_trauma_chemical_and_thermal",
                "laryngeal_trauma",
                "laryngeal_trauma_blunt",
                "organic_trauma",
                "organic_trauma_external",
                "organic_trauma_internal",
            ],
        ),
    ],
)
def test_listing(parent_name: str, expected_diags: List[str]):
    diags = diagnosis_map.find(name=parent_name)
    assert sorted([d.name for d in diags]) == expected_diags
