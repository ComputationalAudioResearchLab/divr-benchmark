import pytest
from typing import List
from divr_diagnosis import diagnosis_maps

diagnosis_map = diagnosis_maps.USVAC_2025()


@pytest.mark.parametrize(
    "query_name, expected_key",
    [
        ("laryngeal_trauma_blunt", "laryngeal_trauma_blunt"),
        ("laryngeal trauma - blunt", "laryngeal_trauma_blunt"),
    ],
)
def test_get(query_name: str, expected_key: str):
    diag = diagnosis_map.get(name=query_name)
    assert diag.name == expected_key


@pytest.mark.parametrize(
    "query_name, expected_key",
    [
        ("laryngeal_trauma_blunt", "laryngeal_trauma_blunt"),
        ("laryngeal trauma - blunt", "laryngeal_trauma_blunt"),
        ("laryNgeal trAuma - BlUnt", "laryngeal_trauma_blunt"),
    ],
)
def test_indexing(query_name: str, expected_key: str):
    diag = diagnosis_map[query_name]
    assert diag.name == expected_key


@pytest.mark.parametrize(
    "query_name, result",
    [
        ("laryngeal_trauma_blunt", True),
        ("laryngeal trauma - blunt", True),
        ("laryNgeal trAuma - BlUnt", True),
        ("laryNgealaaaaa trAuma - BlUntaaaa", False),
    ],
)
def test_key_check(query_name: str, result: bool):
    assert (query_name in diagnosis_map) == result


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
def test_find(parent_name: str, expected_diags: List[str]):
    diags = diagnosis_map.find(name=parent_name)
    assert sorted([d.name for d in diags]) == expected_diags
