import pytest
from typing import Dict, List, Tuple
from uuid import uuid4
from divr_diagnosis import Diagnosis, DiagnosisLink, diagnosis_maps


diagnosis_map = diagnosis_maps.USVAC_2025()


@pytest.mark.parametrize(
    "weights_and_names, expected_parent",
    [
        [[(1, "a")], "a"],
        [[(0.1, "a"), (0.9, "b")], "b"],
        [[(0.1, "a"), (0.4, "b"), (0.5, "c")], "c"],
    ],
)
def test_simple_max_parent(
    weights_and_names: List[Tuple[float, str]],
    expected_parent: str,
):
    root = Diagnosis(name="pathological", level=0, alias=[], parents=[], votes={})
    root_link = DiagnosisLink(parent=root, weight=1)
    parents = [
        DiagnosisLink(
            parent=Diagnosis(
                name=name, level=1, alias=[], parents=[root_link], votes={}
            ),
            weight=weight,
        )
        for (weight, name) in weights_and_names
    ]
    test_diagnosis = Diagnosis(
        name=str(uuid4()), level=2, alias=[], parents=parents, votes={}
    )
    assert test_diagnosis.best_parent_link is not None
    assert test_diagnosis.best_parent_link.parent.name == expected_parent


@pytest.mark.parametrize(
    "weights_names_and_root, expected_parent",
    [
        [[(0.5, "a", "pathological"), (0.5, "b", "normal")], "a"],
        [[(0.4, "a", "pathological"), (0.5, "b", "normal")], "b"],
        [
            [
                (0.1, "a", "pathological"),
                (0.2, "b", "normal"),
                (0.2, "c", "unclassified"),
            ],
            "b",
        ],
        [
            [
                (0.1, "a", "pathological"),
                (0.2, "b", "normal"),
                (0.3, "c", "unclassified"),
            ],
            "c",
        ],
    ],
)
def test_ties_with_different_roots(
    weights_names_and_root: List[Tuple[float, str, str]],
    expected_parent: str,
):
    parents = [
        DiagnosisLink(
            parent=Diagnosis(
                name=name,
                level=1,
                alias=[],
                parents=[
                    DiagnosisLink(
                        parent=Diagnosis(
                            name=root,
                            level=0,
                            alias=[],
                            parents=[],
                            votes={},
                        ),
                        weight=1,
                    )
                ],
                votes={},
            ),
            weight=weight,
        )
        for (weight, name, root) in weights_names_and_root
    ]
    test_diagnosis = Diagnosis(
        name=str(uuid4()), level=2, alias=[], parents=parents, votes={}
    )
    assert test_diagnosis.best_parent_link is not None
    assert test_diagnosis.best_parent_link.parent.name == expected_parent


@pytest.mark.parametrize(
    "expected_parent, data",
    [
        [
            "a2.2",
            {
                # level 0
                "pathological": [0, []],
                # level 1
                "a1": [1, [("pathological", 1)]],
                "b1": [1, [("pathological", 1)]],
                # level 2
                "a2.1": [2, [("a1", 1)]],
                "a2.2": [2, [("a1", 1)]],
                "b2.1": [2, [("b1", 1)]],
                # Final
                "test": [3, [("a2.1", 0.2), ("a2.2", 0.3), ("b2.1", 0.4)]],
            },
        ],
        [
            "b1",
            {
                # level 0
                "pathological": [0, []],
                # level 1
                "a1": [1, [("pathological", 1)]],
                "b1": [1, [("pathological", 1)]],
                # level 2
                "a2.1": [2, [("a1", 1)]],
                "a2.2": [2, [("a1", 1)]],
                # Final
                "test": [3, [("a2.1", 0.1), ("a2.2", 0.3), ("b1", 0.6)]],
            },
        ],
        [
            "b1",
            {
                # level 0
                "pathological": [0, []],
                # level 1
                "a1": [1, [("pathological", 1)]],
                "b1": [1, [("pathological", 1)]],
                # Final
                "test": [3, [("a1", 0.4), ("b1", 0.6)]],
            },
        ],
    ],
)
def test_complex_genealogy(
    expected_parent: str, data: Dict[str, Tuple[int, List[Tuple[str, float]]]]
):
    diagnosis_map: Dict[str, Diagnosis] = {}
    for name, [level, parents] in data.items():
        parent_links: List[DiagnosisLink] = []
        for parent_key, parent_weight in parents:
            parent_links.append(
                DiagnosisLink(parent=diagnosis_map[parent_key], weight=parent_weight)
            )
        diag = Diagnosis(
            name=name,
            level=level,
            alias=[],
            parents=parent_links,
            votes={},
        )
        diagnosis_map[name] = diag

    test_diagnosis = diagnosis_map["test"]
    assert test_diagnosis.best_parent_link is not None
    assert test_diagnosis.best_parent_link.parent.name == expected_parent


@pytest.mark.parametrize(
    "expected_parent, base_diagnosis_name",
    [("organic_structural_epithelial_propria", "hyperkinetic_dysphonia_reinkes_edema")],
)
def test_specific_examples(expected_parent: str, base_diagnosis_name: str):
    test_diagnosis = diagnosis_map.get(base_diagnosis_name)
    assert test_diagnosis.best_parent_link is not None
    assert test_diagnosis.best_parent_link.parent.name == expected_parent
