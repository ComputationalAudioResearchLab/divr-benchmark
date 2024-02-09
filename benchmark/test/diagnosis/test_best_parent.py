import pytest
from typing import Any, Dict, List, Tuple
from uuid import uuid4
from divr_benchmark.diagnosis import Diagnosis, DiagnosisLink


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
        ["b.1", {"a.0": {"a.1": 0.2, "a.2": 0.1}, "b.0": {"b.1": 0.4}}],
        ["a.2", {"a.0": {"a.1": 0.2, "a.2": 0.3}, "b.0": {"b.1": 0.4}}],
    ],
)
def test_complex_genealogy(expected_parent: str, data: Dict[str, Any]):
    def generator(
        val: Dict[str, Any], parents: List[DiagnosisLink]
    ) -> List[DiagnosisLink]:
        max_parent_level = max(parents, key=lambda x: x.parent.level).parent.level
        current_level = max_parent_level + 1
        links: List[DiagnosisLink] = []
        for parent_key, parent_val in val.items():
            parent = Diagnosis(
                name=parent_key,
                level=current_level,
                alias=[],
                parents=parents,
                votes={},
            )
            parent_link = DiagnosisLink(parent=parent, weight=1)
            for key, val in parent_val.items():
                if isinstance(val, dict):
                    links += generator(val=val, parents=[parent_link])
                else:
                    parent = Diagnosis(
                        name=key,
                        level=current_level,
                        alias=[],
                        parents=parents,
                        votes={},
                    )
                    links += [DiagnosisLink(parent=parent, weight=val)]
        return links

    root = Diagnosis(name="pathological", level=0, alias=[], parents=[], votes={})
    root_link = DiagnosisLink(parent=root, weight=1)
    parents = generator(val=data, parents=[root_link])
    test_diagnosis = Diagnosis(
        name=str(uuid4()), level=2, alias=[], parents=parents, votes={}
    )
    assert test_diagnosis.best_parent_link is not None
    assert test_diagnosis.best_parent_link.parent.name == expected_parent
