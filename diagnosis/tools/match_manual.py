import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, TypedDict, List


class Entry(TypedDict):
    alias: List[str] | None
    level: int
    parents: Dict[str, float] | None
    votes: Dict[str, str] | None


@dataclass
class Unmatched:
    key: str
    manual: Entry
    auto: Entry | None


class MatchManual:
    # This script compares the auto generated file with the
    # first version of USVAC_2025 that I created manually
    __curdir = Path(__file__).parent.resolve()
    __matched: List[str] = []
    __ignored: List[str] = []
    __unmatched: List[Unmatched] = []
    __verified_keys = [
        "acute_laryngitis",  # one extra vote
        "amyotrophic_lateral_sclerosis",  # one extra vote
        "anterior_saccular_cyst",  # copy paste error
        "balbuties",  # one extra vote
        "bilateral_recurrent_laryngeal_nerve_(rln)_paralysis_peripheral",  # key had paranthesis
        "carcinoma_in_situ",  # one extra vote
        "chondroma",  # difference in na vs not a diagnosis
        "chordectomy",  # present as cordectomy
        "cyst",  # one extra vote
        "dish_syndrome",  # difference in na vs not a diagnosis
        "diplophony",  # one extra vote
        "dysarthria",  # one extra vote
        "dysarthrophonia",  # difference in na vs not a diagnosis
        "dysody",  # difference in na vs not a diagnosis
        "dysphonia",  # difference in na vs not a diagnosis
        "dysplastic_dysphonia",  # difference in na vs not a diagnosis
        "dysplastic_larynx",  # difference in na vs not a diagnosis
        "epiglottic_carcinoma",  # difference in na vs not a diagnosis
        "functional_dysphonia",  # moved to level 2
        "hyperkinetic_dysphonia_adduction_deficit",  # muscle tension split
        "hyperkinetic_dysphonia_prolapse",  # muscle tension split
        "hyperkinetic_dysphonia_reinkes_edema",  # muscle tension split
        "hyperkinetic_dysphonia_rigid_vocal_fold",  # muscle tension split
        "hyperkinetic_dysphonia_vocal_fold_paralysis",  # bad copy
        "hyperkinetic_dysphonia_vocal_fold_prolapse",  # muscle tension split
        "hypofunctional_dysphonia",  # muscle tension split
        "hypokinetic_dysphonia_adduction_deficit",  # muscle tension split
        "hypokinetic_dysphonia_conversion_dysphonia",  # functional split
        "hypokinetic_dysphonia_dysphonia_by_chordal_groove",  # muscle tension split
        "hypokinetic_dysphonia_glottic_insufficiency",  # muscle tension split
        "juvenile_dysphonia",  # functional split
        "keratosis",  # present as leukoplakia
        "laryngeal_mucosa_trauma",  # present as laryngeal_mucosa_trauma_chemical_and_thermal
        "laryngopharyngeal_reflux",  # difference in na vs not a diagnosis
        "major_depressive_disorder",  # present as major_depressive_disorder_recurrent
        "myasthenia",  # present nowhere, where did we get this?
        "papilloma",  # bad copy,
        "parkinsons",  # present as parkinson_disease
        "polters_syndrome",  # not classified, hence absent
        "psychogenic_aphonia",  # functional split
        "psychogenic_dysphonia",  # functional split
        "psychogenic_microphony",  # functional split
        "puberphonia",  # functional split
        "singers_voice",  # present as singing_voice
        "unclassified_pathology",  # merged with unclassified
        "unilateral_or_bilateral_recurrent_laryngeal_nerve_(rln)_paresis",  # paranthesis difference
        "unilateral_recurrent_laryngeal_nerve_(rln)_paralysis",  # paranthesis difference
        "without_dysarthria",  # is unclassified, needs to be manually fixed
    ]

    __replacement_values = {
        "muscle tension": "muscle_tension",
        "not a diagnosis": "unclassified",
        "organic > neuro-muscular": "organic > neuro_muscular",
        "organic > neuro muscular": "organic > neuro_muscular",
        "organic > organic_structural": "organic > structural",
        "unclassified_pathology": "unclassified",
    }

    def __init__(self) -> None:
        self.__auto_terms = self.__load_map(
            f"{self.__curdir}/../divr_diagnosis/diagnosis_maps/USVAC_2025.yml"
        )
        self.__manual_terms = self.__load_map(f"{self.__curdir}/USVAC_2025.yml")
        self.__replace_manual_values()

    def run(self):
        for key in self.__manual_terms:
            M = self.__manual_terms[key]
            if key in self.__verified_keys:
                self.__ignored += [key]
            elif key not in self.__auto_terms:
                self.__unmatched += [Unmatched(key=key, manual=M, auto=None)]
            else:
                A = self.__auto_terms[key]
                if self.__equals(key=key, M=M, A=A):
                    self.__matched += [key]
                else:
                    self.__unmatched += [Unmatched(key=key, manual=M, auto=A)]
        unmatched = [pair.key for pair in self.__unmatched]
        print(len(unmatched), sorted(unmatched))
        print(len(self.__matched), sorted(self.__matched))
        print(len(self.__ignored), sorted(self.__ignored))

    def __equals(self, key: str, M: Entry, A: Entry) -> bool:
        if A["level"] != M["level"]:
            if A["level"] == 4 and M["level"] == 3:
                return True
            else:
                return False
        if "votes" in M:
            if A["votes"] != M["votes"]:
                return False
        if "parents" in M:
            if "parents" not in A:
                return False
            if A["parents"] != M["parents"]:
                return False
        return True

    def __load_map(self, path: str) -> Dict[str, Entry]:
        with open(path, "r") as map_file:
            return yaml.full_load(map_file)

    def __replace_manual_values(self):
        for val in self.__manual_terms.values():
            if "votes" in val and val["votes"] is not None:
                for v_key, vote in val["votes"].items():
                    if vote in self.__replacement_values:
                        val["votes"][v_key] = self.__replacement_values[vote]
            if "parents" in val and val["parents"] is not None:
                for p_key in list(val["parents"].keys()):
                    if p_key in self.__replacement_values:
                        new_key = self.__replacement_values[p_key]
                        weight = val["parents"][p_key]
                        val["parents"][new_key] = weight
                        del val["parents"][p_key]
