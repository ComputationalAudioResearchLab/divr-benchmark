import re
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class Processor:

    curdir = Path(__file__).parent.resolve()
    input_data = f"{curdir}/List of diagnosis.xlsx"
    output_file = f"{curdir}/../divr_diagnosis/diagnosis_maps/USVAC_2025.yml"

    vote_mapping = {
        # Invalid terms
        "Not a diagnostic term": "unclassified",
        # Muscle tension
        "Muscle tension voice disorder": "muscle_tension",
        "Adaptive": "adaptive",
        # Function
        "Functional (psychogenic) voice disorder": "functional",
        "Aphonia/dysphonia": "dysphonia",
        "Puberphonia": "puberphonia",
        # Organic
        "Organic voice disorder": "organic",
        "Inflammatory - Infective": "inflammatory > infective",
        "Inflammatory - Non-Infective": "inflammatory > non_infective",
        "Neuro-muscular -Movement disorders": "neuro_muscular > movement_disorder",
        "Neuro-muscular - Central nervous system": "neuro_muscular > central_nervous_disorder",
        "Neuro-muscular - Peripheral nervous system": "neuro_muscular > peripheral_nervous_disorder",
        "Structural - Structural abnormality": "structural > structural_abnormality",
        "Structural - Epithelial/lamina propria": "structural > epithelial_propria",
        "Structural - Congenital/maturational": "structural > congenital",
        "Structural - Malignancy": "structural > malignancy",
        "Structural - Vascular": "structural > vascular",
        "Trauma - External": "trauma > external",
        "Trauma - Internal": "trauma > internal",
        "Trauma -Internal": "trauma > internal",
    }
    label_mapping = {
        "a_p_compression___also_a_p_compression_moderate_": "a_p_compression",
        "amyotrophic_lateral_sclerosis___also_amyotrophic_lateral_sclerosis__als__lou_gehrig_s_disease_": "amyotrophic_lateral_sclerosis",
        "arytenoid_dislocation___also_dislocated_left_arytenoid_": "arytenoid_dislocation",
        "athetoid___also_athetoid__or_mixed_": "athetoid",
        "bilateral_recurrent_laryngeal_nerve__rln__paralysis_peripheral": "bilateral_recurrent_laryngeal_nerve_rln_paralysis_peripheral",
        "conversion_aphonia__also_conversion_dysphonia_": "conversion_dysphonia",
        "contact_pachydermia": "contact_pachyderma",
        "diplophonie": "diplophony",
        "dish_syndrom": "dish_syndrome",
        "down_syndrome": "downs_disease",
        "down_s_disease": "downs_disease",
        "dysarthria__also_dysarthria_characteristics__mild_": "dysarthria",
        "gerd___also_gastric_reflux_": "gerd",
        "glottal_ap_compression__mild_": "glottal_ap_compression_mild",
        "hyperasthenie": "hyperasthenia",
        "hyperfunktionelle_dysphonie": "hyperfunctional_dysphonia",
        "hyperfunction__also___hyperfunctional_voice_disorder": "hyperfunction",
        "hyperkinetic_dysphonia___rigid_vocal_fold_": "hyperkinetic_dysphonia_rigid_vocal_fold",
        "hyperkinetic_dysphonia__adduction_deficit_": "hyperkinetic_dysphonia_adduction_deficit",
        "hyperkinetic_dysphonia__cordite_": "hyperkinetic_dysphonia_cordite",
        "hyperkinetic_dysphonia__nodule_": "hyperkinetic_dysphonia_nodule",
        "hyperkinetic_dysphonia__polyps_": "hyperkinetic_dysphonia_polyps",
        "hyperkinetic_dysphonia__prolapse_": "hyperkinetic_dysphonia_prolapse",
        "hyperkinetic_dysphonia__reinke_s_edema_": "hyperkinetic_dysphonia_reinkes_edema",
        "hyperkinetic_dysphonia__vocal_fold_nodules_": "hyperkinetic_dysphonia_vocal_fold_nodules",
        "hyperkinetic_dysphonia__vocal_fold_paralysis_": "hyperkinetic_dysphonia_vocal_fold_paralysis",
        "hyperkinetic_dysphonia__vocal_fold_prolapse_": "hyperkinetic_dysphonia_vocal_fold_prolapse",
        "hypokinetic_dysphonia__adduction_deficit_": "hypokinetic_dysphonia_adduction_deficit",
        "hypokinetic_dysphonia__bilateral_vocal_fold_": "hypokinetic_dysphonia_bilateral_vocal_fold",
        "hypokinetic_dysphonia__conversion_dysphonia_": "hypokinetic_dysphonia_conversion_dysphonia",
        "hypokinetic_dysphonia__dysphonia_by_chordal_groove_": "hypokinetic_dysphonia_dysphonia_by_chordal_groove",
        "hypokinetic_dysphonia__glottic_insufficiency_": "hypokinetic_dysphonia_glottic_insufficiency",
        "hypokinetic_dysphonia__laryngitis_": "hypokinetic_dysphonia_laryngitis",
        "hypokinetic_dysphonia__presbiphonia_": "hypokinetic_dysphonia_presbiphonia",
        "hypokinetic_dysphonia__spasmodic_dysphonia_": "hypokinetic_dysphonia_spasmodic_dysphonia",
        "hypokinetic_dysphonia__vocal_fold_paralysis_": "hypokinetic_dysphonia_vocal_fold_paralysis",
        "idiopathic_neuro__disorder": "idiopathic_neuro_disorder",
        "inflammation_also___inflamed_vocal_folds": "inflammation",
        "keratosis__sometimes_described_as_leukoplakia_or_erythroplasia_": "keratosis",
        "laryngeal_mucosa_trauma__chemical_and_thermal_": "laryngeal_mucosa_trauma_chemical_and_thermal",
        "laryngeal_trauma___blunt": "laryngeal_trauma_blunt",
        "lesion__also_lesions_posterior_left_vocal_fold_": "lesion",
        "major_depressive_disorder__recurrent_": "major_depressive_disorder_recurrent",
        "mixed_adductor___abductor_spasmodic_dysphonia": "mixed_adductor_abductor_spasmodic_dysphonia",
        "non_fluency_syndrom": "non_fluency_syndrome",
        "normal_voice___allergy_minor__": "allergy_minor",
        "normal_voice___cold_minor__": "cold_minor",
        "normal_voice___flu_minor___days_ago__": "flu_minor_days_ago",
        "normal_voice___singing_training__": "singing_training",
        "pathological_voice__diagnosis_n_a": "pathological_voice_diagnosis_n_a",
        "pocket_wrinkled_voice": "pocket_fold_voice",
        "polypoid_degeneration__reinke_s_": "polypoid_degeneration_reinkes",
        "post_intubation_submucosal_edema__mild_": "post_intubation_submucosal_edema_mild",
        "post_irradiation_also_post_radiated_larynx": "post_radiated_larynx",
        "post_surgery___removal_of_keratosis_with_atypia": "post_surgery_removal_of_keratosis_with_atypia",
        "post_surgery__cricoid_removal": "post_surgery_cricoid_removal",
        "post_surgery_also_post_surgical_changes": "post_surgical_changes",
        "post_vocal_fold_stripping__also_post_vocal_cord_stripping": "post_vocal_cord_stripping",
        "reinke_s_edema__also_reinke_edema": "reinkes_edema",
        "rhinophonie_aperta": "rhinophony_aperta",
        "rhinophonie_clausa": "rhinophony_clausa",
        "rhinophonie_mixta": "rhinophony_mixta",
        "sulcus_vocalis__also___vocal_fold_sulcus": "vocal_fold_sulcus",
        "unilateral_or_bilateral_recurrent_laryngeal_nerve__rln__paresis": "unilateral_or_bilateral_recurrent_laryngeal_nerve_rln_paresis",
        "unilateral_recurrent_laryngeal_nerve__rln__paralysis": "unilateral_recurrent_laryngeal_nerve_rln_paralysis",
        "ventricular_compression__also__ventricular_compression__full___ventricular_compression__mild___ventricular_compression__moderate___ventricular_compression__severe___ventricular_compression__slight_": "ventricular_compression",
        "ventricular_fold___also_ventricular_vocal_folds__mild_": "ventricular_fold",
        "ventricular_phonation___also_ventricular_phonation__mild__": "ventricular_phonation",
        "vocal_fold_scar_proper": "vocal_fold_scar",
        "vocal_fold_nodules__also_vocal_nodules_": "vocal_fold_nodules",
        "vocal_fold_polyp__also_vocal_fold_polyp_s___vocal_cord_polyp": "vocal_fold_polyp",
        "voice_disorders__undiagnosed_or_not_otherwise_specified__nos_": "voice_disorders_undiagnosed",
    }
    overrides = {
        "without_dysarthria": {
            "level": 4,
            "alias": ["without dysarthria"],
            "parents": {"normal": 1.00},
        }
    }

    # [(from_level, from_key), (to_level, to_key)]
    replacements = [
        [(4, "keratosis___leukoplakia"), (4, "leukoplakia")],
        [(4, "keratosis"), (4, "leukoplakia")],
        [(4, "normal___also_normal_voice_"), (0, "normal")],
        [(4, "healthy"), (0, "normal")],
        [
            (4, "muscle_tension_adaptive_dysphonia__secondary_"),
            (2, "muscle_tension_adaptive"),
        ],
        [(4, "muscle_tension_dysphonia__primary_"), (1, "muscle_tension")],
        [
            (4, "functional_dysphonia__functional_voice_disorder"),
            (2, "functional_dysphonia"),
        ],
        [(4, "pathological"), (0, "pathological")],
        [(4, "singing_training"), (4, "singing_voice")],
    ]

    extra_aliases = [
        [4, "a_p_compression", ["a-p compression (moderate)"]],
        [
            4,
            "a_p_squeezing",
            [
                "a-p squeezing (mild)",
                "a-p squeezing (moderate)",
                "a-p squeezing (severe)",
            ],
        ],
        [
            4,
            "amyotrophic_lateral_sclerosis",
            [
                "amyotrophe lateralsklerose",
            ],
        ],
        [4, "bulbar_paralysis", ["bulbärparalyse"]],
        [4, "chondroma", ["chondrom"]],
        [4, "cordectomy", ["chordectomy", "chordektomie"]],
        [4, "conversion_dysphonia", ["conversion aphonia", "conversion dysphonia"]],
        [4, "cyst", ["cyste"]],
        [4, "dysarthrophonia", ["dysarthrophonie"]],
        [4, "dysody", ["dysodie"]],
        [4, "dysphonia", ["dysphonie"]],
        [4, "dysplastic_dysphonia", ["dysplastische dysphonie"]],
        [4, "dysplastic_larynx", ["dysplastischer kehlkopf"]],
        [4, "epiglottic_carcinoma", ["epiglottiskarzinom"]],
        [4, "fibroma", ["fibrom"]],
        [4, "frontolateral_partial_resection", ["frontolaterale teilresektion"]],
        [2, "functional_dysphonia", ["funktionelle dysphonie"]],
        [4, "granuloma", ["granulom"]],
        [4, "hypofunctional_dysphonia", ["hypofunktionelle dysphonie"]],
        [4, "hypopharyngeal_tumor", ["hypopharynxtumor"]],
        [4, "hypotonic_dysphonia", ["hypotone dysphonie"]],
        [4, "internal_weakness", ["internusschwäche"]],
        [4, "intubation_granuloma", ["intubationsgranulom"]],
        [4, "intubation_trauma", ["intubationsschaden"]],
        [4, "juvenile_dysphonia", ["juvenile dysphonie"]],
        [4, "laryngeal_tumor", ["kehlkopftumor"]],
        [4, "contact_pachyderma", ["kontaktpachydermie"]],
        [4, "laryngocele", ["laryngozele"]],
        [4, "lesion", ["lesions posterior left vocal fold"]],
        [4, "leukoplakia", ["leukoplakie"]],
        [4, "medial_cervical_cyst", ["mediale halscyste"]],
        [4, "mesopharyngeal_tumor", ["mesopharynxtumor"]],
        [4, "downs_disease", ["morbus down"]],
        [4, "parkinson_disease", ["morbus parkinson", "parkinson's disease"]],
        [4, "mutation", ["mutatio"]],
        [4, "mutation_fistula_voice", ["mutationsfistelstimme"]],
        [4, "superior_laryngeal_nerve_lesion", ["n. laryngeus superior läsion"]],
        [4, "superior_laryngeal_nerve_neuralgia", ["n. laryngeus superior neuralgie"]],
        [4, "orofacial_dyspraxia", ["orofaciale dyspraxie"]],
        [4, "papilloma", ["papillom"]],
        # [4, "myasthenia", ["phonasthenie"]],
        [4, "phonation_nodules", ["phonationsknötchen"]],
        # [4, "polters_syndrome", ["poltersyndrom"]],
        [4, "psychogenic_aphonia", ["psychogene aphonie"]],
        [4, "psychogenic_dysphonia", ["psychogene dysphonie"]],
        [4, "psychogenic_microphony", ["psychogene mikrophonie"]],
        [4, "reinkes_edema", ["reinke ödem"]],
        [4, "recurrent_paralysis", ["rekurrensparese"]],
        [4, "sigmatism", ["sigmatismus"]],
        [4, "spasmodic_dysphonia", ["spasmodische dysphonie"]],
        [4, "vocal_fold_carcinoma", ["stimmlippenkarzinom"]],
        [
            4,
            "vocal_fold_polyp",
            [
                "stimmlippenpolyp",
                "vocal fold polyp",
                "vocal fold polyp(s)",
                "vocal cord polyp",
            ],
        ],
        [4, "synechia", ["synechie"]],
        [4, "singing_voice", ["singer's voice", "sängerstimme", "gesangsstimme"]],
        [4, "pocket_fold_hyperplasia", ["taschenfaltenhyperplasie"]],
        [4, "pocket_fold_voice", ["taschenfaltenstimme"]],
        [4, "velopharyngoplasty", ["velopharyngoplastik"]],
        [4, "ventricular_fold", ["ventricular vocal folds (mild)"]],
        [
            4,
            "ventricular_compression",
            [
                "ventricular compression",
                "ventricular compression (full)",
                "ventricular compression (mild)",
                "ventricular compression (moderate)",
                "ventricular compression (severe)",
                "ventricular compression (slight)",
            ],
        ],
        [
            4,
            "central_laryngeal_movement_disorder",
            ["zentral-laryngale bewegungsstörung"],
        ],
        [
            4,
            "inflammation",
            ["inflammation", "inflammed vocal folds"],
        ],
    ]

    def run(self):
        df = pd.read_excel(self.input_data, sheet_name="Qualtrics Results")
        df = df.map(self.normalize_votes)
        num_cols = len(df.columns)
        votes_map = self.__create_labels(df, num_cols)
        self.__merge_classes(votes_map)
        self.__fix_aliases(votes_map)

        with open(self.output_file, "w") as output_file:
            output_file.write("#### Level 0\n")
            yaml.dump(votes_map[0], stream=output_file, allow_unicode=True)
            output_file.write("\n\n#### Level 1\n")
            yaml.dump(votes_map[1], stream=output_file, allow_unicode=True)
            output_file.write("\n\n#### Level 2\n")
            yaml.dump(votes_map[2], stream=output_file, allow_unicode=True)
            output_file.write("\n\n#### Level 3\n")
            yaml.dump(votes_map[3], stream=output_file, allow_unicode=True)
            output_file.write("\n\n#### Level 4\n")
            yaml.dump(votes_map[4], stream=output_file, allow_unicode=True)

    def __create_labels(self, df, num_cols):
        level_0_votes = {
            "normal": {"level": 0, "alias": ["healthy"]},
            "pathological": {"level": 0, "alias": []},
            "unclassified": {"level": 0, "alias": ["mixed", "not sure", "other"]},
        }
        level_1_votes = {
            "non_laryngeal": {"level": 1, "parents": {"pathological": 1.00}},
        }
        level_2_votes = {
            "metabolic": {
                "level": 2,
                "alias": ["endocrine"],
                "parents": {"non_laryngeal": 1.00},
            },
            "psychiatric": {
                "level": 2,
                "alias": ["psychological"],
                "parents": {"non_laryngeal": 1.00},
            },
            "respiratory": {"level": 2, "parents": {"organic": 1.00}},
            "systemic": {"level": 2, "parents": {"non_laryngeal": 1.00}},
        }
        level_3_votes = {}
        # These 4 labels were accidentally left out of being classified by the clinicians
        level_4_votes = {
            "anterior_mass": {
                "level": 4,
                "alias": ["anterior mass"],
                "parents": {"unclassified": 1.00},
            },
            "hypokinetic_dysphonia_extraglottic_air_leak": {
                "level": 4,
                "alias": ["hypokinetic dysphonia (extraglottic air leak)"],
                "parents": {"unclassified": 1.00},
            },
            "myasthenia": {
                "level": 4,
                "alias": ["phonasthenie"],
                "parents": {"unclassified": 1.00},
            },
            "polters_syndrome": {
                "level": 4,
                "alias": ["polter's syndrome", "poltersyndrom"],
                "parents": {"unclassified": 1.00},
            },
        }
        for i in tqdm(range(0, num_cols, 4)):
            k, v = self.process_group(group=df.iloc[0:8][df.columns[i : i + 4]])
            if k in self.overrides:
                v = self.overrides[k]
            else:
                for vote in v["votes"].values():
                    vote_key = vote.replace(" > ", "_")
                    if ">" in vote:
                        splits = vote.split(" > ")
                        num_splits = len(splits)
                        if num_splits == 2:
                            if vote_key not in level_2_votes:
                                l1, l2 = splits
                                level_2_votes[vote_key] = {
                                    "level": 2,
                                    "alias": [],
                                    "parents": {
                                        l1: 1.00,
                                    },
                                }
                        elif num_splits == 3:
                            if vote_key not in level_3_votes:
                                l1, l2, l3 = splits
                                level_2_key = f"{l1}_{l2}"
                                level_3_votes[vote_key] = {
                                    "level": 3,
                                    "alias": [],
                                    "parents": {
                                        level_2_key: 1.00,
                                    },
                                }
                                if level_2_key not in level_2_votes:
                                    level_2_votes[level_2_key] = {
                                        "level": 2,
                                        "alias": [],
                                        "parents": {
                                            l1: 1.00,
                                        },
                                    }
                        else:
                            raise ValueError(
                                f"unexpected number of splits in vote: {vote}"
                            )
                    elif vote_key not in level_1_votes:
                        if vote_key not in ["na", "unclassified"]:
                            level_1_votes[vote_key] = {
                                "level": 1,
                                "alias": [],
                                "parents": {"pathological": 1.0},
                            }
            level_4_votes[k] = v
        return [
            level_0_votes,
            level_1_votes,
            level_2_votes,
            level_3_votes,
            level_4_votes,
        ]

    def __merge_classes(self, votes_map):
        for rep_from, rep_to in self.replacements:
            (rep_from_level, rep_from_key) = rep_from
            (rep_to_level, rep_to_key) = rep_to
            to_votes = votes_map[rep_to_level]
            from_votes = votes_map[rep_from_level]
            if rep_to_key in to_votes and rep_from_key in from_votes:
                to_votes[rep_to_key]["alias"] += [rep_from_key.replace("_", " ")]
                to_votes[rep_to_key]["alias"] += from_votes[rep_from_key]["alias"]
                del from_votes[rep_from_key]

    def __fix_aliases(self, votes_map):
        for vm in votes_map:
            for v in vm.values():
                if "alias" in v:
                    for i, a in enumerate(v["alias"]):
                        if "also" in a:
                            del v["alias"][i]
                    v["alias"] = sorted(v["alias"])

        for level, key, alias in self.extra_aliases:
            votes_map[level][key]["alias"] += alias

    def normalize_votes(self, cell):
        if cell in self.vote_mapping:
            return self.vote_mapping[cell]
        return cell

    def process_group(self, group: pd.DataFrame):
        label = group.iloc[0].iat[0].lower()
        votes = group.iloc[1:8]
        votes = {}
        vote_list = []
        vote_count = {}
        for idx, row in group.iloc[1:8].iterrows():
            a, b, c, d = row.tolist()
            if pd.notna(a):
                l1 = a
                l2: str | None = None
                if a == "organic":
                    if pd.notna(b):
                        l2 = b
                if a == "functional":
                    if pd.notna(c):
                        l2 = c
                if a == "muscle_tension":
                    if pd.notna(d):
                        l2 = d
                if l2 is not None:
                    if l2.lower() in ["primary", "secondary"]:
                        vote = l1
                    else:
                        vote = f"{l1} > {l2}"
                else:
                    vote = l1
                vote_list += [vote.replace(" > ", "_")]
                votes[f"clinician {idx}"] = vote
            else:
                votes[f"clinician {idx}"] = "na"
        total_votes = len(vote_list)
        for k, v in zip(*np.unique(vote_list, return_counts=True)):
            vote_count[str(k)] = round(float(v / total_votes), ndigits=2)
        result = {
            "level": 4,
            "alias": self.__process_alias(label),
            "parents": vote_count,
            "votes": votes,
        }
        label = re.sub("[^a-z]", "_", label)
        if label in self.label_mapping:
            label = self.label_mapping[label]
        return label, result

    def __process_alias(self, label):
        if "\n" not in label:
            return [label]

        alias = []
        for lrow in label.split("\n"):
            lrow: str = lrow.strip()
            if len(lrow) > 0:
                if lrow.startswith("( also "):
                    lrow = lrow.replace("( also ", "")
                if lrow.startswith("(also "):
                    lrow = lrow.replace("(also ", "")
                if lrow.startswith("also "):
                    lrow = lrow.replace("also ", "")
                if lrow.endswith("))"):
                    lrow = lrow.replace("))", ")")
                if lrow.endswith(")") and not ("(" in lrow):
                    lrow = lrow[:-1]
                if lrow.startswith("- "):
                    lrow = lrow[2:]
                alias += [lrow]
        return alias
