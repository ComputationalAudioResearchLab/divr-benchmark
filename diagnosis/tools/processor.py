import re
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class Processor:

    curdir = Path(__file__).parent.resolve()
    input_data = f"{curdir}/List of diagnosis.xlsx"
    output_file = f"{curdir}/output.yml"

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
        "Inflammatory - Infective": "inflammatory",
        "Inflammatory - Non-Infective": "inflammatory",
        "Neuro-muscular -Movement disorders": "neuro_muscular",
        "Neuro-muscular - Central nervous system": "neuro_muscular",
        "Neuro-muscular - Peripheral nervous system": "neuro_muscular",
        "Structural - Structural abnormality": "structural",
        "Structural - Epithelial/lamina propria": "structural",
        "Structural - Congenital/maturational": "structural",
        "Structural - Malignancy": "structural",
        "Structural - Vascular": "structural",
        "Trauma - External": "trauma",
        "Trauma - Internal": "trauma",
        "Trauma -Internal": "trauma",
    }
    label_mapping = {
        "a_p_compression___also_a_p_compression_moderate_": "a_p_compression",
        "amyotrophic_lateral_sclerosis___also_amyotrophic_lateral_sclerosis__als__lou_gehrig_s_disease_": "amyotrophic_lateral_sclerosis",
        "arytenoid_dislocation___also_dislocated_left_arytenoid_": "arytenoid_dislocation",
        "athetoid___also_athetoid__or_mixed_": "athetoid",
        "bilateral_recurrent_laryngeal_nerve__rln__paralysis_peripheral": "bilateral_recurrent_laryngeal_nerve_rln_paralysis_peripheral",
        "conversion_aphonia__also_conversion_dysphonia_": "conversion_dysphonia",
        "dish_syndrom": "dish_syndrome",
        "down_s_disease": "down_syndrome",
        "dysarthria__also_dysarthria_characteristics__mild_": "dysarthria",
        "gerd___also_gastric_reflux_": "gerd",
        "glottal_ap_compression__mild_": "glottal_ap_compression_mild",
        "hyperfunction__also___hyperfunctional_voice_disorder": "hyperfunction",
        "hyperkinetic_dysphonia___rigid_vocal_fold_": "hyperkinetic_dysphonia_rigid_vocal_fold",
        "hyperkinetic_dysphonia__adduction_deficit_": "hyperkinetic_dysphonia_adduction_deficit",
        "hyperkinetic_dysphonia__cordite_": "hyperkinetic_dysphonia_cordite",
        "hyperkinetic_dysphonia__nodule_": "hyperkinetic_dysphonia_nodule",
        "hyperkinetic_dysphonia__polyps_": "hyperkinetic_dysphonia_polyps",
        "hyperkinetic_dysphonia__prolapse_": "hyperkinetic_dysphonia_prolapse",
        "hyperkinetic_dysphonia__reinke_s_edema_": "hyperkinetic_dysphonia_reinke_s_edema",
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
        "normal_voice___allergy_minor__": "allergy_minor",
        "normal_voice___cold_minor__": "cold_minor",
        "normal_voice___flu_minor___days_ago__": "flu_minor_days_ago",
        "normal_voice___singing_training__": "singing_training",
        "pathological_voice__diagnosis_n_a": "pathological_voice_diagnosis_n_a",
        "polypoid_degeneration__reinke_s_": "polypoid_degeneration_reinke_s",
        "post_intubation_submucosal_edema__mild_": "post_intubation_submucosal_edema_mild",
        "post_irradiation_also_post_radiated_larynx": "post_radiated_larynx",
        "post_surgery___removal_of_keratosis_with_atypia": "post_surgery_removal_of_keratosis_with_atypia",
        "post_surgery__cricoid_removal": "post_surgery_cricoid_removal",
        "post_surgery_also_post_surgical_changes": "post_surgical_changes",
        "post_vocal_fold_stripping__also_post_vocal_cord_stripping": "post_vocal_cord_stripping",
        "reinke_s_edema__also_reinke_edema": "reinke_s_edema",
        "sulcus_vocalis__also___vocal_fold_sulcus": "vocal_fold_sulcus",
        "unilateral_or_bilateral_recurrent_laryngeal_nerve__rln__paresis": "unilateral_or_bilateral_recurrent_laryngeal_nerve_rln_paresis",
        "unilateral_recurrent_laryngeal_nerve__rln__paralysis": "unilateral_recurrent_laryngeal_nerve_rln_paralysis",
        "ventricular_compression__also__ventricular_compression__full___ventricular_compression__mild___ventricular_compression__moderate___ventricular_compression__severe___ventricular_compression__slight_": "ventricular_compression",
        "ventricular_fold___also_ventricular_vocal_folds__mild_": "ventricular_fold",
        "ventricular_phonation___also_ventricular_phonation__mild__": "ventricular_phonation",
        "vocal_fold_nodules__also_vocal_nodules_": "vocal_fold_nodules",
        "vocal_fold_polyp__also_vocal_fold_polyp_s___vocal_cord_polyp": "vocal_fold_polyp",
        "voice_disorders__undiagnosed_or_not_otherwise_specified__nos_": "voice_disorders_undiagnosed",
    }

    # [(from_level, from_key), (to_level, to_key)]
    replacements = [
        [(3, "keratosis___leukoplakia"), (3, "leukoplakia")],
        [(3, "keratosis"), (3, "leukoplakia")],
        [(3, "normal___also_normal_voice_"), (0, "normal")],
        [
            (3, "muscle_tension_adaptive_dysphonia__secondary_"),
            (2, "muscle_tension_adaptive"),
        ],
        [(3, "muscle_tension_dysphonia__primary_"), (1, "muscle_tension")],
        [
            (3, "functional_dysphonia__functional_voice_disorder"),
            (2, "functional_dysphonia"),
        ],
    ]

    def run(self):
        df = pd.read_excel(self.input_data, sheet_name="Qualtrics Results")
        df = df.map(self.normalize_votes)
        num_cols = len(df.columns)
        level_0_votes = {
            "normal": {"level": 0, "alias": ["healthy"]},
            "pathological": {"level": 0, "alias": []},
            "unclassified": {"level": 0, "alias": []},
        }
        level_1_votes = {}
        level_2_votes = {}
        level_3_votes = {}
        for i in tqdm(range(0, num_cols, 4)):
            k, v = self.process_group(group=df.iloc[0:8][df.columns[i : i + 4]])
            for vote in v["votes"].values():
                vote_key = vote.replace(" > ", "_")
                if ">" in vote:
                    if vote_key not in level_2_votes:
                        l1, l2 = vote.split(" > ")
                        level_2_votes[vote_key] = {
                            "level": 2,
                            "alias": [],
                            "parents": {
                                l1: 1.00,
                            },
                        }
                elif vote_key not in level_1_votes:
                    if vote_key not in ["na", "unclassified"]:
                        level_1_votes[vote_key] = {
                            "level": 1,
                            "alias": [],
                            "parents": {"pathological": 1.0},
                        }
            level_3_votes[k] = v

        # merging data
        votes_map = [level_0_votes, level_1_votes, level_2_votes, level_3_votes]
        for rep_from, rep_to in self.replacements:
            (rep_from_level, rep_from_key) = rep_from
            (rep_to_level, rep_to_key) = rep_to
            to_votes = votes_map[rep_to_level]
            from_votes = votes_map[rep_from_level]
            if rep_to_key in to_votes and rep_from_key in from_votes:
                to_votes[rep_to_key]["alias"] += from_votes[rep_from_key]["alias"]
                del from_votes[rep_from_key]

        with open(self.output_file, "w") as output_file:
            output_file.write("#### Level 0\n")
            yaml.dump(level_0_votes, stream=output_file)
            output_file.write("\n\n#### Level 1\n")
            yaml.dump(level_1_votes, stream=output_file)
            output_file.write("\n\n#### Level 2\n")
            yaml.dump(level_2_votes, stream=output_file)
            output_file.write("\n\n#### Level 3\n")
            yaml.dump(level_3_votes, stream=output_file)

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
            "level": 3,
            "alias": [label],
            "parents": vote_count,
            "votes": votes,
        }
        label = re.sub("[^a-z]", "_", label)
        if label in self.label_mapping:
            label = self.label_mapping[label]
        return label, result
