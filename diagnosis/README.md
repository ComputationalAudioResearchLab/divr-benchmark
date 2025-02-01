# DiVR (Disordered Voice Recognition) - Diagnosis

This repository contains work used to standardize diagnosis to different classification systems found across the literatue, including a new classification system constructed at USVAC.

## Installation

```sh
pip install divr-diagnosis
```

## How to use

```python
from divr_diagnosis import diagnosis_maps

# Select a diagnosis map (options in `divr_diagnosis/diagnosis_maps` director)
diagnosis_map = diagnosis_maps.USVAC_2025()

# get a specific diagnosis
diagnosis = diagnosis_map.get("laryngeal_tuberculosis")

# also supported dictionary syntax
diagnosis = diagnosis_map["laryngeal_tuberculosis"]
assert "laryngeal_tuberculosis" in diagnosis_map

# allow fetching by aliases
assert "laryngeal tuberculosis" in diagnosis_map

# check if diagnosis is of a type
assert diagnosis.satisfies("pathological") == True
assert diagnosis.satisfies("normal") == False

# get diagnosis parents
assert diagnosis.at_level(3) == "organic_inflammatory_infective"
assert diagnosis.at_level(2) == "organic_inflammatory"
assert diagnosis.at_level(1) == "organic"
assert diagnosis.at_level(0) == "pathological"
assert diagnosis.root == "pathological"

# check if a diagnosis was not classified
diag_inc = diagnosis_map.get("internal_weakness")
assert diag_inc.incompletely_classified == True

# compare consensus in diagnosis. Here laryngeal_tuberculosis has more consensus than intubation_granuloma
diag_1 = diagnosis_map.get("intubation_granuloma")
diag_2 = diagnosis_map.get("laryngeal_tuberculosis")
assert diag_1 < diag_2

# For mapping any given diagnosis to a single parent we use the classification that had the max vote
diag_dissensus = diagnosis_map.get("intubation_granuloma")
assert diag_dissensus.best_parent_link.parent.name == "organic_trauma_internal"

# Get all possible parents of a class, with their vote percentage
diag_dissensus = diagnosis_map.get("intubation_granuloma")
expected_parents = [
    "organic_inflammatory_non_infective",
    "organic_structural_structural_abnormality",
    "organic_trauma_internal"
]
expected_votes = [0.29, 0.29, 0.43]
for parent_link in diag_dissensus.parents:
    assert parent_link.parent.name in expected_parents
    assert parent_link.weight in expected_votes

# Get all votes of different clinicians
diag_dissensus = diagnosis_map.get("intubation_granuloma")
assert diag_dissensus.votes["clinician 1"] == "organic > trauma > internal"
assert diag_dissensus.votes["clinician 2"] == "organic > trauma > internal"
assert diag_dissensus.votes["clinician 3"] == "organic > trauma > internal"
assert diag_dissensus.votes["clinician 4"] == "organic > inflammatory > non_infective"
assert diag_dissensus.votes["clinician 5"] == "organic > inflammatory > non_infective"
assert diag_dissensus.votes["clinician 6"] == "organic > structural > structural_abnormality"
assert diag_dissensus.votes["clinician 7"] == "organic > structural > structural_abnormality"

# List all pathologies under a parent
expected_diags = [
    "arytenoid_dislocation",
    "laryngeal_trauma",
    "laryngeal_trauma_blunt",
    "organic_trauma_external",
]
for diag in diagnosis_map.find(name="organic_trauma_external"):
    assert diag.name in expected_diags
```

## How was this created

### Databases used

AVFAD [[1]](#[1]) , MEEI [[2]](#[2]), SVD [[3]](#[3]), Torgo [[4]](#[4]), UASpeech [[5]](#[5]), Uncommon Voice [[6]](#[6]), Voiced [[7]](#[7])

### USVAC 2025

Classification labels from all the databases mentioned above were extracted, translated to english where needed (e.g. german to english for SVD), and de-duplicated. These labels were then presented to 7 clinicians (2 otorhinolaryngologists, 5 speech pathologists) at the University of Sydney Voice Activity Clinic (USVAC) who classified the labels into a classification system [[8]](#[8]) in Qualtrics. The votes were then extracted from Qualtrics in a [spreadsheet](tools/List%20of%20diagnosis.xlsx), which was processed into a [classification map](divr_diagnosis/diagnosis_maps/USVAC_2025.yml).

### Other systems

Research on multi-class classification system was identified as part of a scoping review [[9]](#[9]). While a lot of research used a subset of data available to them, or used exact diagnostic labels to classify data, some research grouped certain classes together. We extracted the classification systems used by these papers and implemented them in this module. After the classification system was initially implemented, we went over all the classes available in databases listed above and ensured that all those classes were allocated to one of the classes in the system. Since, we can not make assumptions of how the clinicians of the input research would have mapped the unmapped labels, only labels that **closely** matched any existing label were assigned as such. All other labels were marked as unclassified.

## How to cite

Coming soon

## References

<a id="[1]">[1]</a> L. M. T. Jesus, I. Belo, J. Machado, and A. Hall, “The Advanced Voice Function Assessment Databases (AVFAD): Tools for Voice Clinicians and Speech Research,” in Advances in Speech-language Pathology, F. D. M. Fernandes, Ed., InTech, 2017. doi: 10.5772/intechopen.69643.<br/>
<a id="[2]">[2]</a> Massachusetts Eye and Ear Infirmary, “Voice disorders database, version. 1.03 (cd-rom).” Lincoln Park, NJ: Kay Elemetrics Corporation.<br/>
<a id="[3]">[3]</a> B. Woldert-Jokisz, “Saarbruecken voice database.” 2007.<br/>
<a id="[4]">[4]</a> F. Rudzicz, A. K. Namasivayam, and T. Wolff, “The TORGO database of acoustic and articulatory speech from speakers with dysarthria,” Lang Resources & Evaluation, vol. 46, no. 4, pp. 523–541, Dec. 2012, doi: 10.1007/s10579-011-9145-0.<br/>
<a id="[5]">[5]</a> H. K. Kim et al., “UASpeech.” IEEE DataPort. doi: 10.21227/F9TC-AB45.<br/>
<a id="[6]">[6]</a> M. Moore, P. Papreja, M. Saxon, V. Berisha, and S. Panchanathan, “UncommonVoice: A Crowdsourced Dataset of Dysphonic Speech,” in Interspeech 2020, ISCA, Oct. 2020, pp. 2532–2536. doi: 10.21437/Interspeech.2020-3093.<br/>
<a id="[7]">[7]</a> U. Cesari, G. De Pietro, E. Marciano, C. Niri, G. Sannino, and L. Verde, “A new database of healthy and pathological voices,” Computers & Electrical Engineering, vol. 68, pp. 310–321, May 2018, doi: 10.1016/j.compeleceng.2018.04.008.<br/>
<a id="[8]">[8]</a> C. L. Payten, G. Chiapello, K. A. Weir, and C. J. Madill, “Frameworks, Terminology and Definitions Used for the Classification of Voice Disorders: A Scoping Review.,” J Voice, no. bu2, 8712262, 2022, doi: 10.1016/j.jvoice.2022.02.009.<br/>
<a id="[9]">[9]</a> R. Gupta, D. R. Gunjawate, D. D. Nguyen, C. Jin, and C. Madill, “Voice disorder recognition using machine learning: a scoping review protocol,” BMJ Open, vol. 14, no. 2, 2024, doi: 10.1136/bmjopen-2023-076998.
