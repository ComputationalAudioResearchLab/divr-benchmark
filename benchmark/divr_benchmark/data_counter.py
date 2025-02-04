from tqdm import tqdm
from pathlib import Path
from divr_diagnosis import diagnosis_maps

from .task_generator import generator_map, VERSIONS


async def data_counter(
    version: VERSIONS,
    data_store_path: Path,
    output_path: Path,
) -> None:
    generator = generator_map[version]
    data_store_path = data_store_path
    db_names = [
        "avfad",
        "meei",
        "svd",
        "torgo",
        "uaspeech",
        "uncommon_voice",
        "voiced",
    ]
    diag_maps = [
        diagnosis_maps.Benba_2017,
        diagnosis_maps.Compton_2022,
        diagnosis_maps.Cordeiro_2015,
        diagnosis_maps.daSilvaMoura_2024,
        diagnosis_maps.deMoraesLimaMarinus_2013,
        diagnosis_maps.FEMH_2018,
        diagnosis_maps.Firdos_2016,
        diagnosis_maps.Kim_2024,
        diagnosis_maps.Sztaho_2018,
        diagnosis_maps.Tsui_2018,
        diagnosis_maps.USVAC_2025,
        diagnosis_maps.Zaim_2023,
    ]

    with open(output_path, "w") as output_file:
        output_file.write("# DB Sessions per classification system\n")
        output_file.write(
            "Count of sessions for different databases for different classification systems at all available levels of classification in the classification system.\n"
        )
        for diag_map in tqdm(diag_maps, desc="counting per diag_map", position=0):
            diag_map = diag_map()
            output_file.write(f"## {diag_map.name}\n")
            for db_name in tqdm(db_names, desc="counting per db", position=1):
                db = await generator.count_for_diag_map(
                    db_name=db_name,
                    source_path=data_store_path,
                    diag_map=diag_map,
                )
                output_file.write(f"### {db_name}\n")
                for level in range(diag_map.max_diag_level + 1):
                    counts = db.count_per_diag(level=level)
                    output_file.write(f"* **Level {level}:**\n")
                    for diag, count in counts.items():
                        output_file.write(f"\t * **{diag.name}:** {count}\n")
