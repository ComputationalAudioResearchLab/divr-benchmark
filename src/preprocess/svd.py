import json
from pathlib import Path
from .base import BaseProcessor
from .processed import ProcessedFile, ProcessedSession


class SVD(BaseProcessor):
    async def __call__(self, source_path: Path, dest_path: Path) -> None:
        db_key = "svd"
        print(f"processing {db_key}")
        data = []

        with open(f"{source_path}/data.json", "r") as inputfile:
            for speaker_id, val in json.loads(inputfile.read()).items():
                gender = val["gender"]
                for session in val["sessions"]:
                    session_id = session["session_id"]
                    age = session["age"]
                    classification = session["classification"]
                    pathologies = session["pathologies"]
                    files = session["files"]
                    diagnosis = pathologies if pathologies != "" else classification
                    data += [
                        ProcessedSession(
                            db=db_key,
                            id=f"{speaker_id}.{session_id}",
                            age=age,
                            gender=gender,
                            diagnosis=[x.strip() for x in diagnosis.split(",")],
                            files=[
                                ProcessedFile(
                                    path=Path(
                                        f"{source_path}/{classification}/{gender}/{speaker_id}/{session_id}/{file.split('file=')[1]}.wav"
                                    )
                                )
                                for file in files
                            ],
                        )
                    ]
        with open(f"{dest_path}/{db_key}.json", "w") as outfile:
            json.dump(data, outfile, indent=2, default=vars)
