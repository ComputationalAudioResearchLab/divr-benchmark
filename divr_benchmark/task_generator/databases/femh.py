import pandas as pd
from pathlib import Path
from typing import List, Set
from divr_diagnosis import DiagnosisMap

from .Base import Base
from .gender import Gender
from ...prepare_dataset.processed import (
    ProcessedSession,
    ProcessedFile,
)
import re

class FEMH(Base):

    DB_NAME = "femh"

    async def _collect_diagnosis_terms(self, source_path: Path) -> Set[str]:
        df = self.__read_data(source_path=source_path)
        return set(df["Disease category"].tolist())

    async def prepare_dataset(
        self,
        source_path: Path,
        allow_incomplete_classification: bool,
        min_tasks: int | None,
        diagnosis_map: DiagnosisMap,
    ) -> List[ProcessedSession]:
        sessions = []
        df = self.__read_data(source_path)

        # Iterate through each row of data in the Excel sheet (excluding header, 2000 patient records)
        for _, row in df.iterrows():
            speaker_id = row["ID"]  # Patient ID, used as speaker identifier
            diagnosis = row[
                "Disease category"
            ]  # Disease category, original diagnosis information

            # Map original diagnosis terms to standardized diagnosis categories
            diagnosis = (
                diagnosis_map[diagnosis]
                if diagnosis in diagnosis_map
                else diagnosis_map.unclassified  # Unclassified diagnoses use default classification
            )

            # Extract patient basic information
            age = int(row["Age"])  # Age, converted to integer
            gender = Gender.format(row["Sex"])  # Gender, standardized format

            # Decide whether to include this data based on classification completeness
            if allow_incomplete_classification or not diagnosis.incompletely_classified:
                num_files = 1  # Each patient has only one audio file
                # Check if minimum task count requirement is met
                if min_tasks is None or num_files >= min_tasks:
                    sessions += [
                        ProcessedSession(
                            id=f"femh_{speaker_id}",  # Session ID: femh_ prefix + patient ID
                            speaker_id=speaker_id,  # Speaker ID
                            age=age,  # Patient age
                            gender=gender,  # Patient gender
                            diagnosis=[diagnosis],  # Diagnosis list (single diagnosis)
                            files=[
                                ProcessedFile(
                                    path=Path(f"{source_path}/selectwav/{speaker_id}.wav")
                                )
                            ],  # Audio file list
                            num_files=num_files,  # File count: 1
                        )
                    ]
        return sessions

    def __read_data(self, source_path):
        """
        Read raw data from FEMH database

        Data structure description:
        - File location: selectwav/medicalhistory.xlsx
        - Total rows: 2001 (including header)
        - Column structure: ID | Sex | Age | Disease category
        - Data type: Patient medical record information

        Args:
            source_path: FEMH database root directory path

        Returns:
            tuple: (data_path, all_data)
                - data_path: Data root path string
                - all_data: Cleaned pandas DataFrame

        Raises:
            FileNotFoundError: When Excel file is not found
            ValueError: When Excel file column structure doesn't match expectations
        """
       

        

        # Read Excel file (contains 2001 rows: 1 header row + 2000 data rows)
        df = pd.read_excel(f"{source_path}/selectwav/medicalhistory.xlsx")

        # Keep only the four required columns
        df = df[["ID", "Sex", "Age", "Disease category"]]

        # Convert gender column to string type, convert 1 and 2 to 'male' and 'female'
        df["Sex"] = df["Sex"].astype(str).apply(self.__clean_sex)
        # Clean diagnosis terms
        df["Disease category"] = df[
            "Disease category"
        ].apply(self.__clean_diagnosis)
       
        return df
    def __clean_diagnosis(self, diagnosis: str) -> str:
        diagnosis = diagnosis.lower().strip()
        diagnosis = re.sub(r"[0-9\.]+", "", diagnosis)
        diagnosis = diagnosis.replace("’", "'")
        return diagnosis
    
    def __clean_sex(self, sex: str) -> str:
        sex = sex.replace("1", "male").replace("2", "female")
        return sex
