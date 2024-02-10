val_map = {
    "f": "female",
    "m": "male",
    "female": "female",
    "male": "male",
    "unknown": "unknown",
}


class Gender(str):
    @staticmethod
    def format(value: str) -> str:
        return val_map[value.strip().lower()]
