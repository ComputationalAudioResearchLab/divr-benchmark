from typing import Tuple


def age_to_bracket(age: int | None) -> Tuple[int, int] | None:
    if age is None:
        return None
    lower = (age // 10) * 10
    upper = lower + 10
    return (lower, upper)
