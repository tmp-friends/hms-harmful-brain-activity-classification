import numpy as np
import pandas as pd

from typing import Optional

from .metric_utilities import validate_probabilities, safe_call_score


class ParticipantVisibleError(Exception):
    pass


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str,
    epsilon: float = 10**-15,
    micro_average: bool = True,
    sample_weights_column_name: Optional[str] = None,
) -> float:
    del solution[row_id_column_name]
    del submission[row_id_column_name]

    sample_weights = None
    if sample_weights_column_name:
        if sample_weights_column_name not in solution.columns:
            raise ParticipantVisibleError(f"{sample_weights_column_name} not found in solution columns")

        sample_weights = solution.pop(sample_weights_column_name)

    if sample_weights_column_name and not micro_average:
        raise ParticipantVisibleError("Sample weights are only valid if `micro_average` is `True`")

    for col in solution.columns:
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Missing submission column {col}")

    validate_probabilities(solution, "solution")
    validate_probabilities(submission, "submission")

    return safe_call_score(
        metric_function=kl_divergence,
        solution=solution,
        submission=submission,
        epsilon=epsilon,
        micro_average=micro_average,
        sample_weights=sample_weights,
    )


def kl_divergence(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    epsilon: float,
    micro_average: bool,
    sample_weights: Optional[pd.Series],
):
    # Overwrite solution for convenience
    for col in solution.columns:
        if not pd.api.types.is_float_dtype(solution[col]):
            solution[col] = solution[col].astype(float)

        submission[col] = np.clip(submission[col], epsilon, 1 - epsilon)

        y_nonzero_indices = solution[col] != 0
        solution[col] = solution[col].astype(float)
        solution.loc[y_nonzero_indices, col] = solution.loc[y_nonzero_indices, col] * np.log(
            solution.loc[y_nonzero_indices, col] / submission.loc[y_nonzero_indices, col]
        )

        # Set the loss equal to zero where y_true equals zero following the scipy convention:
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.rel_entr.html#scipy.special.rel_entr
        solution.loc[~y_nonzero_indices, col] = 0

    if micro_average:
        return np.average(solution.sum(axis=1), weights=sample_weights)
    else:
        return np.average(solution.mean())
