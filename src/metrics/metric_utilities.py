import numpy as np
import pandas as pd

from typing import Union


class ParticipantVisibleError(Exception):
    pass


class HostVisibleError(Exception):
    pass


def validate_probabilities(df: pd.DataFrame, df_name: str):
    """推論結果のvalidation"""
    if not pd.api.types.is_numeric_dtype(df.values):
        raise ParticipantVisibleError(f"All target values in {df_name} must be numeric")

    # if df.min().min() < 0:
    #     raise ParticipantVisibleError(f"All target values in {df_name} must be at least 0")

    if df.max().max() > 1:
        raise ParticipantVisibleError(f"All target values in {df_name} must be no greater than 1")

    if not np.allclose(df.sum(axis=1), 1):
        raise ParticipantVisibleError(f"Target values in {df_name} do not add to one within all rows")


def safe_call_score(metric_function, solution, submission, **metric_func_kwargs):
    try:
        score_result = metric_function(solution, submission, **metric_func_kwargs)
    except Exception as err:
        error_message = str(err)
        if err.__class__.__name__ == "ParticipantVisibleError":
            raise ParticipantVisibleError(error_message)
        elif err.__class__.__name__ == "HostVisibuleError":
            raise HostVisibleError
        else:
            if _treat_as_participant_error(error_message=error_message, solution=solution):
                raise ParticipantVisibleError(error_message)
            else:
                raise err

    return score_result


def _treat_as_participant_error(error_message: str, solution: Union[pd.DataFrame, np.ndarray]) -> bool:
    # boolをnumericとして扱うcheck
    if isinstance(solution, pd.DataFrame):
        solution_is_all_numeric = all([pd.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
        solution_has_bools = any([pd.api.types.is_numeric_dtype(x) for x in solution.dtypes.values])
    elif isinstance(solution, np.ndarray):
        solution
