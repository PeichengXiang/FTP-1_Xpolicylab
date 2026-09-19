"""Resolve independent action arrays without importing the heavy dataset loader."""


def resolve_ftp1_action_array_key(data_key_set: set[str], state_key: str) -> str:
    """Prefer an independent ``{state_key}_action`` array when the converter exported one.

    Official UniVTAC / FTP-1 zarrs only store one proprio array per modality, so
    the loader historically supervised on that same array at future indices
    (next-state). Spark0 real HDF5 has a distinct ``action/*`` group; those
    values must not be replaced by ``state[t+1]``.
    """
    action_key = f"{state_key}_action"
    if action_key in data_key_set:
        return action_key
    return state_key
