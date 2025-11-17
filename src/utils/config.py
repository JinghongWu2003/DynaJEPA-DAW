
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in {"yes", "true", "t", "1", "y"}:
        return True
    if v.lower() in {"no", "false", "f", "0", "n"}:
        return False
    raise ValueError(f"Boolean value expected, got {v}")
