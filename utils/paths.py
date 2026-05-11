from pathlib import Path

def resolve_path(base : str | Path , path : str | Path):
    path = Path(path)
    if path.is_absolute():
        return path.resolve()
    
    return Path(base).resolve() / path


def is_binary_file(path : str | Path) -> bool  :
    try :
       with open(path , "rb") as f:
        chunk = f.read(8192)
        return b"\x00" in chunk
    except (OSError , IOError):
        return False

def display_path_rel_to_cwd(path: str, cwd: Path | None) -> str:
    try:
        p = Path(path)
    except Exception:
        return path

    if cwd:
        try:
            return str(p.relative_to(cwd))
        except ValueError:
            pass

    return str(p)

