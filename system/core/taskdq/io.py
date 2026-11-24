"""Small artifact helpers shared by the runtime and command line tools."""
import json
from pathlib import Path
import numpy as np
import pandas as pd


def plain(value):
    if isinstance(value, dict): return {str(k): plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)): return [plain(v) for v in value]
    if isinstance(value, Path): return str(value)
    if isinstance(value, np.generic): value=value.item()
    if value is None or pd.isna(value): return None
    if isinstance(value, float) and not np.isfinite(value): return str(value)
    return value


def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(plain(value),ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))


def append(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:f.write(json.dumps(plain(value),ensure_ascii=False,allow_nan=False)+'\n')


def records(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def resolved(base, value):
    p=Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(base)/p).resolve()
