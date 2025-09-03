import time
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, LiteralString
import numpy as np

def ensure_dir(path: Path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")

def save_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]):
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w",encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False)+"\n")

def load_jsonl(path: str | Path) -> Iterable[Dict[str, Any]]:
    path = Path(path)
    with path.open("r",encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def set_seed(seed:int = 42):
    import random
    np.random.seed(seed)
    random.seed(seed)

@dataclass
class Example:
    q: str
    a: str
    meta: Dict[str, Any]