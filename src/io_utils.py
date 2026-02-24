from pathlib import Path
import shutil
import re
import pandas as pd
from .config import RAW_DIR, DEFAULT_FILES

def copy_raw_files():
    copied = {}
    for k, p in DEFAULT_FILES.items():
        if p and p.exists():
            target = RAW_DIR / p.name
            if str(p.resolve()) != str(target.resolve()):
                shutil.copy2(p, target)
            copied[k] = target
    return copied

def read_excel_any(path: Path) -> dict[str, pd.DataFrame]:
    xl = pd.ExcelFile(path)
    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name)
        df.columns = [re.sub(r"\s+", "_", str(c).strip().lower()) for c in df.columns]
        sheets[name] = df
    return sheets
