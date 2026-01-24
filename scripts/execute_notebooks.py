from pathlib import Path
import sys
import asyncio
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"
ORDER = [
    "01_eda.ipynb",
    "02_limpeza_e_engenharia.ipynb",
    "03_analise_perguntas_de_negocio.ipynb",
    "04_relatorio.ipynb",
]

def run_one(nb_path: Path):
    print(f"Executing {nb_path}")
    with nb_path.open("r", encoding="utf-8") as f:
        nb = nbf.read(f, as_version=4)
    # Corrige loop no Windows para compatibilidade com ZMQ
    if sys.platform.startswith("win"):
        try:
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        except Exception:
            pass
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})
    with nb_path.open("w", encoding="utf-8") as f2:
        nbf.write(nb, f2)
    print(f"Done {nb_path}")

def main():
    for name in ORDER:
        run_one(NB_DIR / name)

if __name__ == "__main__":
    main()
