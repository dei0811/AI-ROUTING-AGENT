"""
Genera el dataset de features para el entrenamiento del router.

Carga el dataset crudo (prompts + posiblemente etiquetas), extrae las
features de cada prompt usando `features.extractor` en paralelo, y
guarda el resultado como un archivo Parquet listo para entrenar el
modelo XGBoost.
"""

import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from dataset.loader import load_dataset
from ..features.extractor import extract_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Rutas relativas a la raíz del proyecto (ai-router/)
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "features_dataset.parquet"

# Número de procesos worker. Deja 1 core libre para no congelar la máquina.
N_WORKERS = max(1, cpu_count() - 1)

# Número de registros que cada worker procesa por lote. Valores más
# altos reducen overhead de comunicación entre procesos.
CHUNK_SIZE = 200


def _normalize_dataset(dataset):
    """Convierte el dataset a una lista de dicts, sea cual sea su forma.

    Soporta pandas.DataFrame, lista de dicts o lista de strings.
    """
    if isinstance(dataset, pd.DataFrame):
        return dataset.to_dict(orient="records")

    if isinstance(dataset, list):
        return [row if isinstance(row, dict) else {"prompt": row} for row in dataset]

    raise TypeError(f"Formato de dataset no soportado: {type(dataset)}")


def _extract_row(row: dict):
    """Extrae features de un único registro. Diseñada para multiprocessing.

    Debe ser una función a nivel de módulo (no un closure ni método)
    para que sea "picklable" y funcione con multiprocessing en Windows.

    Args:
        row: Un registro del dataset, con al menos la clave "prompt".

    Returns:
        Un dict con las columnas originales + las features extraídas,
        o None si el registro es inválido o falló la extracción.
    """
    prompt = row.get("prompt")

    if not prompt or not isinstance(prompt, str):
        return None

    try:
        features = extract_features(prompt)
    except Exception:
        # No se puede usar `logger` de forma fiable dentro de un
        # worker de multiprocessing; se retorna None y se cuenta
        # como error en el proceso principal.
        return None

    merged_row = dict(row)
    merged_row.update(features)
    return merged_row


def build_feature_dataset(dataset) -> pd.DataFrame:
    """Extrae las features de cada prompt del dataset, en paralelo.

    Args:
        dataset: DataFrame o lista de registros. Cada registro debe
            tener al menos la clave/columna "prompt". Cualquier otra
            columna (por ejemplo una etiqueta "target" o "model") se
            conserva junto con las features extraídas.

    Returns:
        DataFrame donde cada fila es un prompt original + sus features.
    """
    records = _normalize_dataset(dataset)
    total = len(records)

    logger.info("Procesando %d registros con %d workers...", total, N_WORKERS)

    with Pool(processes=N_WORKERS) as pool:
        results = list(
            tqdm(
                pool.imap(_extract_row, records, chunksize=CHUNK_SIZE),
                total=total,
                desc="Extrayendo features",
                unit="prompt",
            )
        )

    processed_rows = [r for r in results if r is not None]
    errors = total - len(processed_rows)

    logger.info(
        "Procesados %d registros correctamente (%d con error/omitidos).",
        len(processed_rows),
        errors,
    )
    return pd.DataFrame.from_records(processed_rows)


def main() -> None:
    """Punto de entrada: carga el dataset, extrae features y guarda el parquet."""
    logger.info("Cargando dataset...")
    dataset = load_dataset()

    if dataset is None:
        raise RuntimeError(
            "load_dataset() devolvió None. Verifica que la función "
            "tenga un 'return' explícito y que el archivo de datos exista."
        )

    df = build_feature_dataset(dataset)

    if df.empty:
        raise RuntimeError("El DataFrame resultante está vacío. Revisa el dataset de entrada.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_FILE, index=False, engine="pyarrow")

    logger.info("Dataset de features guardado en: %s", OUTPUT_FILE)
    logger.info("Shape final: %s", df.shape)
    logger.info("Columnas: %s", list(df.columns))


if __name__ == "__main__":
    main()