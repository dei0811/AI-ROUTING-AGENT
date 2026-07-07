"""
Loader genérico de datasets.

Resuelve rutas de forma robusta usando `__file__` como ancla, por lo
que funciona sin importar desde qué directorio se ejecute el script
(python -m ..., doble click, IDE, etc.) ni cómo esté organizada la
carpeta de datos, siempre que exista dentro del proyecto.

Soporta múltiples formatos de archivo: .json, .jsonl, .csv, .parquet,
.tsv, .xlsx.
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Ancla física del proyecto: la carpeta que contiene este archivo.
# parents[0] = dataset/, parents[1] = src/, parents[2] = raíz del proyecto.
_THIS_FILE = Path(__file__).resolve()

# Nombres de carpeta típicos donde podría vivir la raíz del proyecto.
# Se usa como fallback si la heurística de __file__ no es suficiente.
_PROJECT_MARKERS = (".git", "requirements.txt", "pyproject.toml", "setup.py")


def find_project_root(start: Optional[Path] = None) -> Path:
    """Encuentra la raíz del proyecto subiendo desde `start`.

    Busca hacia arriba en el árbol de directorios hasta encontrar una
    carpeta que contenga alguno de los `_PROJECT_MARKERS` (p. ej.
    ".git" o "requirements.txt"). Si no encuentra ninguno, usa como
    fallback la carpeta dos niveles arriba de este archivo (asumiendo
    la estructura src/dataset/loader.py -> raíz).

    Args:
        start: Ruta desde la cual empezar a buscar. Por defecto, la
            ubicación de este archivo.

    Returns:
        Path absoluto a la raíz del proyecto.
    """
    current = (start or _THIS_FILE).resolve()
    if current.is_file():
        current = current.parent

    for candidate in [current, *current.parents]:
        if any((candidate / marker).exists() for marker in _PROJECT_MARKERS):
            return candidate

    # Fallback: estructura conocida src/dataset/loader.py -> raíz
    fallback = _THIS_FILE.parents[2]
    logger.warning(
        "No se encontró un marcador de proyecto (%s). Usando fallback: %s",
        ", ".join(_PROJECT_MARKERS),
        fallback,
    )
    return fallback


def find_data_file(filename: str, search_root: Optional[Path] = None) -> Path:
    """Busca un archivo por nombre dentro del proyecto, en cualquier subcarpeta.

    Útil cuando no se sabe (o puede cambiar) si el dataset vive en
    "data/", "data/processed/", "data/raw/", etc.

    Args:
        filename: Nombre exacto del archivo a buscar (p. ej. "dataset.json").
        search_root: Carpeta desde la cual buscar recursivamente. Por
            defecto, la raíz del proyecto detectada automáticamente.

    Returns:
        Path absoluto al primer archivo encontrado con ese nombre.

    Raises:
        FileNotFoundError: Si no se encuentra ningún archivo con ese nombre.
    """
    root = search_root or find_project_root()
    matches = list(root.rglob(filename))

    if not matches:
        raise FileNotFoundError(
            f"No se encontró ningún archivo llamado '{filename}' "
            f"dentro de '{root}'. Verifica el nombre o la ubicación."
        )

    if len(matches) > 1:
        logger.warning(
            "Se encontraron %d archivos llamados '%s'. Usando el primero: %s",
            len(matches),
            filename,
            matches[0],
        )

    return matches[0]


def _read_by_extension(path: Path, **kwargs) -> pd.DataFrame:
    """Lee un archivo tabular según su extensión.

    Args:
        path: Ruta al archivo a leer.
        **kwargs: Argumentos adicionales pasados directamente a la
            función de pandas correspondiente (p. ej. lines=True).

    Returns:
        DataFrame con el contenido del archivo.

    Raises:
        ValueError: Si la extensión no está soportada.
    """
    suffix = path.suffix.lower()

    readers = {
        ".json": lambda p: pd.read_json(p, **kwargs),
        ".jsonl": lambda p: pd.read_json(p, lines=True, **kwargs),
        ".csv": lambda p: pd.read_csv(p, **kwargs),
        ".tsv": lambda p: pd.read_csv(p, sep="\t", **kwargs),
        ".parquet": lambda p: pd.read_parquet(p, **kwargs),
        ".xlsx": lambda p: pd.read_excel(p, **kwargs),
    }

    if suffix not in readers:
        raise ValueError(
            f"Extensión '{suffix}' no soportada. "
            f"Formatos válidos: {', '.join(readers.keys())}"
        )

    return readers[suffix](path)


def load_dataset(
    filename: str = "dataset.json",
    data_dir: Optional[str] = None,
    **read_kwargs,
) -> pd.DataFrame:
    """Carga un dataset desde cualquier ubicación dentro del proyecto.

    Resuelve la ruta de forma robusta:
    1. Si se pasa `data_dir`, busca `filename` directamente ahí.
    2. Si no, busca `filename` recursivamente desde la raíz del
       proyecto (detectada vía `find_project_root`), sin importar en
       qué subcarpeta de datos se encuentre.

    Args:
        filename: Nombre del archivo del dataset (con extensión).
            Por defecto "dataset.json".
        data_dir: Ruta absoluta o relativa a la raíz del proyecto donde
            buscar el archivo directamente. Si se omite, se hace
            búsqueda recursiva automática.
        **read_kwargs: Argumentos extra pasados al lector de pandas
            correspondiente (p. ej. lines=True para JSON Lines).

    Returns:
        pandas.DataFrame con el contenido del dataset.

    Raises:
        FileNotFoundError: Si el archivo no existe en la ruta indicada
            ni se encuentra en la búsqueda recursiva.
        ValueError: Si la extensión del archivo no está soportada.
    """
    project_root = find_project_root()

    if data_dir is not None:
        candidate_dir = Path(data_dir)
        if not candidate_dir.is_absolute():
            candidate_dir = project_root / candidate_dir
        file_path = candidate_dir / filename

        if not file_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo en: {file_path}")
    else:
        file_path = find_data_file(filename, search_root=project_root)

    logger.info("Cargando dataset desde: %s", file_path)
    df = _read_by_extension(file_path, **read_kwargs)
    logger.info("Dataset cargado: %d filas, %d columnas.", len(df), len(df.columns))

    return df