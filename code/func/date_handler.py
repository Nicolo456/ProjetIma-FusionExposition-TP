import re
from datetime import datetime
import time


def date_format_to_regex(date_format):
    """
    Convertit un format de date en une expression régulière.

    Args:
        date_format (str): Format de date compatible avec `datetime.strptime`.

    Returns:
        str: Expression régulière correspondante.
    """
    mapping = {
        "%Y": r"\d{4}",      # Année à 4 chiffres
        "%y": r"\d{2}",      # Année à 2 chiffres
        "%m": r"\d{2}",      # Mois (01-12)
        "%d": r"\d{2}",      # Jour (01-31)
        "%H": r"\d{2}",      # Heure (00-23)
        "%M": r"\d{2}",      # Minute (00-59)
        "%S": r"\d{2}"       # Seconde (00-59)
    }
    # Échappe les caractères spéciaux dans le format
    regex = re.escape(date_format)
    # Remplace les codes de format par les regex correspondantes
    for key, value in mapping.items():
        regex = regex.replace(re.escape(key), value)
    return regex


def get_latest_file(files, date_format):
    """
    Récupère le fichier avec la date la plus récente.

    Args:
        files (list of str): Liste des noms de fichiers.
        date_format (str): Format de la date dans les noms de fichiers.

    Returns:
        str: Nom du fichier avec la date la plus récente.
    """
    # Expression régulière pour extraire la date
    date_regex = re.compile(date_format_to_regex(
        date_format))  # Par défaut YYYY-MM-DD

    latest_file = None
    latest_date = None

    for file in files:
        match = date_regex.search(file)
        if match:
            file_date = datetime.strptime(match.group(), date_format)
            if latest_date is None or file_date > latest_date:
                latest_date = file_date
                latest_file = file

    return latest_file


def time_counter(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()  # Record start time
        result = func(*args, **kwargs)  # Execute the function
        end_time = time.time()  # Record end time
        print(f"Execution time of {func.__name__}: {
              end_time - start_time:.6f} seconds")
        return result
    return wrapper
