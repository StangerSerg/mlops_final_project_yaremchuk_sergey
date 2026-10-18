from pathlib import Path

import pandas as pd


def upload_prepared_data(
        data: pd.DataFrame,
        path: Path
) -> tuple[bool, str]:
    """Загружает подготовленные данные во внешнее хранилище
     или сохраняет файл локально

     Args:
         data: Подготовленный DataFrame для сохранения
         path: Путь сохранения для фолбэка

    Returns:
        tuple[bool, str]: (успех, сообщение об ошибке). Сообщение
            пустое при успешной записи, иначе содержит repr исключения.
     """
    try:
        data.to_parquet(path, index=False)
        return True, ""

    except Exception as e:
        return False, repr(e)
