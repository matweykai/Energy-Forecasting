from openpyxl.reader.excel import load_workbook
from openpyxl.utils.exceptions import InvalidFileException
from typing import Optional, List


def get_data(file_path: str, column: int) -> Optional[List[int]]:
    """Reads data from excel file and returns it as list"""
    try:
        worksheet = load_workbook(file_path).active

        result_values = list()
        # Getting column with power values
        for row in worksheet.iter_rows(min_row=2, min_col=column + 1, max_col=column + 1, values_only=True):
            result_values.append(row[0])

        return result_values
    except InvalidFileException:
        # TODO: Add logging
        return None
