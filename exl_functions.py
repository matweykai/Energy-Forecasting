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


def write_data(file_path: str, column: int, data: list) -> None:
    """Writes data to existing excel file in the specified column"""
    try:
        workbook = load_workbook(file_path)
        worksheet = workbook.active
        # Filling cells with values
        for row_ind, value in enumerate(data, 2):
            temp_cell = worksheet.cell(row=row_ind, column=column + 1)
            temp_cell.value = value

        workbook.save(file_path)
    except InvalidFileException:
        # TODO: Add logging
        pass
