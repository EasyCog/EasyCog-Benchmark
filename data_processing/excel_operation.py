from openpyxl import load_workbook
import os


def read_xlsx_to_dict(file, sh="Sheet1"):
    sheet = load_workbook(file, data_only=True)[sh]
    rows = list(sheet.rows)
    cols = list(sheet.columns)
    title = []
    for i in rows[0]:
        title.append(i.value)

    cols_val = []

    for c in cols:
        data = []
        skip = 1
        for e in c:
            if skip == 0:
                data.append(e.value)
            else:
                skip -= 1
        cols_val.append(data)
    return dict(zip(title, cols_val))