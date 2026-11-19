"""Libro de Excel en memoria con varias hojas (consenso, escenarios, reposición, parámetros)."""
import io

import pandas as pd


def build_workbook(sheets):
    """sheets: {nombre: DataFrame}. Nombres de hoja de hasta 31 caracteres. Devuelve bytes de un .xlsx."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xw:
        for name, df in sheets.items():
            safe = "".join(ch for ch in name if ch not in '[]:*?/\\')[:31] or "Hoja"
            df.to_excel(xw, sheet_name=safe, index=False)
            ws = xw.sheets[safe]
            ws.freeze_panes = "A2"
            for i, col in enumerate(df.columns, start=1):
                width = max([len(str(col))] + [len(str(v)) for v in df[col].head(200)]) + 2
                ws.column_dimensions[ws.cell(row=1, column=i).column_letter].width = min(width, 45)
    return buf.getvalue()
