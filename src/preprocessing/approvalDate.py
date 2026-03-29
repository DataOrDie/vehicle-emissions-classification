import pandas as pd

def preprocess_approvaldate(df, option="A", source_col="ApprovalDate"):
    """
    Preprocesamiento de ApprovalDate.
    Opción A: Convertir a fecha, extraer Año y Mes numéricos, y eliminar la original.
    Opción B: Solo convertir el texto a formato datetime de Pandas y conservarlo.
    """
    df_out = df.copy()
    
    if source_col not in df_out.columns:
        return df_out
        
    # Convertimos a formato datetime (útil para ambas opciones)
    fechas = pd.to_datetime(df_out[source_col], format='%d-%b-%y', errors='coerce')
    
    if option == "A":
        # Opción A: Extraer características (Año y Mes) y eliminar la columna original
        df_out['ApprovalYear'] = fechas.dt.year
        df_out['ApprovalMonth'] = fechas.dt.month
        
        # Rellenar valores nulos con el valor más frecuente (moda)
        moda_y = df_out['ApprovalYear'].mode()[0]
        moda_m = df_out['ApprovalMonth'].mode()[0]
        df_out['ApprovalYear'] = df_out['ApprovalYear'].fillna(moda_y).astype(int)
        df_out['ApprovalMonth'] = df_out['ApprovalMonth'].fillna(moda_m).astype(int)
        
        # Eliminar la columna original de texto
        df_out = df_out.drop(columns=[source_col])
        
    elif option == "B":
        # Opción B: Conservar la columna original pero transformada a objeto datetime
        df_out[source_col] = fechas
        
    else:
        # Error en caso de que alguien ponga una opción que no existe
        raise ValueError(f"Opción no válida para {source_col}: {option}")
        
    return df_out
