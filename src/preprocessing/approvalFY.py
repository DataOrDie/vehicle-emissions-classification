import pandas as pd

def preprocess_approvalfy(df, option="A", source_col="ApprovalFY"):
    """
    Preprocesamiento de ApprovalFY.
    Opción A: Eliminar la columna para evitar multicolinealidad con ApprovalDate.
    Opción B: Mantener la columna, rellenar valores nulos con la moda y convertir a entero.
    """
    df_out = df.copy()
    
    if source_col not in df_out.columns:
        return df_out
        
    if option == "A":
        # Opción A: Eliminar la columna
        df_out = df_out.drop(columns=[source_col])
        
    elif option == "B":
        # Opción B: Limpiar valores nulos y convertir a formato numérico entero
        df_out[source_col] = pd.to_numeric(df_out[source_col], errors='coerce')
        mode_val = df_out[source_col].mode()[0]
        df_out[source_col] = df_out[source_col].fillna(mode_val).astype(int)
        
    else:
        # Error en caso de que alguien ponga una opción que no existe
        raise ValueError(f"Opción no válida para {source_col}: {option}")
        
    return df_out
