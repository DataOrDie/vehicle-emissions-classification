import pandas as pd

def preprocess_franchise_code(df, option="binary", source_col="FranchiseCode"):
    """
    Preprocesa la columna FranchiseCode.
    
    Opciones:
    - 'binary': Convierte a 0 (No franquicia: códigos 0 o 1) y 1 (Sí franquicia: > 1).
    - 'raw': Devuelve la columna original sin cambios (rellenando nulos con 0).
    """
    df_out = df.copy()
    
    if source_col not in df_out.columns:
        return df_out
        
    # Limpieza de seguridad básica
    df_out[source_col] = pd.to_numeric(df_out[source_col], errors='coerce').fillna(0)
    
    if option == "binary":
        df_out['IsFranchise'] = df_out[source_col].apply(lambda x: 0 if x <= 1 else 1)
        df_out = df_out.drop(columns=[source_col])
    elif option == "raw":
        pass # Se deja tal cual
    else:
        raise ValueError(f"Opción no reconocida para FranchiseCode: {option}")
        
    return df_out
