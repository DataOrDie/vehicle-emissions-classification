import pandas as pd

def preprocess_urban_rural(df, option="onehot", source_col="UrbanRural"):
    """
    Preprocesa la columna UrbanRural.
    
    Opciones:
    - 'text': Mapea los números (0, 1, 2) a strings ('Undefined', 'Urban', 'Rural').
    - 'onehot': Mapea a texto y aplica One-Hot Encoding creando 3 columnas binarias.
    """
    df_out = df.copy()
    
    if source_col not in df_out.columns:
        return df_out
        
    # Limpieza básica
    df_out[source_col] = pd.to_numeric(df_out[source_col], errors='coerce').fillna(0)
    
    # Mapeo base a texto
    mapeo = {0: 'Undefined', 1: 'Urban', 2: 'Rural'}
    
    if option == "text":
        df_out[source_col] = df_out[source_col].map(mapeo)
    elif option == "onehot":
        df_out[source_col] = df_out[source_col].map(mapeo)
        # prefix='Zone' para crear Zone_Urban, Zone_Rural, Zone_Undefined
        df_out = pd.get_dummies(df_out, columns=[source_col], prefix='Zone')
        # Convertir booleanos a enteros (1/0)
        cols_zone = [c for c in df_out.columns if c.startswith('Zone_')]
        df_out[cols_zone] = df_out[cols_zone].astype(int)
    else:
        raise ValueError(f"Opción no reconocida para UrbanRural: {option}")
        
    return df_out
