import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_approvalfy(df, option="A", source_col="ApprovalFY"):
    """
    Preprocesa la columna ApprovalFY (Año Fiscal).
    Opción A: Elimina la columna por completo (considerada redundante respecto a ApprovalDate).
    Opción B: Limpia el texto, convierte a número, rellena nulos con moda y normaliza (MinMax).
    Opción C: Limpia el texto, convierte a número y rellena nulos con moda (sin normalizar).
    """
    # Creamos una copia del DataFrame
    df_out = df.copy()
    
    # Si la columna ya fue borrada previamente, regresamos el df para evitar errores (KeyError)
    if source_col not in df_out.columns:
        return df_out
        
    if option == "A":
        # Opción A: Simplemente eliminamos la columna porque el equipo decidió no usarla
        df_out = df_out.drop(columns=[source_col])
        
    elif option == "B":
        # Opción B (Plan de respaldo): Limpiar y Normalizar en lugar de borrar
        
        # 1. Convertimos a número. 'coerce' convierte letras basura (ej. '1997A') en nulos (NaN)
        df_out[source_col] = pd.to_numeric(df_out[source_col], errors='coerce')
        
        # 2. Calculamos la moda y rellenamos los nulos
        mode_val = df_out[source_col].mode()[0]
        df_out[source_col] = df_out[source_col].fillna(mode_val).astype(int)
        
        # 3. NORMALIZACIÓN: Escalamos los años fiscales para que queden entre 0 y 1
        scaler = MinMaxScaler()
        # Creamos la columna nueva estandarizada
        df_out['approvalfy_normalized'] = scaler.fit_transform(df_out[[source_col]])
        
        # 4. Borramos la columna original (la que no está normalizada)
        df_out = df_out.drop(columns=[source_col])

    elif option == "C":
        # Opción C: Limpiar y conservar el año fiscal sin escalar

        # 1. Convertimos a número. 'coerce' convierte letras basura (ej. '1997A') en nulos (NaN)
        df_out[source_col] = pd.to_numeric(df_out[source_col], errors='coerce')

        # 2. Calculamos la moda y rellenamos los nulos
        mode_val = df_out[source_col].mode()[0]
        df_out[source_col] = df_out[source_col].fillna(mode_val).astype(int)
    else:
        # Protección contra opciones inválidas
        raise ValueError(f"Opción no válida para {source_col}: {option}. Usa 'A', 'B' o 'C'.")
        
    return df_out