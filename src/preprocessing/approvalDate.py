import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def preprocess_approvaldate(df, option="A", source_col="ApprovalDate"):
    """
    Preprocesa la columna ApprovalDate.
    Opción A: Extrae Año y Mes, rellena nulos con la moda y normaliza (MinMax) entre 0 y 1.
    Opción B: Convierte la columna a formato datetime y la mantiene entera.
    """
    # Creamos una copia para no modificar el DataFrame original por accidente
    df_out = df.copy()
    
    # Si la columna ya no existe (por ejemplo, si la celda se corrió 2 veces), devolvemos el df como está
    if source_col not in df_out.columns:
        return df_out
        
    # Convertimos el texto a formato de fecha (datetime). 'coerce' pone NaT (nulo) si hay formatos raros
    fechas = pd.to_datetime(df_out[source_col], format='%d-%b-%y', errors='coerce')
    
    if option == "A":
        # 1. Extraemos el Año y el Mes en dos columnas temporales
        df_out['ApprovalYear'] = fechas.dt.year
        df_out['ApprovalMonth'] = fechas.dt.month
        
        # 2. Calculamos la moda (el valor más frecuente) para rellenar los datos faltantes
        moda_y = df_out['ApprovalYear'].mode()[0]
        moda_m = df_out['ApprovalMonth'].mode()[0]
        
        # Aplicamos la moda a los nulos y aseguramos que sean números enteros
        df_out['ApprovalYear'] = df_out['ApprovalYear'].fillna(moda_y).astype(int)
        df_out['ApprovalMonth'] = df_out['ApprovalMonth'].fillna(moda_m).astype(int)
        
        # 3. NORMALIZACIÓN: Escalamos los valores para que estén estrictamente entre 0 y 1
        scaler = MinMaxScaler()
        # Nombramos las columnas con el sufijo '_normalized' para seguir el estándar del equipo
        df_out['approvalyear_normalized'] = scaler.fit_transform(df_out[['ApprovalYear']])
        df_out['approvalmonth_normalized'] = scaler.fit_transform(df_out[['ApprovalMonth']])
        
        # 4. Limpieza: Borramos la fecha original y las temporales que no están normalizadas
        df_out = df_out.drop(columns=[source_col, 'ApprovalYear', 'ApprovalMonth'])
        
    elif option == "B":
        # Opción B: Simplemente guardamos la fecha limpia en formato datetime por si alguien la necesita entera
        df_out[source_col] = fechas
    else:
        # Protección por si alguien en el equipo escribe una opción incorrecta
        raise ValueError(f"Opción no válida para {source_col}: {option}")
        
    return df_out