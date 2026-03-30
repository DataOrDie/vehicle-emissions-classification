import pandas as pd

def clean_base_columns(df):
    """
    Limpia columnas generales:
    - Elimina columnas inútiles
    - Limpieza robusta de City
    - Limpia Bank
    - Crea IsLocalBank
    - Elimina State y BankState
    """

    # 1. Eliminar columnas que no aportan
    df = df.drop(['id', 'LoanNr_ChkDgt', 'Name'], axis=1)

    # 2. Limpieza robusta de City
    df['City'] = df['City'].fillna('UNKNOWN_CITY')

    # 2.1 Normalizar formato
    df['City'] = df['City'].str.upper().str.strip()

    # 2.2 Eliminar paréntesis abiertos o cerrados y todo lo que sigue
    df['City'] = df['City'].str.replace(r'\(.*$', '', regex=True).str.strip()

    # 2.3 Eliminar puntos sueltos o múltiples
    df['City'] = df['City'].str.replace(r'\.+', ' ', regex=True).str.strip()

    # 2.4 Eliminar comas
    df['City'] = df['City'].str.replace(',', '', regex=False).str.strip()

    # 2.5 Eliminar números (códigos postales)
    df['City'] = df['City'].str.replace(r'\d+', '', regex=True).str.strip()

    # 2.6 Corregir caracteres basura como @ y `
    df['City'] = df['City'].str.replace('@', "'", regex=False)
    df['City'] = df['City'].str.replace('`', '', regex=False)

    # 2.7 Normalizar espacios múltiples
    df['City'] = df['City'].str.replace(r'\s+', ' ', regex=True).str.strip()

    # 3. Limpieza de Bank
    df['Bank'] = df['Bank'].fillna('UNKNOWN_BANK')

    # 4. Crear IsLocalBank
    df['IsLocalBank'] = (df['BankState'] == 'IL').astype(int)

    # 5. Eliminar columnas redundantes
    df = df.drop(['State', 'BankState'], axis=1)

    return df
