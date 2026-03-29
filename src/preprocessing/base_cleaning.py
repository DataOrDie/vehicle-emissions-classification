import pandas as pd

def clean_base_columns(df):
    """
    Limpia columnas generales:
    - Elimina columnas inútiles
    - Limpia City
    - Limpia Bank
    - Crea IsLocalBank
    - Elimina State y BankState
    """

    # 1. Eliminar columnas que no aportan
    df = df.drop(['id', 'LoanNr_ChkDgt', 'Name'], axis=1)

    # 2. Limpieza de City
    df['City'] = df['City'].fillna('UNKNOWN_CITY')
    df['City'] = df['City'].str.upper().str.strip()

    # 3. Limpieza de Bank
    df['Bank'] = df['Bank'].fillna('UNKNOWN_BANK')

    # 4. Crear IsLocalBank
    df['IsLocalBank'] = (df['BankState'] == 'IL').astype(int)

    # 5. Eliminar columnas redundantes
    df = df.drop(['State', 'BankState'], axis=1)

    return df
