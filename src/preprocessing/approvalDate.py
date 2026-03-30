import pandas as pd

def preprocess_approvaldate(df, option="A", source_col="ApprovalDate"):
    df_out = df.copy()
    if source_col not in df_out.columns:
        return df_out
        
    fechas = pd.to_datetime(df_out[source_col], format='%d-%b-%y', errors='coerce')
    
    if option == "A":
        df_out['ApprovalYear'] = fechas.dt.year
        df_out['ApprovalMonth'] = fechas.dt.month
        
        moda_y = df_out['ApprovalYear'].mode()[0]
        moda_m = df_out['ApprovalMonth'].mode()[0]
        df_out['ApprovalYear'] = df_out['ApprovalYear'].fillna(moda_y).astype(int)
        df_out['ApprovalMonth'] = df_out['ApprovalMonth'].fillna(moda_m).astype(int)
        
        df_out = df_out.drop(columns=[source_col])
    elif option == "B":
        df_out[source_col] = fechas
    else:
        raise ValueError(f"Opción no válida para {source_col}: {option}")
        
    return df_out