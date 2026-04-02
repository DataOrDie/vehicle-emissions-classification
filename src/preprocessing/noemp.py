import numpy as np
import pandas as pd

def preprocess_noemp(df, option="raw", source_col="NoEmp"):
    """
    Opciones:
        - raw: dejar NoEmp tal cual caso Bagging
        - log: aplicar log1p
        - binning: categorizar tamaño empresa
        - C: aplicar log1p + estandarizacion z-score
    """

    if option == "log":
        df[source_col + "_Log"] = np.log1p(df[source_col])
        df = df.drop([source_col], axis=1, errors='ignore')

    elif option in {"C", "c", "log_standardize", "log_std"}:
        noemp_num = pd.to_numeric(df[source_col], errors="coerce")
        noemp_log = np.log1p(noemp_num.clip(lower=0))
        mean_val = noemp_log.mean(skipna=True)
        std_val = noemp_log.std(skipna=True, ddof=0)

        if pd.isna(std_val) or std_val == 0:
            df[source_col + "_LogStd"] = pd.Series(pd.NA, index=df.index, dtype="Float64")
        else:
            df[source_col + "_LogStd"] = ((noemp_log - mean_val) / std_val).astype("Float64")

        df = df.drop([source_col], axis=1, errors='ignore')

    elif option == "binning":
        def categorize(n):
            if n <= 5:
                return 1
            elif n <= 15:
                return 2
            else:
                return 3
        df[source_col + "_Bin"] = df[source_col].apply(categorize)
        df = df.drop([source_col], axis=1, errors='ignore')

    elif option == "raw":
        pass

    else:
        raise ValueError(f"Opción no válida: {option}")

    return df
