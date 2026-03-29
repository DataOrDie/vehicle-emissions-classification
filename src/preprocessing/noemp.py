import numpy as np

def preprocess_noemp(df, option="raw", source_col="NoEmp"):
    """
    Opciones:
        - raw: dejar NoEmp tal cual caso Bagging
        - log: aplicar log1p
        - binning: categorizar tamaño empresa
    """

    if option == "log":
        df[source_col + "_Log"] = np.log1p(df[source_col])

    elif option == "binning":
        def categorize(n):
            if n <= 5:
                return 1
            elif n <= 15:
                return 2
            else:
                return 3
        df[source_col + "_Bin"] = df[source_col].apply(categorize)

    elif option == "raw":
        pass

    else:
        raise ValueError(f"Opción no válida: {option}")

    return df
