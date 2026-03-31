from category_encoders import BinaryEncoder

def get_city_bank_encoder(df):
    """
    Recibe un DataFrame, aplica Binary Encoding a las columnas City/Bank
    (tambien soporta city/bank), y devuelve un nuevo DataFrame transformado.
    """
    if not hasattr(df, "columns"):
        raise TypeError("get_city_bank_encoder expects a DataFrame-like object with columns.")

    col_map = {c.lower(): c for c in df.columns}
    required_lower = ["city", "bank"]
    missing = [name for name in required_lower if name not in col_map]
    if missing:
        raise KeyError(f"Missing required columns for encoding: {missing}")

    cols_to_encode = [col_map["city"], col_map["bank"]]
    encoder = BinaryEncoder(cols=cols_to_encode)
    transformed_df = encoder.fit_transform(df)
    return transformed_df
