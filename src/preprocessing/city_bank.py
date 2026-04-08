from __future__ import annotations

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


def _normalize_category_value(value):
    """Normalize category tokens so train/test mapping is consistent."""
    if value is None:
        return "MISSING"

    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "<na>"}:
        return "MISSING"
    return text


def configure_city_bank_frequency_buckets(
    df,
    config: dict | None = None,
    *,
    city_top_k: int = 120,
    bank_top_k: int = 80,
    city_min_count: int | None = None,
    bank_min_count: int | None = None,
    other_label: str = "OTHER",
    suffix: str = "_bucket",
    drop_original: bool = False,
):
    """Fit/apply high-frequency bucketing for City and Bank.

    This function supports two modes:
    - Fit mode: pass ``config=None`` to learn kept categories from ``df``.
    - Apply mode: pass a previously learned ``config`` to transform new data.

    Parameters
    ----------
    df : pd.DataFrame-like
        Input data containing City/Bank (case-insensitive names supported).
    config : dict | None
        Previously learned bucket config. If ``None``, config is learned.
    city_top_k : int
        Number of most frequent City categories to keep when
        ``city_min_count`` is not provided.
    bank_top_k : int
        Number of most frequent Bank categories to keep when
        ``bank_min_count`` is not provided.
    city_min_count : int | None
        Keep City categories with count >= this value. Overrides ``city_top_k``.
    bank_min_count : int | None
        Keep Bank categories with count >= this value. Overrides ``bank_top_k``.
    other_label : str
        Label assigned to infrequent/unseen categories.
    suffix : str
        Suffix for generated bucket columns.
    drop_original : bool
        If True, drop original City/Bank columns after creating bucket columns.

    Returns
    -------
    tuple
        (transformed_df, fitted_config)
    """
    if not hasattr(df, "columns"):
        raise TypeError("configure_city_bank_frequency_buckets expects a DataFrame-like object with columns.")

    col_map = {c.lower(): c for c in df.columns}
    required_lower = ["city", "bank"]
    missing = [name for name in required_lower if name not in col_map]
    if missing:
        raise KeyError(f"Missing required columns for bucketing: {missing}")

    city_col = col_map["city"]
    bank_col = col_map["bank"]

    result = df.copy()
    result[city_col] = result[city_col].map(_normalize_category_value)
    result[bank_col] = result[bank_col].map(_normalize_category_value)

    is_fit = config is None
    fitted_config = config.copy() if isinstance(config, dict) else {}

    def _fit_keep_list(series, *, top_k, min_count):
        counts = series.value_counts(dropna=False)
        if min_count is not None:
            return counts[counts >= int(min_count)].index.tolist()
        return counts.head(int(top_k)).index.tolist()

    if is_fit:
        city_keep = _fit_keep_list(result[city_col], top_k=city_top_k, min_count=city_min_count)
        bank_keep = _fit_keep_list(result[bank_col], top_k=bank_top_k, min_count=bank_min_count)

        fitted_config = {
            "columns": {
                "city": city_col,
                "bank": bank_col,
            },
            "strategy": {
                "city": "min_count" if city_min_count is not None else "top_k",
                "bank": "min_count" if bank_min_count is not None else "top_k",
            },
            "params": {
                "city_top_k": int(city_top_k),
                "bank_top_k": int(bank_top_k),
                "city_min_count": None if city_min_count is None else int(city_min_count),
                "bank_min_count": None if bank_min_count is None else int(bank_min_count),
            },
            "keep_values": {
                "city": city_keep,
                "bank": bank_keep,
            },
            "other_label": str(other_label),
            "suffix": str(suffix),
        }
    else:
        keep_values = fitted_config.get("keep_values", {})
        if "city" not in keep_values or "bank" not in keep_values:
            raise ValueError("config must include keep_values for both 'city' and 'bank'")

    keep_city = set(fitted_config["keep_values"]["city"])
    keep_bank = set(fitted_config["keep_values"]["bank"])
    out_other = str(fitted_config.get("other_label", other_label))
    out_suffix = str(fitted_config.get("suffix", suffix))

    city_bucket_col = f"{city_col}{out_suffix}"
    bank_bucket_col = f"{bank_col}{out_suffix}"

    result[city_bucket_col] = result[city_col].where(result[city_col].isin(keep_city), out_other)
    result[bank_bucket_col] = result[bank_col].where(result[bank_col].isin(keep_bank), out_other)

    if drop_original:
        result = result.drop(columns=[city_col, bank_col])

    return result, fitted_config
