from category_encoders import BinaryEncoder
from sklearn.compose import ColumnTransformer

def get_city_bank_encoder():
    """
    Devuelve un ColumnTransformer que aplica Binary Encoding
    a City y Bank.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('binary', BinaryEncoder(), ['City', 'Bank'])
        ],
        remainder='passthrough'
    )
    return preprocessor
