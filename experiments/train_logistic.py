import wandb
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. CONFIGURACIÓN INICIAL DE W&B
# ==========================================
print("Iniciando conexión con Weights & Biases...")
wandb.init(
    entity="team-data-or-die",                  # El espacio de tu equipo
    project="Geometric- Logistic Regression",   # Tu proyecto
    name="baseline_model",                      # Nombre de este intento
    config={
        "model_type": "Logistic Regression",
        "solver": "lbfgs",
        "max_iter": 1000,
        "random_state": 42
    }
)
config = wandb.config

# ==========================================
# 2. CARGA DE DATOS (AQUÍ TRABAJAREMOS AHORA)
# ==========================================
def load_data():
    # Aquí es donde pondremos tu train.csv limpio
    pass 

# ==========================================
# 3. ENTRENAMIENTO DEL MODELO
# ==========================================
def train_model(X_train, y_train):
    # Aquí irá el código que entrena el modelo
    pass

if __name__ == "__main__":
    print("¡Archivo configurado correctamente!")
    # wandb.finish() # Cierra la conexión cuando termine

