import pandas as pd  # Construire et manipuler le tableau de données
import numpy as np  # Générer des valeurs numériques aléatoires

# Fixer la graine aléatoire pour recréer les mêmes données à chaque exécution.
# 42 ne signifie pas que Python commence par ce nombre : il fixe le point
# de départ du générateur aléatoire et garantit la reproductibilité.
np.random.seed(42)



n = 500  # Nombre de clients fictifs à générer

# Construire le tableau colonne par colonne.
# Chaque colonne est une caractéristique d'entrée du futur modèle.
data = pd.DataFrame(
    {
        "age": np.random.randint(18, 70, n),  # Âge entre 18 et 69 ans
        "salaire": np.random.randint(1200, 8000, n),  # Salaire mensuel simulé
        "abonnement": np.random.choice(
            ["Basique", "Standard", "Premium"], n
        ),  # Type d'offre
        "anciennete": np.random.randint(0, 15, n),  # Ancienneté en années
        "consommation": np.random.randint(
            0, 100, n
        ),  # Score d'utilisation du service, sur 100
    }
)

# Créer la cible à prédire : le churn.
# Règle simplifiée et volontairement arbitraire :
# un client peu actif (consommation < 30) et récent (ancienneté < 2 ans)
# est considéré comme présentant un risque de départ.
data["churn"] = (
    (data["consommation"] < 30) & (data["anciennete"] < 2)
).astype(int)  # Convertir les valeurs True/False en 1/0

# Enregistrer le tableau dans un fichier CSV.
# Attention : à ce stade, ce fichier n'est protégé par aucun système de version.
data.to_csv("churn_data.csv", index=False)

# Afficher un aperçu et vérifier le taux de churn.
print(data.head())
print(f"Taux de churn : {data['churn'].mean():.1%}")

