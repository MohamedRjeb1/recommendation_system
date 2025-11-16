import pandas as pd
import numpy as np

# Charger le dataset
animes = pd.read_csv(r"C:\Users\moham\OneDrive\Desktop\Recommendation_system\anime_with_views.csv")

# Remplacer "Unknown" par un entier aléatoire entre 20 et 100
for col in ["episodes"]:
    animes[col] = animes[col].replace("Unknown", np.random.randint(20, 101))


animes.to_csv(r"C:\Users\moham\OneDrive\Desktop\Recommendation_system\anime_with_views.csv", index=False)



