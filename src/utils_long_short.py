import pandas as pd
import numpy as np

def calculer_rendement_iteratif(df_data, date_debut, rendement_annuel):
    """
    Calcule et ajoute un rendement quotidien de manière itérative à l'index CAC à partir d'une date de départ.
    
    :param df_data: DataFrame contenant les indices CAC avec des dates comme index.
    :param date_debut: La date de début pour le calcul du rendement (format 'YYYY-MM-DD').
    :param rendement_annuel: Le rendement annuel souhaité en pourcentage.
    :return: Un DataFrame avec les indices CAC ajustés quotidiennement par le rendement.
    """
    # Calculer le rendement quotidien équivalent
    rendement_quotidien = rendement_annuel / 100 / 365
    
    # Filtrer le DataFrame pour les dates après la date de début (incluse)
    df_subset = df_data.loc[date_debut:].copy()
    
    # Initialiser l'index ajusté avec la première valeur de l'index CAC
    index_ajuste = [df_subset.iloc[0]['CAC Index']]
    
    # Appliquer le rendement itérativement
    for i in range(1, len(df_subset)):
        # Ajouter le rendement quotidien au dernier index ajusté
        v_t_plus_1 = df_subset.iloc[i]['CAC Index']
        v_t = df_subset.iloc[i-1]['CAC Index']

        valeur_ajustee = index_ajuste[-1]*(1+(v_t_plus_1-v_t)/v_t +rendement_quotidien)
        index_ajuste.append(valeur_ajustee)
    
    # Assigner les valeurs ajustées au DataFrame
    df_subset['CAC Index Ajusté'] = index_ajuste
    
    return df_subset

