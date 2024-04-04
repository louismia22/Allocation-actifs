import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import scipy.stats as stats

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




def perform_regression(df_data):
    """
    This function takes a dataframe with price columns and the CAC Index Adjusted column,
    performs a regression to predict the log of the CAC Index Adjusted using the logs of the prices,
    and returns the calculated weights, the initial intercept, and the residuals over time.
    """
    # Take all columns from the second to the penultimate
    independent_vars = df_data.iloc[:, 1:-1]
    # Take the last column as the dependent variable
    dependent_var = df_data.iloc[:, -1]

    # Apply logarithm to the selected data, ensuring no zero or negative values
    independent_vars = independent_vars.replace([np.inf, -np.inf, 0], np.nan).dropna()
    dependent_var = dependent_var.replace([np.inf, -np.inf, 0], np.nan).dropna()

    # Apply logarithm to the selected data
    log_independent_vars = np.log(independent_vars)
    log_dependent_var = np.log(dependent_var)

    # Add a constant term for the intercept
    log_independent_vars_with_constant = sm.add_constant(log_independent_vars)

    # Perform the regression
    model = sm.OLS(log_dependent_var, log_independent_vars_with_constant).fit()

    # Calculate the residuals
    residuals = model.resid

    # Check if 'const' is in the model parameters
    if 'const' in model.params:
        intercept = model.params['const']
        # Return the intercept, model parameters (excluding the intercept), and residuals
        return intercept, model.params.drop('const'), residuals
    else:
        # If 'const' is not present, return a message and the rest of the parameters
        return "No intercept found", model.params, residuals
    



def rebalance_and_evaluate(df_plus, df_minus, start_date, costs):
    # Convertir start_date en objet datetime si nécessaire
    current_date = pd.to_datetime(start_date)+ pd.DateOffset(days=10)

    end_date = df_plus.iloc[-1].name # Supposons que l'index du df est une série de dates
    adf_critic_value = []


    # Initialiser la liste pour stocker les résultat
    prices_list = []
    adf_stats_list = []
    portfolio = pd.DataFrame()
    replication_max = pd.DataFrame()
    replication_min = pd.DataFrame()
    params_plus_old =perform_regression(df_plus)[2]
    params_minus_old =  perform_regression(df_minus)[2]
    
    while current_date < end_date:
        # Sélectionner la tranche de données jusqu'à la date courante
        data_slice_plus = df_plus.loc[start_date:current_date]
        data_slice_minus = df_minus.loc[start_date:current_date]
        

        intercept_plus, params_plus, residuals_plus = perform_regression(data_slice_plus)
        intercept_minus, params_minus, residuals_minus = perform_regression(data_slice_minus)
        #print(np.sum(abs((params_plus-params_minus) - (params_plus_old-params_minus_old))*data_slice_plus.iloc[-1,1:-1]))
        #print("ok")

        costs = 0.002*np.sum(abs((params_plus-params_minus) - (params_plus_old-params_minus_old))*data_slice_plus.iloc[-1,1:-1])

 

        params_plus_old = params_plus
        params_minus_old = params_minus
        
        

        
        # Effectuer le test ADF sur les résidus
        adf_result = adfuller(residuals_plus)
        adf_stats_list.append(adf_result[0])
        
        # Vérifier les valeurs critiques pour la cointégration
        #if adf_result[0] > adf_result[4]['1%']:  # Si la stat est plus grande que la valeur critique à 1%
            # Ne pas procéder au rebalancement si la cointégration n'est pas présente
           
           # break
        data_plus = df_plus.loc[current_date-pd.DateOffset(days=10):current_date]
        data_moins = df_minus.loc[current_date-pd.DateOffset(days=10):current_date]
       
        # Calculer le prix du portfolio après soustraction des coûts
        portfolio_price_plus = np.exp(np.sum(params_plus * np.log(data_plus.iloc[:, 1:-1]),axis=1)) #le prix de notre portefeuille, il manque les dates uniquement
        portfolio_price_minus= np.exp(np.sum(params_minus * np.log(data_moins.iloc[:, 1:-1]),axis=1)) + costs #le prix de notre portefeuille, il manque les dates uniquement


        prices_list.append(portfolio_price_plus-portfolio_price_minus)
        portfolio = pd.concat([portfolio, portfolio_price_plus-portfolio_price_minus])
        replication_max = pd.concat([replication_max, portfolio_price_plus])
        replication_min = pd.concat([replication_min,portfolio_price_minus ])

     
        current_date += pd.DateOffset(days=10) #on modifie la date.. 
        adf_critic_value.append(adf_result[4]['1%'])
    
  
    return  adf_critic_value,adf_stats_list,replication_max,replication_min


#calcul des métriques importantes.
def calculate_annual_global_metrics(replication_max, replication_min):
    """
    Calculer la variance annuelle du rendement global, le rendement global annuel,
    ainsi que la skewness et la kurtosis pour chaque année.

    :param daily_returns_diff: Une série pandas de rendements quotidiens avec des dates en index.
    :return: Un DataFrame avec la variance annuelle du rendement global, le rendement global annuel,
             la skewness et la kurtosis pour chaque année.
    """
    daily_returns_max = replication_max.pct_change().dropna()
    daily_returns_min = replication_min.pct_change().dropna()

    # Calculer la différence des rendements quotidiens : ce qui nous intéresse 
    daily_returns_diff = daily_returns_max - daily_returns_min
    # Grouper les données par année
    grouped = daily_returns_diff.groupby(daily_returns_diff.index.year)
    
    # Initialiser un DataFrame pour stocker les résultats
    annual_metrics = pd.DataFrame()
    
    # Calculer le rendement global annuel et la variance de ce rendement pour chaque groupe (année)
    annual_metrics['Annual Global Return'] = grouped.apply(lambda x: (1 + x).prod() - 1)
    annual_metrics['Daily mean Return'] = grouped.mean()
    annual_metrics['Daily Variance of Global Return'] = grouped.var() #on regarde la variance du rendement journalier 

    # Calculer la skewness et la kurtosis pour chaque année
    #annual_metrics['Annual Skewness'] = grouped.skew()
   # annual_metrics['Annual Kurtosis'] = grouped.kurtosis()
    
    return annual_metrics