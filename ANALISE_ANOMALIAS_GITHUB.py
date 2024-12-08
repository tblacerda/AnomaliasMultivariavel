import shap
import time
import pandas as pd
import numpy as np
import sqlite3
from sqlite3 import Error
import seaborn as sns
from datetime import timedelta, date
from tqdm import tqdm
import warnings
import glob
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
import requests
from io import StringIO
from concurrent.futures import ProcessPoolExecutor
import logging
import os

__DIAS_MAX__ = 45   # Populacao maxima considerada
__DIAS_MIN__ = 30   # Populacao minima considerada
__DIAS_AMOSTRA__ = 3 # ao menos __DIAS_AMOSTRA__ / 2. Amostra de dias recentes a ser avaliada 
__AVAIL__ = 0.8  # Considera apenas sites com 100% de amostras
__RODAR_PARALELO__ = True
_ANOMALY_FILTER_ = -0.05

# Suppress warnings
warnings.filterwarnings("ignore")

class AnomalyDetection:
    # https://medium.com/@limyenwee_19946/unsupervised-outlier-detection-with-isolation-forest-eab398c593b2
    def __init__(self, N=__DIAS_AMOSTRA__):
        """
        Initializes the MULTIVARIADO class.

        Parameters:
        N (int): Number of recent days for comparison. Default is __DIAS_AMOSTRA__.

        Attributes:
        N (int): Stores the number of recent days for comparison.
        model (IsolationForest): Model to detect anomalies.
        transformer (PowerTransformer): Transformer to apply Yeo-Johnson transformation and standardization.
        """
        self.N = N  # Number of recent days for comparison
        self.model = IsolationForest(contamination='auto', random_state= 1, n_estimators= 250)  # Model to detect anomalies
        self.transformer = PowerTransformer(method='yeo-johnson', standardize=True)
    
    def apply_transformations(self, df, columns):
        """
        Apply Box-Cox or Yeo-Johnson transformation to the 'jitter', 'ploss', and 'latency' columns for each unique site in the DataFrame.
        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be transformed. 
                       It must have columns 'Site', 'jitter', 'ploss', and 'latency'.
        Returns:
        pandas.DataFrame: The DataFrame with transformed 'jitter', 'ploss', and 'latency' columns for each site.
        Notes:
        - The transformation is applied separately for each unique site in the 'Site' column.
        - A small value (0.001) is added to the data to ensure all values are positive before applying the transformation.
        - If the Box-Cox transformation fails for a site, a warning is issued, and the original data for that site is retained.
        """
        # Apply Box-Cox OR Yeo-johnson transformation for each site
        sites = df['Site'].unique()
        for site in tqdm(sites, desc="Applying transformations"):
            site_data = df[df['Site'] == site][columns]
            
            # Add a small value to ensure data is positive
            site_data_adj = site_data
            try:
                # Apply Box-Cox transformation
                df.loc[df['Site'] == site, columns] = self.transformer.fit_transform(site_data_adj)
            except Exception as e:
                warnings.warn(f"Box-Cox transformation failed for site {site}: {e}")
                # Skip the transformation for this site and keep original data
                df.loc[df['Site'] == site, columns] = site_data
        return df
    
    def detect_anomalies_for_site(self, site_data, columns):
        """
        Detects anomalies for a given site based on recent and historical data.
        Parameters:
        site_data (DataFrame): A pandas DataFrame containing the site's data with columns 'jitter', 'ploss', 'latency', and 'Site'.
        Returns:
        float: The average anomaly score for the recent data. Returns 0 if there is insufficient data.
        Raises:
        UserWarning: If there is insufficient data for the site or insufficient training data.
        """
        # Check if the data is sufficient
        if len(site_data) <= self.N:
            warnings.warn(f"Insufficient data for site: {site_data['Site'].iloc[0]}")
            return 0
        
        recent_days = site_data.tail(self.N)
        historical_data = site_data.iloc[:-self.N]

        # Train model on historical data
        X_train = historical_data[columns].values
        if len(X_train) < 1:
            warnings.warn(f"Insufficient training data for site: {site_data['Site'].iloc[0]}")
            return 0

        # Fit model and calculate anomaly scores for the recent data
        self.model.fit(X_train)
        X_recent = recent_days[columns].values
        time_weights = np.linspace(1.0, 2.0, len(X_recent)) 
        anomaly_scores = self.model.decision_function(X_recent)  # Get anomaly scores for recent data # tiago
        anomaly_scores *= time_weights
        try:
            explanations = self.explain_anomalies(site_data, columns)
        except:
            explanations = {}
        #print(explanations)
        # Create a result array for the entire data, with zeros for historical data
        #result = np.zeros(len(site_data))
        #result[-self.N:] = anomaly_scores  # Assign scores to the last N entries (recent data)
        
        # Return average anomaly score for the recent data
        avg_score = anomaly_scores.mean()
        return avg_score, explanations
    
    def explain_anomalies(self, df, columns):
        """
        Explain anomalies using SHAP values.
        This method uses SHAP (SHapley Additive exPlanations) to explain the anomaly scores
        and identifies the top 3 variables that contribute to the anomalies.
        Args:
            df (pd.DataFrame): The input DataFrame containing site data.
            columns (list): A list of column names to be used for anomaly detection.
        Returns:
            dict: A dictionary where keys are site identifiers and values are lists of the top 3 variables
                    that contribute to the anomalies for each site.
        """
        explanations = {}
        sites = df['Site'].unique()
        for site in sites:
            site_data = df[df['Site'] == site]
            if len(site_data) <= self.N:
                continue

            recent_days = site_data.tail(self.N)
            historical_data = site_data.iloc[:-self.N]

            X_train = historical_data[columns].values
            X_recent = recent_days[columns].values

            self.model.fit(X_train)
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(X_recent)

            # Calculate mean absolute SHAP values for each feature
            mean_shap_values = np.abs(shap_values).mean(axis=0)
            top_features = np.argsort(mean_shap_values)[-3:][::-1]
            top_feature_names = [columns[i] for i in top_features]

            explanations[site] = top_feature_names
            #print(f"Site {site}: Top 3 features contributing to anomalies: {top_feature_names}")

        return explanations

    def detect_anomalies(self, df, columns):
        """
        Detect anomalies in the given DataFrame.
        This method processes each unique site in the DataFrame to detect anomalies
        and assigns an anomaly score to each site. It also ranks the sites based on
        their average anomaly scores.
        Args:
            df (pd.DataFrame): The input DataFrame containing site data. It must have
                               a column named 'Site' which contains site identifiers.
        Returns:
            tuple: A tuple containing:
                - pd.DataFrame: The DataFrame with an additional column 'Anomaly' 
                                indicating the anomaly score for each site.
                - list: A list of tuples where each tuple contains a site identifier 
                        and its average anomaly score, sorted by the anomaly score 
                        in ascending order (lower scores are worse).
        """
        # Process each site to detect anomalies
        site_ranking = []
        sites = df['Site'].unique()
        for site in tqdm(sites, desc="Detecting anomalies"):
            site_data = df[df['Site'] == site]
            try:
                avg_score, explanation = self.detect_anomalies_for_site(site_data, columns)
                df.loc[df['Site'] == site, 'Anomaly'] = avg_score
                concatenated_explanation = ''
            except:
                print(site)
            try:
                for key, value in explanation.items():
                    concatenated_explanation += ', '.join(value)
            except:
                print(site)
            df.loc[df['Site'] == site, 'Explanation'] = str(concatenated_explanation)  
            # Append site and its average score for ranking
            site_ranking.append((site, avg_score))
        
        # Sort the ranking by the average anomaly score (lower scores are worse)
        site_ranking = sorted(site_ranking, key=lambda x: x[1])
        return df, site_ranking
    
    def run_daily_analysis(self, df, columns):
        """
        Perform daily analysis on the provided DataFrame.
        This method sorts the DataFrame by 'Dia', applies necessary transformations,
        detects anomalies, and ranks sites based on their anomaly scores.
        Parameters:
        df (pandas.DataFrame): The input DataFrame containing the data to be analyzed.
        Returns:
        pandas.DataFrame: A DataFrame containing the median anomaly scores for each site,
                          with the most recent 'Dia' value.
        """
        # Sort the DataFrame by 'Dia'
        df = df.sort_values(by='Dia')

        # Apply transformations
        df_transformed = self.apply_transformations(df,columns)
        
        # Detect anomalies and get site rankings
        #    anomalies, site_ranking = self.detect_anomalies(df_transformed)
        anomalies, _ = self.detect_anomalies(df_transformed, columns)
       
        # Filter anomalies to include only those <= -0.05
        #anomalies = anomalies[anomalies['Anomaly'] <= _ANOMALY_FILTER_]

        # Rank sites based on their anomaly scores (lowest score gets rank 1)
        anomalies = anomalies.sort_values(by='Anomaly')
        #anomalies['Rank'] = anomalies.groupby('Dia').cumcount() + 1

        # Add the rank column to the anomalies DataFrame

       # anomalies = anomalies[['Dia', 'Site', 'Anomaly', 'Rank']]
        anomalies = anomalies[['Dia', 'Site', 'Anomaly', 'Explanation']]
        Explanations = anomalies[['Site', 'Explanation']]
        Explanations.drop_duplicates(inplace=True)

        anomalies = anomalies[['Dia', 'Site', 'Anomaly']]
        anomalies = anomalies.groupby('Site').tail(__DIAS_AMOSTRA__)
        anomalies = anomalies.groupby('Site').median().reset_index()
        anomalies['Dia'] = df_transformed['Dia'].max()
        anomalies = anomalies.merge(Explanations, on='Site', how='left')

        return anomalies


def process_chunk(chunk, columns):
    """
    Processes a given chunk of data to detect anomalies.
    This function logs the start and end of the processing. It uses the 
    AnomalyDetection class to perform daily analysis on the provided chunk 
    of data and returns any detected anomalies.
    Args:
        chunk (DataFrame): A chunk of data to be analyzed for anomalies.
    Returns:
        DataFrame: A DataFrame containing the detected anomalies.
    """

    logging.info('Processing a chunk')
    anomaly_detector = AnomalyDetection(N=__DIAS_AMOSTRA__)
    anomalies = anomaly_detector.run_daily_analysis(chunk, columns)
    logging.info('Finished processing a chunk')
    return anomalies

def split_data_into_chunks(dfConsolidado, num_chunks):
    """
    Split the DataFrame into chunks based on unique 'Site' values.

    Parameters:
    dfConsolidado (pd.DataFrame): The input DataFrame to be split.
    num_chunks (int): The number of chunks to split the DataFrame into.

    Returns:
    list: A list of DataFrame chunks.
    """
    # Step 1: Get unique 'Site' values
    unique_sites = dfConsolidado['Site'].unique()
    
    # Step 2: Split unique 'Site' values into chunks
    site_chunks = np.array_split(unique_sites, num_chunks)
    
    # Step 3: Create DataFrame chunks based on 'Site' chunks
    chunks = [dfConsolidado[dfConsolidado['Site'].isin(sites)] for sites in site_chunks]
    
    return chunks


# Example usage
def plot_anomaly_histogram(anomalies):
    """
    Plot a histogram of anomaly scores with a vertical line separating the lower 2.5% percentile.
    
    Parameters:
    anomalies (pd.DataFrame): DataFrame containing anomaly scores and site information.
    """
    # Plot histogram with a vertical line separating the lower 2.5% percentile
    plt.figure(figsize=(10, 6))
    anomalies['Anomaly'].hist(bins=100, alpha=0.7, color='blue', edgecolor='black')

    # Calculate the 2.5% percentile
    percentile_2_5 = np.percentile(anomalies['Anomaly'], 2.5)
    # Add a vertical line at the 2.5% percentile
    plt.axvline(percentile_2_5, color='red', linestyle='dashed', linewidth=2, label=f'Percentil 2.5%: {percentile_2_5:.2f}')

    # Add labels and title
    plt.xlabel('Pontuação de Anomalia')
    plt.ylabel('Frequência')
    plt.title('Histograma das Pontuações de Anomalia com Linha do Percentil 2.5%')
    
    # Add text annotation for the total number of analyzed sites
    total_sites = anomalies['Site'].nunique()
    plt.text(0.95, 0.7, f'Total de Sites Analisados: {total_sites}', 
                horizontalalignment='right', 
                verticalalignment='top', 
                transform=plt.gca().transAxes, 
                fontsize=12, 
                bbox=dict(facecolor='white', alpha=0.5))
    
    # Add text annotation for the number of sites below the 2.5% percentile
    sites_below_percentile = anomalies[anomalies['Anomaly'] <= percentile_2_5]['Site'].nunique()
    plt.text(0.32, 0.7, f'Sites Abaixo do Percentil 2.5%: {sites_below_percentile}', 
                horizontalalignment='right', 
                verticalalignment='top', 
                transform=plt.gca().transAxes, 
                fontsize=10, 
                bbox=dict(facecolor='white', alpha=0.5))
    
    plt.legend()
    # Show plot
    plt.show()



def main():

    dfConsolidado = pd.read_excel('Entrada.xlsx')
    dfConsolidado = dfConsolidado[dfConsolidado['disp_acesso'] >= (__AVAIL__ * 100)] 
    dfConsolidado = dfConsolidado[dfConsolidado['disp_twamp'] >= (__AVAIL__ * 100)]       
    # from dfConsolidado, create a list of the columns right from 'eNodeB'
    numeric_columns= dfConsolidado.columns[dfConsolidado.columns.get_loc('Site') + 1:].tolist()
    
    print(numeric_columns)

    if __RODAR_PARALELO__:
            # Split the DataFrame into chunks
        num_chunks = 4  # Adjust based on your CPU cores
        chunks = split_data_into_chunks(dfConsolidado, num_chunks)
        with ProcessPoolExecutor() as executor:
            print("Executing in parallel...")
            results = list(executor.map(process_chunk, chunks, numeric_columns))
            anomalies = pd.concat(results, ignore_index=True)
    else:
        anomaly_detector = AnomalyDetection(N = __DIAS_AMOSTRA__)
        anomalies = anomaly_detector.run_daily_analysis(dfConsolidado, numeric_columns)
        anomalies.dropna(subset=['Anomaly'], inplace= True)
        anomalies.to_excel('RESULTADO.xlsx', index=False)

        # Call the new method to plot the histogram
        plot_anomaly_histogram(anomalies)

if __name__ == '__main__':
    main()



