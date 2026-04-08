import pandas as pd
import numpy as np
import logging
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings

warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_clustering(data_path='Datasets/mall_customers.csv', k_min=3, k_max=8):
    """
    Executes the Unsupervised Clustering pipeline.
    """
    logging.info("Starting Unsupervised Clustering Pipeline")
    results = {}
    
    try:
        logging.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        logging.info("Training KMeans on Annual_Income and Spending_Score (for metrics display)")
        # try using a for loop for 2 features
        k = range(k_min, k_max + 1)
        K_vals = []
        WCSS = []
        ss = []
        try:
            for i in k:
                # KMeans parameters set explicitly to avoid memory leak warnings on Windows
                kmodel = KMeans(n_clusters=i, n_init=10, random_state=42)
                kmodel.fit(df[['Annual_Income','Spending_Score']])
                wcss_score = kmodel.inertia_
                sil_score = silhouette_score(df[['Annual_Income','Spending_Score']], kmodel.labels_)
                
                WCSS.append(wcss_score)
                K_vals.append(i)
                ss.append(sil_score)
                
            results['cluster_stats_2d'] = pd.DataFrame({'cluster': K_vals, 'WCSS_Score': WCSS, 'Silhouette_Score': ss})
        except Exception as k_err:
             logging.warning(f"Error during KMeans 2D clustering: {k_err}")
             
        logging.info("Training KMeans on Age, Annual_Income, and Spending_Score")
        # try using a for loop for 3 features
        K_vals_3 = []
        ss_3 = []
        WCSS_3 = []
        try:
            for i in k:
                kmodel = KMeans(n_clusters=i, n_init=10, random_state=42)
                kmodel.fit(df[['Age','Annual_Income','Spending_Score']])
                wcss_score = kmodel.inertia_
                sil_score = silhouette_score(df[['Age','Annual_Income','Spending_Score']], kmodel.labels_)
                
                K_vals_3.append(i)
                ss_3.append(sil_score)
                WCSS_3.append(wcss_score)

            results['cluster_stats_3d'] = pd.DataFrame({'cluster': K_vals_3, 'WCSS_Score': WCSS_3, 'Silhouette_Score': ss_3})
        except Exception as k_err_3:
            logging.warning(f"Error during KMeans 3D clustering: {k_err_3}")
            
        logging.info("Clustering pipeline executed successfully.")
        return results

    except FileNotFoundError:
        logging.error(f"Dataset not found at {data_path}. Please check the file path.")
        raise
    except Exception as e:
        logging.error(f"An error occurred during pipeline execution: {str(e)}")
        raise
