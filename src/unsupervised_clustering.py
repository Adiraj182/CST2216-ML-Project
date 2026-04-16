import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
from logger import get_logger

warnings.filterwarnings("ignore")

logger = get_logger(__name__)

def run_clustering(data_path='Datasets/mall_customers.csv', k_min=3, k_max=8):
    """
    Executes the Unsupervised Clustering pipeline.
    Returns metrics AND the best trained KMeans models so app.py can predict new customers.
    """
    logger.info("Starting Unsupervised Clustering Pipeline")
    results = {}
    
    try:
        logger.info(f"Loading dataset from {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info("Training KMeans on Annual_Income and Spending_Score (2D)")
        k = range(k_min, k_max + 1)
        K_vals, WCSS, ss, models_2d = [], [], [], []

        try:
            for i in k:
                kmodel = KMeans(n_clusters=i, n_init=10, random_state=42)
                kmodel.fit(df[['Annual_Income', 'Spending_Score']])
                wcss_score = kmodel.inertia_
                sil_score = silhouette_score(df[['Annual_Income', 'Spending_Score']], kmodel.labels_)
                WCSS.append(wcss_score)
                K_vals.append(i)
                ss.append(sil_score)
                models_2d.append(kmodel)
                
            results['cluster_stats_2d'] = pd.DataFrame({
                'cluster': K_vals,
                'WCSS_Score': WCSS,
                'Silhouette_Score': ss
            })
            # Return the best model (highest silhouette score) for live prediction
            best_idx_2d = int(np.argmax(ss))
            results['best_k_2d'] = K_vals[best_idx_2d]
            results['model_2d'] = models_2d[best_idx_2d]
            
        except Exception as k_err:
            logger.warning(f"Error during KMeans 2D clustering: {k_err}")

        logger.info("Training KMeans on Age, Annual_Income, and Spending_Score (3D)")
        K_vals_3, ss_3, WCSS_3, models_3d = [], [], [], []

        try:
            for i in k:
                kmodel = KMeans(n_clusters=i, n_init=10, random_state=42)
                kmodel.fit(df[['Age', 'Annual_Income', 'Spending_Score']])
                wcss_score = kmodel.inertia_
                sil_score = silhouette_score(df[['Age', 'Annual_Income', 'Spending_Score']], kmodel.labels_)
                K_vals_3.append(i)
                ss_3.append(sil_score)
                WCSS_3.append(wcss_score)
                models_3d.append(kmodel)

            results['cluster_stats_3d'] = pd.DataFrame({
                'cluster': K_vals_3,
                'WCSS_Score': WCSS_3,
                'Silhouette_Score': ss_3
            })
            # Return the best 3D model for live prediction
            best_idx_3d = int(np.argmax(ss_3))
            results['best_k_3d'] = K_vals_3[best_idx_3d]
            results['model_3d'] = models_3d[best_idx_3d]

        except Exception as k_err_3:
            logger.warning(f"Error during KMeans 3D clustering: {k_err_3}")
            
        logger.info("Clustering pipeline executed successfully.")
        return results

    except FileNotFoundError:
        logger.error(f"Dataset not found at {data_path}. Please check the file path.")
        raise
    except Exception as e:
        logger.error(f"An error occurred during pipeline execution: {str(e)}", exc_info=True)
        raise
