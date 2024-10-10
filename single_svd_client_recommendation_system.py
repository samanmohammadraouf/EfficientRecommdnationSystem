import pandas as pd
import numpy as np
import time
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

# Setup logging
logging.basicConfig(filename='single_client_svd_recommendation_system.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(file_path):
    return pd.read_csv(file_path)
    
def create_user_item_matrix_with_column_means(ratings_df):
    R_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
    R_df_filled = R_df.apply(lambda x: x.fillna(x.mean()), axis=0)
    return R_df_filled.fillna(0), R_df

def predict_ratings(U, sigma, Vt, R_df):
    predicted_ratings = np.dot(np.dot(U, sigma), Vt)
    return predicted_ratings

def evaluate_model(test_df, R_full_df, predicted_ratings):
    test_actual_ratings = test_df.set_index(['userId', 'movieId']).rating
    test_rating = []
    test_predicted_ratings = []
    for index, actual_rating in test_actual_ratings.items():
        try:
            predicted_rating = predicted_ratings[R_full_df.index.get_loc(index[0]), R_full_df.columns.get_loc(index[1])]
            test_predicted_ratings.append(predicted_rating)
            test_rating.append(actual_rating)
        except Exception as e:
            continue
    return mean_absolute_error(test_rating, test_predicted_ratings)

def apply_svd(R):
    start_time = time.time()

    U, sigma, Vt = np.linalg.svd(R, full_matrices=True)
    sigma_matrix = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(sigma_matrix, sigma)

    end_time = time.time()

    svd_calculation_time = end_time - start_time
    logging.info(f"SVD calculation time: {svd_calculation_time} seconds")

    return U, sigma_matrix, Vt

def simple_main():
    overall_start_time = time.time()

    ratings_file_path = 'ratings_percent.csv'
    ratings_df = load_dataset(ratings_file_path)

    train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    R_train_df_filled, _ = create_user_item_matrix_with_column_means(train_df)

    U_filled, sigma_filled, Vt_filled = apply_svd(R_train_df_filled)

    full_df = pd.concat([train_df, test_df])
    R_full_df_filled, _ = create_user_item_matrix_with_column_means(full_df)

    predicted_ratings_filled = predict_ratings(U_filled, sigma_filled, Vt_filled, R_full_df_filled)

    mae_test_filled = evaluate_model(test_df, R_full_df_filled, predicted_ratings_filled)

    print(f"Mean Absolute Error on Test Set: {mae_test_filled}")
    logging.info(f"Mean Absolute Error on Test Set: {mae_test_filled}")

    overall_end_time = time.time()
    total_runtime = overall_end_time - overall_start_time
    logging.info(f"Total runtime: {total_runtime} seconds")

if __name__ == "__main__":
    simple_main()
