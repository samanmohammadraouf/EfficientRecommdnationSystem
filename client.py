import pandas as pd
import numpy as np
import time
import logging
import key_generator
# import server
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import socket
import pickle

logging.basicConfig(filename='recommendation_systems.log', level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s')

class Client():
    def __init__(self) -> None:
        self.K1 = None
        self.k2 = None
        self.epsilon = None

    def load_dataset(self,file_path):
        return pd.read_csv(file_path)
        
    def create_user_item_matrix_with_column_means(self,ratings_df):
        R_df = ratings_df.pivot(index='userId', columns='movieId', values='rating')
        R_df_filled = R_df.apply(lambda x: x.fillna(x.mean()), axis=0)
        return R_df_filled.fillna(0), R_df

    def predict_ratings(self,U, sigma, Vt, R_df):
        predicted_ratings = np.dot(np.dot(U, sigma), Vt)
        return predicted_ratings

    def evaluate_model(self,test_df, R_full_df, predicted_ratings):
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
    
     
    def blind(self,R):
        R_blinded = np.dot(np.dot(self.K1, self.epsilon * R), self.K2)
        return R_blinded

    def verify(slef, R_blind, U_prime, sigma_prime, Vt_prime):
        n = R_blind.shape[1]
        r = np.random.choice([0, 1], size=(n, 1))
        
        right_side = np.dot(U_prime, np.dot(sigma_prime, np.dot(Vt_prime, r)))
        left_side = np.dot(R_blind, r)
        return np.allclose(left_side, right_side)

    def recovery(self,U_filled_prime,sigma_filled_prime,Vt_filled_prime):
        U_filled = np.dot(self.K1.T, U_filled_prime)
        Vt_filled = np.dot(Vt_filled_prime, self.K2.T)
        sigma_filled = sigma_filled_prime / self.epsilon
        return U_filled,Vt_filled,sigma_filled

    def simple_main(self):
        import server
        start_time = time.time()
        logging.info("Starting Client-Server Normal SVD Recommendation System")
        
        ratings_file_path = 'ratings.csv'
        ratings_df = self.load_dataset(ratings_file_path)
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        R_train_df_filled, _ = self.create_user_item_matrix_with_column_means(train_df)
        
        server_svd_start = time.time()
        U_filled, sigma_filled, Vt_filled = server.apply_svd(R_train_df_filled)
        server_svd_end = time.time()
        logging.info(f"Server SVD calculation time (Normal): {server_svd_end - server_svd_start} seconds")
        
        full_df = pd.concat([train_df, test_df])
        R_full_df_filled, _ = self.create_user_item_matrix_with_column_means(full_df)
        predicted_ratings_filled = self.predict_ratings(U_filled, sigma_filled, Vt_filled, R_full_df_filled)
        mae_test_filled = self.evaluate_model(test_df, R_full_df_filled, predicted_ratings_filled)
        print(f"Mean Absolute Error on Test Set: {mae_test_filled}")
        logging.info(f"MAE (Normal): {mae_test_filled}")
        
        end_time = time.time()
        logging.info(f"Total runtime (Normal): {end_time - start_time} seconds")


    def main(self):
        import server
        start_time = time.time()
        logging.info("Starting Client-Server Private SVD Recommendation System")
        
        ratings_file_path = 'ratings.csv'
        ratings_df = self.load_dataset(ratings_file_path)
        train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
        R_train_df_filled, _ = self.create_user_item_matrix_with_column_means(train_df)
        
        keygen_start = time.time()
        if not self.K1 or not self.K2 or not self.epsilon: # Consider this important point which we do not need to calculate the keys each time because they are independent from our data .... 
            self.K1, self.K2, self.epsilon = key_generator.generate_keys(R_train_df_filled.shape[0], R_train_df_filled.shape[1])
        keygen_end = time.time()
        logging.info(f"Key generation time (Private): {keygen_end - keygen_start} seconds")
        
        R_train_df_filled = np.array(R_train_df_filled)

        # Blind ...
        blind_r_start = time.time()
        R_blinded = self.blind(R_train_df_filled)
        blind_r_end = time.time()
        logging.info(f"R blinding calculation time (Private): {blind_r_end - blind_r_start} seconds")
        
        # SVD ...
        server_svd_start = time.time()
        U_filled_prime, sigma_filled_prime, Vt_filled_prime = server.apply_svd(R_blinded)
        server_svd_end = time.time()
        logging.info(f"Server SVD calculation time (Private): {server_svd_end - server_svd_start} seconds")
        
        # Verify ...
        verification_start = time.time()
        self.verify(R_blinded, U_filled_prime, sigma_filled_prime, Vt_filled_prime)
        verification_end = time.time()
        logging.info(f"Server verification calculation time (Private): {verification_end - verification_start} seconds")


        # Recovery ...
        recovery_start = time.time()
        U_filled,Vt_filled,sigma_filled = self.recovery(U_filled_prime,sigma_filled_prime,Vt_filled_prime)
        recovery_end = time.time()
        logging.info(f"Server Recovery phase (Private): {recovery_end - recovery_start} seconds")
        
        full_df = pd.concat([train_df, test_df])
        R_full_df_filled, _ = self.create_user_item_matrix_with_column_means(full_df)
        predicted_ratings_filled = self.predict_ratings(U_filled, sigma_filled, Vt_filled, R_full_df_filled)
        mae_test_filled = self.evaluate_model(test_df, R_full_df_filled, predicted_ratings_filled)
        print(f"Mean Absolute Error on Test Set: {mae_test_filled}")
        logging.info(f"MAE (Private): {mae_test_filled}")
        
        end_time = time.time()
        logging.info(f"Total runtime (Private): {end_time - start_time} seconds")

    def client_main(self):
        """
        Interactive version of main using exposed ports to communicate with server.
        """

        HOST = '127.0.0.1'
        
        # HOST = 'server-container'   
        PORT = 5000
        

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))
            # Send data to server (blinded matrix and other necessary info)
            start_time = time.time()
            logging.info("NETWORK BASED, Starting Client-Server Private SVD Recommendation System")

            ratings_file_path = 'ratings_percent.csv'
            ratings_df = self.load_dataset(ratings_file_path)
            train_df, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
            R_train_df_filled, _ = self.create_user_item_matrix_with_column_means(train_df)

            # Check if keys need generation
            keygen_start = time.time()
            if not self.K1 or not self.K2 or not self.epsilon:
                self.K1, self.K2, self.epsilon = key_generator.generate_keys(R_train_df_filled.shape[0], R_train_df_filled.shape[1])
            keygen_end = time.time()
            logging.info(f"NETWORK BASED, Key generation time (Private): {keygen_end - keygen_start} seconds")
            R_train_df_filled = np.array(R_train_df_filled)

            # Blind data and send to server
            blind_r_start = time.time()
            R_blinded = self.blind(R_train_df_filled)
            blind_r_end = time.time()
            logging.info(f"NETWORK BASED, R blinding calculation time (Private): {blind_r_end - blind_r_start} seconds")

            # svd 
            server_svd_start = time.time()
            chunk_size = 4096
            bytes_sent = 0
            total_bytes = R_blinded.nbytes
            R_blinded_bytes = R_blinded.tobytes()
            
            while bytes_sent < R_blinded.nbytes:
                end_index = min(bytes_sent + chunk_size, total_bytes)
                chunk = R_blinded_bytes[bytes_sent:end_index]
                s.sendall(chunk)
                bytes_sent += len(chunk)
                print(f'Bytes sent: {bytes_sent}, {bytes_sent / total_bytes * 100:.2f}% of total')

            finished_message = 'finished'.encode()
            s.sendall(finished_message)

            shape_bytes = pickle.dumps({"R_blinded_shape": R_blinded.shape})
            s.sendall(shape_bytes)

            results = b''
            chunks_received_count = 0
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                results += chunk
                chunks_received_count += 1
                print(f'chunks which has been received : {chunks_received_count}')

            results = pickle.loads(results)
            U_filled_prime = results["U_filled_prime"]
            sigma_filled_prime = results["sigma_filled_prime"]
            Vt_filled_prime = results["Vt_filled_prime"]
            server_svd_end = time.time()
            logging.info(f"NETWORK BASED, Server SVD calculation time (Private): {server_svd_end - server_svd_start} seconds")


            # Verify ...
            verification_start = time.time()
            self.verify(R_blinded, U_filled_prime, sigma_filled_prime, Vt_filled_prime)
            verification_end = time.time()
            logging.info(f"NETWORK BASED, Server verification calculation time (Private): {verification_end - verification_start} seconds")

            # Recovery ...
            recovery_start = time.time()
            U_filled, Vt_filled, sigma_filled = self.recovery(U_filled_prime, sigma_filled_prime, Vt_filled_prime)
            recovery_end = time.time()
            logging.info(f"NETWORK BASED, Server Recovery phase (Private): {recovery_end - recovery_start} seconds")

            full_df = pd.concat([train_df, test_df])
            R_full_df_filled, _ = self.create_user_item_matrix_with_column_means(full_df)
            predicted_ratings_filled = self.predict_ratings(U_filled, sigma_filled, Vt_filled, R_full_df_filled)
            mae_test_filled = self.evaluate_model(test_df, R_full_df_filled, predicted_ratings_filled)
            print(f"Mean Absolute Error on Test Set: {mae_test_filled}")
            logging.info(f"MAE (Private): {mae_test_filled}")

            end_time = time.time()
            logging.info(f"NETWORK BASED, Total runtime (Private): {end_time - start_time}")


if __name__ == "__main__":
    client = Client()
    # client.simple_main()
    # client.main()
    client.client_main()


