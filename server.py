import numpy as np
import socket
import pickle

def apply_svd(R):
    U, sigma, Vt = np.linalg.svd(R, full_matrices=True)
    
    sigma_matrix = np.zeros((U.shape[1], Vt.shape[0]))
    np.fill_diagonal(sigma_matrix, sigma)
    
    return U, sigma_matrix, Vt

HOST = '127.0.0.1'
# HOST = '0.0.0.0'
PORT = 5000

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.bind((HOST, PORT))
    s.listen()

    while True:
        conn, addr = s.accept()
        with conn:
            print(f'Connected by {addr}')

            R_blinded_bytes = b''
            while True:
                chunk = conn.recv(4096)
                if b'finished' in chunk:
                    R_blinded_bytes += chunk.replace(b'finished', b'')
                    break
                R_blinded_bytes += chunk
            
            shape_data = pickle.loads(conn.recv(4096))
            R_blinded_shape = shape_data["R_blinded_shape"]
            R_blinded = np.frombuffer(R_blinded_bytes, dtype=np.float64).reshape(R_blinded_shape)

            # Apply SVD and send results in chunks
            print(f'applying svd computation...')
            U, sigma, Vt = apply_svd(R_blinded)

            results = {
                "U_filled_prime": U,
                "sigma_filled_prime": sigma,
                "Vt_filled_prime": Vt
            }
            results_bytes = pickle.dumps(results)

            bytes_sent = 0
            while bytes_sent < len(results_bytes):
                chunk = results_bytes[bytes_sent:bytes_sent+4096]
                conn.sendall(chunk)
                bytes_sent += 4096
                print(f'Bytest has been sent {bytes_sent} : {bytes_sent / len(results_bytes)} % ')