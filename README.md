# EfficientRecommendationSystem

One of the key challenges for recommendation systems is improving their efficiency, especially when dealing with thousands of users and millions of products. A popular approach in recommendation systems is the creation of a user-item matrix followed by matrix decompositions, such as Singular Value Decomposition (SVD).

The paper ["Efficient, Secure, and Verifiable Outsourcing Scheme for SVD-based Collaborative Filtering Recommender System"](https://doi.org/10.1016/j.future.2023.07.042) introduces a method to improve the efficiency of SVD computations on user-item matrices. It addresses efficiency, privacy, and verifiability. In this project, I have implemented the approach described in the paper using containers (representing the client and server), experimenting with various configurations and specifications for both the client and server.

I have created the following sequence diagram, which illustrates the implemented approach based on the paper:  
![Sequence Diagram](https://github.com/user-attachments/assets/39037a42-2d06-41b8-b154-169b623e7630)

The paper suggests that the client, which may have limited computational resources, should only encrypt the data to ensure privacy, while offloading the more computationally expensive SVD calculation to a more powerful server.

## Implementation Details

While implementing this approach, I explored the question: *What should the ratio of server to client specifications be for optimal efficiency? What if the client has sufficient resources?* Additionally, network costs are a significant consideration in this model.

To investigate this, I divided my code into containers with different specifications for both the client and the server to test and implement the algorithm.

Below are the Dockerfiles I created for the client and server:  
![Dockerfiles](https://github.com/user-attachments/assets/4b048781-c5e6-4afe-b0e0-4d369b430355)

**Files:**
- `dockerfile.client`
- `dockerfile.server`

Additionally, I developed three main scripts:
1. `client.py`
2. `server.py`
3. `keyGenerator.py`

For comparison, I also implemented a simple version of the system where all computations (including SVD) happen on a single instance:
1. `single_svd_client_recommendation_system.py`

## Findings

One important observation during the experiment was that network overhead (data transmission) played a significant role in performance. Despite the reduced computation time for SVD (from 0.75s to 0.016s, as shown in the figure below), the network overhead created only a slight performance difference between the two approaches.

![SVD Time Results](https://github.com/user-attachments/assets/dfad2c1c-c71c-4029-94da-3799e27e583f)  
![Performance Comparison](https://github.com/user-attachments/assets/26690ce3-3846-4327-b58b-b4f7ecf11c0e)

## Running the Code

To run this implementation, follow these steps:

1. Build and configure the containers using the Dockerfiles (`dockerfile.server` and `dockerfile.client`).
2. Copy `client.py` and `keyGenerator.py` to the client machine.
3. Copy `server.py` to the server machine.
4. On the server, run the command:  
   ```bash
   python server.py
    ```
   This will start a socket server to listen for incoming requests.
5. On the client side, run client.py with the command:
     ```bash
     python client.py
     ```
     

