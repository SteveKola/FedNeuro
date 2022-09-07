# FedNeuro

**How it works:**

1. initilialize dataset on the client

2. create population (collection of models - each individual genome generates a model) from dataset on the client

3. run for up to 300 generations on the client 

4. save results to bitstring representation of genome

5. send bitstring (genome) to server

6. restore progress from each genome - take one genome and make it evolve using knowledge from every other genome  (Bayes)


## Why?

"Federated learning enables multiple actors to build a common, robust machine learning model without sharing data" - Wikipedia

FedNeuro doesn't require clients to send data to the server. FedNeuro builds a "common" model by learning from the progress each client made training populations on their own device.

