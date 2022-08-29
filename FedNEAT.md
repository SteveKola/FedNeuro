# FedNEAT

**How to run:**

1. initilialize dataset on the client

2. create population (collection of models - each individual is a model) from dataset on the client

3. run for up to 300 generations on the client 

4. save results to Python pickle on the client

5. send pickle to server

6. restore progress from each pickle - take one model and have it learn from avery other model


## Why?


"Federated learning enables multiple actors to build a common, robust machine learning model without sharing data" - Wikipedia

FedNEAT doesn't require clients to send data to the server. FedNEAT builds a "common" model by learning from the progress each client made training populations on their own device.

NEAT (neuroevolution in general) is less computationally expensive than gradient based learning. Computational efficiency is the initial reason I first started looking into this.

## Our research

There are absolutely no papers that exist related to federated neuroevolution. I am shocked.

Not only is this a much more innovative idea than exploring the convergence of federated averaging, it is also easier to execute. I started coding this less than an hour ago after learning about neuroevolution at a C4AI Zoom call last week.