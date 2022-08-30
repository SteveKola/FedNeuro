# FedNEAT

**How it works:**

1. initilialize dataset on the client

2. create population (collection of models - each individual genome generates a model) from dataset on the client

3. run for up to 300 generations on the client 

4. save results to Python pickle on the client

5. send pickle to server

6. restore progress from each pickle - take one model and have it learn from avery other model


## Why?


"Federated learning enables multiple actors to build a common, robust machine learning model without sharing data" - Wikipedia

FedNEAT doesn't require clients to send data to the server. FedNEAT builds a "common" model by learning from the progress each client made training populations on their own device.

NEAT (neuroevolution in general) is less computationally expensive than gradient based learning. Computational efficiency is the initial reason I first started looking into this.

## Code notes


The main function loads the configuration file, which is static and typed out by hand.
```python

# this is the main function
def run(config_file):
    # Load configuration.
  config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file) # we only modified the activation functions really 
                         # the rest are defaults

```
The population consists of individuals. Individuals are genomes. Each genome contains two sets of genes that describe how to build an artificial neural network.

Here, we create a population:
```python
    # Create the population, which uses our configuration to create individual genomes.
    p = neat.Population(config)
```

We have a fitness function. This functions like a loss function for gradient based learning - how do you decide which models perform well?
```python
#fitness function
    def eval_genomes(genomes, config):
    # each genome includes instructions for creating a neural network
    # config = number of generations to run the algorithm for 
```

Here, we evaluate the fitness of a population: 
```python
    # this starts at generation 0
    # run evolution algorithm run with the eval_genomes fitness function - for 10 generations
    winner = p.run(eval_genomes, 10)
    # winner returns the best genome seen
```

This is how you save the progress you've made (checkpoint that can be restored on server):

```python
 #creates a checkpoint at the 10th generation!
    p.add_reporter(neat.Checkpointer(11)) 
```
I am working on saving progress without sending it between the server and client in pickle form within `saver.py`.