
import numpy as np

#CRDT import
from json import load
from py3crdt.gset import GSet

# global state 
gsetid = 0

def getid():
    number = gsetid + 1
    return number

class individual:
    def __init__(self, genotype):
        num = getid()
        genotype = GSet(id=num)
        self.g = genotype
     # genotype is individual specific, genome is not


# in the single threaded version, the dataset is all of MNIST
# load mnist data (preprocessed through unsupervised learning)
def mnist():
    data = np.load("data.npy")
    return data



def initialize():
    # apply random mutations to create 50 individuals for crossover
    global agents
    agents = []
    for i in 50:
        ag = individual()

    # environment state = dataset
    env = mnist()

    # derive genotype from environment

    # derive linkage map (close chromosomes go through crossover together)

    # generate multi sensory spiking neural network (from genotype) using linkage map

    # network connections probabilistically generated from linkage map





        


def crossover():


    


    # symmetrical self-repair when damaged

    # combine with other chromosomes at similar regions (gset merge)

    # phenotype = genotype #some transformation here
    # phenotype is influenced by environment, phenome is not

    # neural anisotropy to decide fitness for next phase
    return

def performance():


    #novelty search decides what gets sent to server
    return





