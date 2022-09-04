
#CRDT import
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
  

def initialize():
    # apply random mutations to create 50 individuals for crossover
    global agents
    agents = []
    for i in 50:
        ag = individual()

    # environment state = dataset

    # generate multi sensory spiking neural network (from genotype) using data and "linkage map"

    # genes are portions of chromosomes - could be symmetrically generated from data 

    # linkage map relates to distance between chromosomes that go thru crossover together

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



# in the single threaded version, the dataset is all of MNIST


# CRDTs help with encoding, deduplication, and novelty search
# gset1 = GSet(id=1)
# gset2 = GSet(id=2)
# gset1.add('a')
# gset1.add('b')
# gset1.display() # ['a', 'b']   ----- Output
# gset2.add('b')
# gset2.add('c')
# gset2.display() # ['b', 'c']   ----- Output
# gset1.merge(gset2)   
# gset1.display() # ['a', 'b', 'c']   ----- Output