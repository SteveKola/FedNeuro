from py3crdt.gset import GSet

gsetid = 0

def getid():
    eed = gsetid + 1
    return eed

def testid():
    one = getid()
    print(one) 

testid() #should print 1