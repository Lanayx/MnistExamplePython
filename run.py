import sys
sys.path.append('./src')
sys.path.append('./data')


def test():

    import numpy as np
    import time

    size = 500
    a = np.random.randn(size, size)
    b = np.random.randn(size, size)
    z = np.random.randn(size, size)
    start_time = time.time()
    for i in xrange(1,1000):
        z= np.dot(a,b)

    print len(z)
    print time.time() - start_time

def main():


    import mnist_loader
    import network

    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    net = network.Network([784, 30, 10])
    net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
    #test()



main()