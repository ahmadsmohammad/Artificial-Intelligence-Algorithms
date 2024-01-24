import SumofGaussians as SG
import numpy as np
import sys

def main():
    seed = int(sys.argv[1])
    dims = int(sys.argv[2])
    ncenters = int(sys.argv[3])

    rng = np.random.default_rng(seed)
    sog = SG.SumofGaussians(dims, ncenters, rng)

    #data (epsilon not needed in sa???)
    epsilon = 1e-8

    #random starting point in the d-dimensional [0, 10] cube
    x = rng.uniform(size=(dims)) * 10.0
    
    #temperature vars
    temperature = 1.0
    cooling_rate = 0.999

    #start
    iterations = 0
    while iterations < 100000 and temperature > .0001:

        #this is the previous (coordinates) that we will use to compare below
        prev_value = sog.Evaluate(x)

        #generate a random neighbor within the [-0.05, 0.05] range
        neighbor = x + rng.uniform(low=-0.05, high=0.05, size=dims)

        #calculate difference
        difference = sog.Evaluate(neighbor) - prev_value

        #accept with Metropolis criterion
        if difference > 0 or rng.random() < np.exp(difference / temperature):
            x = neighbor

            #reduce temp by * .999 to make better informed decisions over time
            temperature *= cooling_rate

        #print var
        y = sog.Evaluate(x)

        iterations += 1

        #print the location (x) and the sog function value at each step
        if isinstance(x, np.ndarray):
            print(f"{' '.join(map(str, x))} {y}")
        else:
            print(f"{x} {y}")


main()

