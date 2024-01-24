import SumofGaussians as SG
import numpy as np, sys

def main():
    seed = int(sys.argv[1])
    dims = int(sys.argv[2])
    ncenters = int(sys.argv[3])
    
    rng = np.random.default_rng(seed)
    sog = SG.SumofGaussians(dims,ncenters,rng)
    
    #data
    epsilon = 1e-8
    data_input = np.loadtxt("data-1.in")

    #random starting point in the d dimensional [0,10] cube
    x = rng.uniform(size=(dims))*10.0
    step_size = 0.01

    iterations = 0
    while iterations < 100000:
        prev_value = sog.Evaluate(x)
        gradient = sog.Gradient(x)
        
        #update x in each dimension using step size of (0.01 * change in gradient)
        x = x + step_size * gradient
        
        #get new coords
        new_values = sog.Evaluate(x)
        
        #break when no longer increasing in epsilon tolerance
        if np.all(new_values - prev_value < epsilon):
            print(f"{' '.join(map(str, x))} {new_values}")
            break

        iterations += 1

        # Print the location (x) and the SumofGaussians function value at each step
      #  if isinstance(x, np.ndarray):
          #  print(f"{' '.join(map(str, x))} {new_values}")
      #  else:
        #    print(f"{x} {new_values}")


main()


    