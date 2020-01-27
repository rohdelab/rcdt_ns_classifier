import sys
import pickle

with open(sys.argv[1], 'rb') as f:
    data = pickle.load(f)
    print(data)
