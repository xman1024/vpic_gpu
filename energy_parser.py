#/usr/bin/env python
import sys

def main():
    energies1 = []
    energies2 = []
    with open(sys.argv[1]) as f:
        for line in f:
            energies1.append([float(x) for x in line.split(" ")[1:]])
    with open(sys.argv[2]) as f:
        for line in f:
            energies2.append([float(x) for x in line.split(" ")[1:]])

    res = 0
    for e1, e2 in zip(energies1, energies2):
        for v1, v2 in zip(e1, e2):
            res = max(res, abs(v1 - v2))
    print(res)

if __name__ == "__main__":
    main()

