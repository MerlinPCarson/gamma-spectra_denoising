import sys
import time
import h5py
import argparse


def show_hierarchy(datafile):
    with h5py.File('data/templates.h5', 'r') as h5f:
        for k in h5f.keys():
            print(f'{k}: {len(h5f[k])} nuclides')
            for k2 in h5f[k].keys():
                print(f'  {k2}')
                for k3 in h5f[k][k2].keys():
                    print(f'    {k3} {h5f[k][k2][k3].shape}: {h5f[k][k2][k3][()]}') 

def main():
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("-df", "--datafile", help="data file containing templates", default="data/templates.h5")
    arg = parser.parse_args()

    show_hierarchy(arg.datafile)

    print(f'\nScript completed in {time.time()-start:.2f} secs')

    return 0

if __name__ == '__main__':
    sys.exit(main())
