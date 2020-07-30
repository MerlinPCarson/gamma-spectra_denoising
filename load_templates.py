import sys
import time
import h5py
import argparse


def load_templates(datafile, dettype):

    templates = {'name': [], 'intensity': [], 'keV': []}
    with h5py.File(datafile, 'r') as h5f:
        for item in h5f[dettype]:
            templates['name'].append(item)
            templates['intensity'].append(h5f[dettype][item]['intensity'][()])
            templates['keV'].append(h5f[dettype][item]['keV'][()])

    return templates

def show_hierarchy(datafile):
    with h5py.File(datafile, 'r') as h5f:
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
