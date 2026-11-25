"""Read-only inspection of extracted ETH3D undistorted datasets."""
import argparse
import hashlib
import re
from pathlib import Path
from run_compare import ROOT, image_folder


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',required=True)
    args=ap.parse_args()
    try:
        folder, photos=image_folder(args.dataset)
    except ValueError as e:
        ap.error(str(e))
    expected={'office_26_undistorted':26,'kicker_31_undistorted':31}.get(args.dataset)
    print('Project:', ROOT)
    print('Dataset:',args.dataset)
    print('Images folder:',folder)
    print('Image count:',len(photos),'Expected:',expected)
    print('First:', photos[0].name, 'Last:', photos[-1].name)
    if expected and len(photos)!=expected:
        print('WARNING: unexpected number of input images. Check archive extraction.')
    ref=list((ROOT/'data'/args.dataset).rglob('cameras.txt'))
    if ref:
        for f in ref:
            models=sorted(set(x.split()[1] for x in f.read_text(encoding='utf-8',errors='replace').splitlines()
                              if x.strip() and not x.lstrip().startswith('#') and len(x.split())>3))
            print('Reference camera intrinsics (not fed to reconstruction):',f)
            print('Camera model(s):',','.join(models))
    else:
        print('No cameras.txt found. This is okay for the image-only baseline.')
    print('Images with sizes under 1 kB:',[p.name for p in photos if p.stat().st_size<1024])
    print('\nReady to run dry-run on selected views.')

if __name__=='__main__':
    main()
