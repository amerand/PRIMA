#!/Library/Frameworks/Python.framework/Versions/Current/bin/python2.7

import prima
import os
import sys

def main():
    if not os.path.exists(sys.argv[1]):
        print 'ERROR: file '+sys.argv[1]+' does not exist'
    if len(sys.argv)>2:
        if not sys.argv[2].split('=')[0]=='pssguiRecorderDir':
            print 'ERROR: second argument should be pssguiRecorderDir=/xxx/xxx/xxx'
        if not os.path.exists(sys.argv[2].split('=')[1]):
            print 'ERROR: directory '+sys.argv[2].split('=')[1]+' does not exist'
        else:
            pssDir = sys.argv[2].split('=')[1]
    else:
        pssDir=None
    a = prima.drs(sys.argv[1], pssRec=pssDir)
    a.astrometryFtk(writeOut=1, overwrite=1, max_GD_um=2.5, max_err_um = 2.0,
                    sigma_clipping=3.5)
    a=[]
main()

