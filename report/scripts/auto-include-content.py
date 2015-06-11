#!/usr/bin/env python2.7

import os
import sys


def print_includes(directory):
    if os.path.exists(directory):
        for f in sorted(os.listdir(directory)):
            f, ext = f.rsplit('.', 1)
            if ext == 'tex':
                print r'\include{{{}}}'.format(os.path.join(directory, f))


if __name__ == '__main__':
    for d in sys.argv[1:]:
        print_includes(d)
