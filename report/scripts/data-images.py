#!/usr/bin/env python2.7

import os
import sys
import random

def print_includes(directory):
    if os.path.exists(directory) and os.path.isdir(directory):
        for subd in sorted(os.listdir(directory)):
            subdabs = os.path.join(directory, subd)
            if os.path.isdir(subdabs):
                files = list(os.listdir(subdabs))
                for _ in range(0, 2):
                    f = random.choice(files)
                    fabs = os.path.join(subdabs, f)
                    f, ext = f.rsplit('.', 1)
                    if ext == 'png':
                        print r'\includegraphics[width=0.06\textwidth]{{{}}}'.format(fabs)


if __name__ == '__main__':
    for d in sys.argv[1:]:
        print_includes(d)
