import numpy as np

import itertools
permu_iter = itertools.permutations(np.arange(1, 10))

for seq in permu_iter:
    # print(seq)
    if (seq[0]*10+seq[1])*seq[2] == (seq[3]*10+seq[4]):
        if (seq[3]*10+seq[4]) + (seq[5]*10+seq[6]) == (seq[7]*10+seq[8]):
            print('{}{}x{}={}{}, {}{}+{}{}={}{}'.format(seq[0], seq[1], seq[2], seq[3], seq[4],
                                                              seq[3], seq[4], seq[5], seq[6], seq[7], seq[8]))
        else:
            continue
    else:
        continue
