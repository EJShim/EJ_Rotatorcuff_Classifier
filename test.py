import numpy as np
import collections


gtdata = np.load("./additional/annotation_tool/data/5cl_gt.npz")['targets']
humandata = np.load("./data/human_test/swchung.npz")['data']

# for idx, data in enumerate(gtdata):
#     if not data == 0:
#         gtdata[idx] = 1
    
#     if not humandata[idx] == 0:
#         humandata[idx] = 1

sub = np.absolute(np.subtract(gtdata, humandata))


print(sub)
print(collections.Counter(sub))

