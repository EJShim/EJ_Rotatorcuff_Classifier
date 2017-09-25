import numpy as np
import glob

for filename in glob.iglob('/Volumes/EJ/RCT/**/*.npz', recursive = True):
    print(filename)
    data = dict(np.load(filename))
    print(data)




# # import os

# # none_RCT = os.listdir('/Volumes/새 볼륨/RCT_DATA/none RCT')
# # print("# of None-RCT patients : ", len(none_RCT))


# # konkuk_RCT_small = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/small sized RCT')
# # RCT_small = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/Small 14개')

# # print("# of konkuk RCT Small : ", len(konkuk_RCT_small))
# # print("# of RCT small : ", len(RCT_small))
# # print("total RCT small : ", len(konkuk_RCT_small) + len(RCT_small))

# # konkuk_RCT_medium = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/medium sized RCT')
# # RCT_medium = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/Medium 30개')

# # print("# of konkuk RCT Small : ", len(konkuk_RCT_medium))
# # print("# of RCT medium : ", len(RCT_medium))
# # print("total RCT medium : ", len(konkuk_RCT_medium) + len(RCT_medium))

# # konkuk_RCT_large = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/large sized RCT')
# # RCT_large = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/Large 15개')

# # print("# of konkuk RCT Large : ", len(konkuk_RCT_large))
# # print("# of RCT large : ", len(RCT_large))
# # print("total RCT large : ", len(konkuk_RCT_large) + len(RCT_large))

# # konkuk_RCT_massive = os.listdir('/Volumes/새 볼륨/RCT_DATA/RCT/massive RCT')

# # print("total Massive RCT : ", len(konkuk_RCT_massive))


# # print("total : ", len(konkuk_RCT_small) + len(RCT_small) + len(konkuk_RCT_medium) + len(RCT_medium) + len(konkuk_RCT_large) + len(RCT_large) + len(konkuk_RCT_massive))



