import numpy as np

feature_a = np.arange(10)
target_a = np.zeros(len(feature_a))

feature_b = np.arange(20)
target_b = np.ones(len(feature_b))


features = np.concatenate((feature_a, feature_b))
targets = np.concatenate((target_a, target_b))

print(features)
print(targets)

data_length = len(features)
random_seed = np.arange(data_length)
np.random.shuffle(random_seed)

features = features[random_seed]
targets = targets[random_seed]

print(features)
print(targets)


