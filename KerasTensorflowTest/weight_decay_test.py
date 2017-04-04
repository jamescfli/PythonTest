# time-based (decrease gradually based on the epoch)
# and
# drop-based (decrease at punctuated large drops at specific epochs)
# learning rate schedule
# simplest time-based

import numpy as np

# 1) time-based
# LearningRate = LearningRate * 1/(1 + decay * epoch)
LearningRate = 0.1 * 1/(1 + 0.0 * 1)
LearningRate = 0.1
# if initial learning rate value of 0.1 and the decay of 0.001
lr_initial = 0.1
decay = 0.001
for epoch in np.arange(5):
    LearningRate = lr_initial * 1.0/(1+decay * epoch)
    print LearningRate

# Decay = LearningRate / Epochs
Decay = 0.1 / 100
Decay = 0.001