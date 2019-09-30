# About Catastrophic Forgetting
In conventional neural networks, when new knowledge is learned, there is a problem called catastrophic forgetting that catastrophicly forgets previously learned knowledge. The parameters (weight and bias) obtained by learning in the past are saved with parameters suitable for the knowledge learned at that time. Therefore, when new knowledge is learned, the parameters are updated to parameters suitable for the new knowledge, and the knowledge learned in the past is forgotten.

# About Elastic Weight Consolidation(EWC)
As a measure against catastrophic forgetting, a method of learning knowledge (parameters such as weight and bias) suitable for all learning data by simultaneously learning " previously learned data " and " newly learned data " There is. But this way

1. Increase learning time once
2. The amount of learning data to be saved increases

There are problems such as.