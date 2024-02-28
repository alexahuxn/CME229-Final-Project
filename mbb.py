#!/usr/bin/env python
# coding: utf-8

# In[1]:


def moving_block_bootstrap(data, block_length, num_samples):
    num_obs = len(data)
    output = []
    for i in range(int(num_samples//block_length)):
        start_index = np.random.randint(0, num_obs - block_length + 1)
        output = output + (list(data[start_index:start_index+block_length]))
    return np.array(output)

