#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import subprocess  

# to store output  to txt file 
with open('output.txt', 'a') as file:          
    
    # models 1 to 6 iteration     
    for mode in range(1, 6):          
        subprocess.run(['python', 'testCNN.py', '--mode', str(mode)], stdout=file)

