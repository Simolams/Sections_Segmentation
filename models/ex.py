import os 

current_directory = os.getcwd()
import sys
current_directory = current_directory.replace('\models','')
sys.path.append(current_directory)
print((current_directory))

import variables