import os

import pkg_resources

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(os.path.join(dir_path, 'requirements.txt'), 'r') as f:
    requirement = f.read().splitlines()

# check requirements
print("Checking dependencies...")
for req in requirement:
    try:
        pkg_resources.require([req])
    except Exception as e:
        print(e)
