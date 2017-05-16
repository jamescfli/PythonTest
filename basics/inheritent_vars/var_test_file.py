import os, sys


print __file__
# .. ~/Development/PythonTest/basics/inheritent_vars/var_test.py

print os.path.dirname(__file__)
# .. ~/Development/PythonTest/basics/inheritent_vars

print sys.path
print sys.path.__class__    # list

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
print sys.path
# .. add '~/Development/PythonTest/basics/inheritent_vars/..'
