class decorator(object):
    def __init__(self, func):
        print('inside decorator.__init__()')
        func()  # prove function definition has been completed

    def __call__(self, *args, **kwargs):
        print('inside decorator.__call__()')

@decorator
def function():
    print('inside function()')
    print('finished decorating function()')

function()
# inside decorator.__init__()
# inside function()
# finished decorating function()
# inside decorator.__call__()