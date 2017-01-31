class demo(object):
    pass

obj = demo()
print "class of obj is {0}".format(obj.__class__)   # __main__.demo
print "class of demo is {0}".format(demo.__class__) # type 'type'
# where 'type' is a meta-class, i.e. Module in Ruby

