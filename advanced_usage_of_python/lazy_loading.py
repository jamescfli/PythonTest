import weakref
class lazyattribute(object):
    def __init__(self, f):
        self.data = weakref.WeakKeyDictionary()
        self.f = f

    def __get__(self, obj, cls):
        if obj not in self.data:
            self.data[obj] = self.f(obj)
        return self.data[obj]

class Foo(object):
    @lazyattribute
    def bar(self):          # if already had, then skip
        print 'being lazy'  # conduct
        return 42

f = Foo()
print f.bar
print f.bar