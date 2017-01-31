class WebFramework(object):
    def __init__(self, name = 'Flask'):
        self.name = name

    def __get__(self, instance, owner):
        print instance  # None                  __main__.WebFramework object
        print owner     # __main__.PythonSite   __main__.WebFramework
        return self.name

    def __set__(self, instance, value):
        print instance
        self.name = value

class PythonSite(object):
    webframework = WebFramework()
    version = 0.01

    def __init__(self, site):
        self.site = site

# # Test (1)
# print PythonSite.webframework       # Flask
# PythonSite.webframework = 'Tornado'
# print PythonSite.webframework       # Tornado

# # Test (2)
# webframework = WebFramework()
# print webframework.__get__(webframework, WebFramework)  # Flask

# Test (3)
pysite = PythonSite('Ghost')
print vars(PythonSite).items()
print vars(pysite).items()  # only ('site', 'Ghost')
print PythonSite.__dict__

# Test (4)
print pysite.version                    # 0.01
print pysite.__dict__['version']        # KeyError: 'version'
print type(pysite).__dict__['version']  # 0.01
