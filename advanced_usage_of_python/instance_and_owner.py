# bound method is the core of classic classes
# resort to basic class
class Celsius(object):
    def __init__(self, value=0.0):
        print '__init__() called'
        self.value = float(value)

    def __get__(self, instance, owner):     # instance = temp, owner = Temperature
        print '__get__() called'
        return self.value

    def __set__(self, instance, value):
        print '__set__() called'
        self.value = float(value)

class Temperature(object):
    celsius = Celsius()     # __init__()

temp = Temperature()
temp.celsius        # call __get__()
print temp.celsius  # call __get__()
temp.celsius = 3.0  # call __set__()
print temp.celsius  # call __get__()