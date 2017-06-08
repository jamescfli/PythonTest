# Decorators in Python make an extensive use of closures


# decorating function
def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner


# decorated function
def ordinary():
    print("I am ordinary")

ordinary()
pretty = make_pretty(ordinary)
pretty()

# test decorator
@make_pretty
def ordinary():
    print("I am ordinary")

ordinary()
