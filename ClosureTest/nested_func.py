def print_msg(msg):

    # nested function
    def printer():
        print(msg)

    return printer()    # return a nested function


if __name__ == '__main__':
    print_msg("Hello")
    # define closure func
    another = print_msg("Hello")
    # another     # another() -> TypeError: 'NoneType' object is not callable
    # .. output 'Hello' again
    # message was still remembered although we had already finished executing the print_msg()

    del print_msg
    another
    # print_msg("Hello")      # NameError: name 'print_msg' is not defined

    # a closure in Python when a nested function references a value in its enclosing scope
    # Closures can avoid the use of global values and provides some form of data hiding

    
