# closures can provide an alternate and more elegant solutions
# closures can avoid the use of global values and provides some form of data hiding.
# It can also provide an object oriented solution to the problem.

def make_multiplier_of(n):
    def multiplier(x):
        return x * n
    return multiplier

# Multiplier of 3
times3 = make_multiplier_of(3)

# Multiplier of 5
times5 = make_multiplier_of(5)

# Output: 27
print(times3(9))

# Output: 15
print(times5(3))

# Output: 30
print(times5(times3(2)))

print make_multiplier_of.__closure__    # None
print times3.__closure__    # (<cell at 0x106d1ead0: int object at 0x7ff52ec0b9d8>,)
print times3.__closure__[0].cell_contents   # 3
