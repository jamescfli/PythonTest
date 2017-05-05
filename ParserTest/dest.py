# Omit the dest parameter when using a positional argument.
# The name supplied for the positional argument will be the name of the argument:

import argparse


myparser = argparse.ArgumentParser(description='parser test')
myparser.add_argument("product_1", help="enter product1")
myparser.add_argument("product_2", help="enter product2")

args = myparser.parse_args()
firstProduct = args.product_1
secondProduct = args.product_2

print(firstProduct, secondProduct)

# $ python dest.py
# usage: dest.py [-h] product_1 product_2
# dest.py: error: too few arguments

# $ python dest.py abc bcd
# ('abc', 'bcd')