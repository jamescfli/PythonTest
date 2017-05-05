import argparse

def func1(): pass
def func2(): pass
def func3(): pass

parser = argparse.ArgumentParser()
parser.add_argument(
    "-l", "--list",
    dest='funcs', action="append_const", const=func1,
    help="Create CSV of images", )
parser.add_argument(
    "-i", "--interactive",
    dest='funcs', action="append_const", const=func2,
    help="Run script in interactive mode",)
parser.add_argument(
    "-d", "--dimensions",
    dest='funcs', action='append_const', const=func3,
    help="Copy images with incorrect dimensions to new directory")
args = parser.parse_args()
if not args.funcs:
    args.funcs = [func1, func2, func3]

for func in args.funcs:
    print(func.func_name)
    func()

# $ python default_op.py
# func1
# func2
# func3
