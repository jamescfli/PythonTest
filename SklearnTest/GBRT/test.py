from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_hastie_10_2

X, y = make_hastie_10_2(n_samples=10000)
est = GradientBoostingClassifier(n_estimators=200, max_depth=3)     # shallow trees, multiple steps
est.fit(X, y)

pred = est.predict(X)
print est.predict_proba(X)[0:10]
# [[ 0.38754196  0.61245804]    1       Y
#  [ 0.87634627  0.12365373]    -1      Y
#  [ 0.96030223  0.03969777]    -1      Y
#  [ 0.07891596  0.92108404]    1       Y
#  [ 0.14321529  0.85678471]    1       Y
#  [ 0.32209744  0.67790256]    1       Y
#  [ 0.96563392  0.03436608]    -1      Y
#  [ 0.79922611  0.20077389]    -1      Y
#  [ 0.79275279  0.20724721]    -1      Y
#  [ 0.10367769  0.89632231]]   1       Y

print y[0:10]  # [ 1. -1. -1.  1.  1.  1. -1. -1. -1.  1.]

# regularization:
#   tree structure
#   shrinkage - learning rate decay
#   stochastic gradient boosting
