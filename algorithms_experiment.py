
from functools import partial
from math import log

from algorithms import *

database = {'a': 501, 'b': 500}
queries = [lambda x: x['a'], lambda x: x['b']]

result = report_noisy_max(database, queries, epsilon=0.1)

diff = result[0].difference(result[1])
diffCDF = result[0].differenceCDF(result[1])

divergence = State.fromfun(lambda x: log(result[0](x)/result[1](x)),R)
def interval_divergence(x, interval=100):
	first = result[0].cdf(x+interval/2)-result[0].cdf(x-interval/2)
	second = result[1].cdf(x+interval/2)-result[1].cdf(x-interval/2)
	return log(first/second)
divergence2 = State.fromfun(interval_divergence,R)

plot(map(lambda x: x.state, result), interval=R(480,520), title="Query distributions", block=False)
plot([diff], interval=R(0,200), title="PDF of difference between queries", block=False)
plot([diffCDF], interval=R(0,200), title="CDF of difference between queries", block=False)
plot([divergence], interval=R(400,700), title="Pointwise divergence", block=False)
plot([divergence2], interval=R(400,700), title="Divergence on interval of 100", block=False)
print("P('a' > 'b'):")
print(result[0].larger(result[1]))

input("Press [enter] to continue.")
