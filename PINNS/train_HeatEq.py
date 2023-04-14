from HeatEq import *

lr = 0.01
net = PDENN(input_size=2, output_size=1, hidden_depth=10, hidden_size=12, lr=lr)

eq = HeatEq(net, [0, 1], [0, 1], 1000, 1000, True)
# eq.setOptim(eq.net.lbfgs)

eq.train(1)