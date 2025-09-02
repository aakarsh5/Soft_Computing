# Write a Python program to implement a Multi-Layer Perceptron (MLP) on logical XOR function. Do not use built in functions and consider bipolar inputs and outputs.

import math
import random

def bipolar_sigmoid(net):
    e = math.exp(-net)
    return (1.0 - e)/(1.0 + e)

def bipolar_sigmoid_derivative_from_output(f_net):
    return 0.5 * (1.0 - f_net *f_net)

def dot(weights, input):
    s = 0.0
    for w,x in zip(weights, input):
        s += w * x
    return s

class MLP_XOR:
    def __init__(self, learning_rate=0.1, seed=1):
        random.seed(seed)
        self.lr = learning_rate

        self.wh = [
            [random.uniform(-1.0,1.0) for _ in range(3)],
            [random.uniform(-1.0,1.0) for _ in range(3)]
        ]

        self.wo = [random.uniform(-1.0, 1.0) for _ in range(3)]

    def forward(self, x):
        # x is input of two values 
        xin = [x[0], x[1], 1.0] # one bias term is added

        # hidden layer
        hidden_net = []
        hidden_out = []
        for i in range(2):
            net = dot(self.wh[i],xin)
            out = bipolar_sigmoid(net)
            hidden_net.append(net)
            hidden_out.append(out)

        # output layer 
        hout_bias = [hidden_out[0], hidden_out[1], 1.0]
        net_o = dot(self.wo, hout_bias)
        out_o = bipolar_sigmoid(net_o)

        return hidden_net, hidden_out, net_o, out_o
    
    def train(self, pattern, epochs=10000, error_threshold=1e-3, verbose=False):

        for epoch in range(epochs):
            sum_sq_error = 0.0

            for x, target in pattern:
                
                hidden_net, hidden_out, net_o, out_o = self.forward(x)

                #error calculation
                error_o = target - out_o
                sum_sq_error += error_o * error_o

                # output neuron
                delta_o = error_o * bipolar_sigmoid_derivative_from_output(out_o)

                # output weight update
                self.wo[0] += self.lr * delta_o * hidden_out[0]
                self.wo[1] += self.lr * delta_o * hidden_out[1]
                
                # bias update
                self.wo[2] += self.lr * delta_o

                # delta for hidden layer
                delta_h = [0.0, 0.0]
                for i in range(2):
                    # f'(hidden)
                    f_h = hidden_out[i]
                    deriv_h = bipolar_sigmoid_derivative_from_output(f_h)
                    # note: wo[i] is weight from hidden i -> output
                    delta_h[i] = deriv_h * (self.wo[i] * delta_o)

                    # update hidden weights
                    # inputs with bias:
                    xin = [x[0], x[1], 1.0]
                    for j in range(3):
                        self.wh[i][j] += self.lr * delta_h[i] * xin[j]

            mse = sum_sq_error / len(pattern)
            if verbose and (epoch % (epochs // 10 + 1) == 0):
                print(f"Epoch {epoch:5d}, MSE={mse:.6f}")

            if mse < error_threshold:
                if verbose:
                    print(f"Stopping early at epoch {epoch} with MSE={mse:.6f}")
                break

    def predict(self, x):
        _, _, _, out_o = self.forward(x)
        return 1 if out_o >= 0 else -1, out_o


if __name__ == "__main__":
    pattern = [
        ([-1, -1], -1),
        ([-1,  1],  1),
        ([ 1, -1],  1),
        ([ 1,  1], -1),
    ]

    mlp = MLP_XOR(learning_rate=0.2, seed=42)
    mlp.train(pattern, epochs=20000, error_threshold=1e-5, verbose=True)

    print("\nFinal weights (hidden):")
    for i, wh in enumerate(mlp.wh):
        print(f" hidden neuron {i}: {wh}")
    print("Output weights:", mlp.wo)

    print("\nPredictions after training:")
    for x, t in pattern:
        pred_class, out_cont = mlp.predict(x)
        print(f" Input={x} Target={t}  Output_cont={out_cont:.4f}  Pred={pred_class}")