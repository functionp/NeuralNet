#-*- coding: utf-8 -*-

from math import exp
from math import tanh
from math import sin
import random

def sign(x):
    if x > 0 :
        return 1
    else:
        return -1

def sigmoid(x):
    return 1.0 / (1.0 + exp(-x))

def dtanh(x):
    return 1.0 - x * x

def argmax(list):
    current_max = 0
    current_max_index = 0
    for i, element in enumerate(list):
        if current_max <= element:
            current_max = element
            current_max_index = i

    return current_max_index

class NeuralNet(object):
    scale = 0.1

    def __init__(self, input_nodes=[], number_of_hidden=1, number_of_output=1):
        self.set_input_nodes(input_nodes)
        self.set_hidden_nodes([random.uniform(-1.0, 1.0) for i in range(number_of_hidden)])
        self.set_output_nodes([random.uniform(-1.0, 1.0) for i in range(number_of_output)])
        self.set_input_weights([[random.uniform(-1.0, 1.0) for column in range(number_of_hidden)] for row in range(len(self.input_nodes))])
        self.set_output_weights([[random.uniform(-1.0, 1.0) for column in range(number_of_output)] for row in range(number_of_hidden + 1)]) #定数項のため指定の長さより1長い

    def set_input_nodes(self, input_nodes):
        input_nodes.append(1.0) #末尾に定数項を付け加える
        self.input_nodes = input_nodes[:]

    def set_hidden_nodes(self, hidden_nodes):
        hidden_nodes.append(1.0) #末尾に定数項を付け加える
        self.hidden_nodes = hidden_nodes[:]

    def set_output_nodes(self, output_nodes):
        self.output_nodes = output_nodes[:]

    def set_input_weights(self, input_weights):
        self.input_weights = input_weights[:]

    def set_output_weights(self, output_weights):
        self.output_weights = output_weights[:]

    def get_output_errors(self, target_output):
        output_errors = [0.0 for k in range(len(self.output_nodes))]
        for k in range(len(self.output_nodes)):
            error = target_output[k] - self.output_nodes[k]
            output_errors[k] = self.d_activate_output(self.output_nodes[k], error)

        return output_errors

    def get_hidden_errors(self, output_errors):
        """出力の誤差と強度を掛けあわせて中間層の誤差を出す"""

        hidden_errors = [0.0 for j in range(len(self.hidden_nodes))]
        for j in range(len(self.hidden_nodes)):
            error = 0.0
            for k in range(len(self.output_nodes)):
                error += output_errors[k] * self.output_weights[j][k]
            hidden_errors[j] = self.d_activate_hidden(self.hidden_nodes[j], error)

        return hidden_errors

    def d_activate_output(self, output_value, error):
        return dtanh(output_value) * error

    def d_activate_hidden(self, hidden_value, error):
        return dtanh(hidden_value) * error

    def update_output_weights(self, output_errors):
        for j in range(len(self.hidden_nodes)):
            for k in range(len(self.output_nodes)):
                change = output_errors[k] * self.hidden_nodes[j]
                self.output_weights[j][k] += self.scale * change

    def update_input_weights(self, hidden_errors):
        for i in range(len(self.input_nodes)):
            for j in range(len(self.hidden_nodes) - 1):
                change = hidden_errors[j] * self.input_nodes[i]
                self.input_weights[i][j] += self.scale * change

    def back_propagate(self, target_output):
        output_errors = self.get_output_errors(target_output)
        hidden_errors = self.get_hidden_errors(output_errors)

        self.update_output_weights(output_errors)
        self.update_input_weights(hidden_errors)

    def feed_forward(self):
        """入力から順伝播で出力を求める"""

        #隠しノードの更新 定数項を更新しないように-1する
        for j in range(len(self.hidden_nodes) - 1):
            sum_of_values = 0
            for i in range(len(self.input_nodes)):
                 sum_of_values += self.input_nodes[i] * self.input_weights[i][j]
            self.hidden_nodes[j] = self.activate_hidden(sum_of_values)

        #出力ノードの更新
        for k in range(len(self.output_nodes)):
            sum_of_values = 0
            for j in range(len(self.hidden_nodes)):
                 sum_of_values += self.hidden_nodes[j] * self.output_weights[j][k]
            self.output_nodes[k] = self.activate_output(sum_of_values)

    def activate_hidden(self, x):
        return tanh(x)

    def activate_output(self, x):
        return tanh(x)

    def train(self, target_output=[], input_nodes=[]):
        # 入力の指定があればノードを上書き
        if input_nodes != []: self.set_input_nodes(input_nodes[:])

        self.feed_forward()
        self.back_propagate(target_output)

    def train_cases(self, cases, iteration=1000):
        """ (出力, 入力)のペアの配列(cases)をとりiterationの回数反復"""

        for i in range(iteration):
            for case in cases:
                self.train(case[0], case[1])


    def classify(self, input_nodes=[]):
        """inputノードからフィードフォワードして結果を推測"""

        # 入力の指定があればノードを上書き
        if input_nodes != []: self.set_input_nodes(input_nodes)
        self.feed_forward()
        print self.output_nodes

        return argmax(self.output_nodes)

class TwoClassNeuralNet(NeuralNet):

    def __init__(self, input_nodes=[], number_of_hidden=1):
        # 出力ノードの数は1に設定
        super(TwoClassNeuralNet, self).__init__(input_nodes, number_of_hidden, 1)

    def classify(self, input_nodes=[]):
        # 入力の指定があればノードを上書き
        if input_nodes != []: self.set_input_nodes(input_nodes)
        self.feed_forward()

        return sign(self.output_nodes[0])

#関数近似用ニューラルネットワーク
class FunctionNeuralNet(NeuralNet):
    def __init__(self, input_nodes=[], number_of_hidden=1):
        super(FunctionNeuralNet, self).__init__(input_nodes, number_of_hidden, 1)

    def d_activate_output(self, output_value, error):
        return error

    def d_activate_hidden(self, hidden_value, error):
        return dtanh(hidden_value) * error

    #関数近似の際の隠れ層活性化関数はtanh
    def activate_hidden(self, x):
        return tanh(x)

    #関数近似の際の出力活性化関数は線形関数
    def activate_output(self, x):
        return x

    def classify(self, input_nodes=[]):
        """inputノードからフィードフォワードして結果を推測"""

        # 入力の指定があればノードを上書き
        if input_nodes != []: self.set_input_nodes(input_nodes)
        self.feed_forward()

        return self.output_nodes[0]

if __name__ == "__main__":
    nn = TwoClassNeuralNet([0.0, 0.0], 3)

    """
    cases = []
    cases.append(([-1.0], [0.0, 0.0]))
    cases.append(([1.0], [1.0, 0.0]))
    cases.append(([1.0], [0.0, 1.0]))
    cases.append(([-1.0], [1.0, 1.0]))

    nn.train_cases(cases, 1000)

    print nn.classify([0.0, 0.0])
    print nn.classify([1.0, 0.0])
    print nn.classify([0.0, 1.0])
    print nn.classify([1.0, 1.0])

    """
    nn = FunctionNeuralNet([0.0], 3)
    pi = 3.14
    cases = []
    for i in range(70):
        x = (i / 70.0) * 2 * pi
        cases.append(([sin(x)], [x]))
        #print str(x) + ":" + str(sin(x))

    nn.train_cases(cases, 2000)

    print nn.classify([0.0])
    print nn.classify([pi / 6])
    print nn.classify([pi / 2])
    print nn.classify([pi])
    print nn.classify([pi * 7 / 6])
    print nn.classify([pi * 3 / 2])
    print nn.classify([pi * 2])
    print nn.classify([pi * -3 / 2])
    print nn.classify([pi * 5 / 2])
    print nn.classify([pi * 5])
