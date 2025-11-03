import numpy as np

class MLP:
    def __init__(self):
        self.weights = None

    def train(self, inputs, outputs, alpha, epochs):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.alpha = alpha
        self.epochs = epochs

        w11 = np.random.uniform(0, 1)
        w12 = np.random.uniform(0, 1)
        w21 = np.random.uniform(0, 1)
        w22 = np.random.uniform(0, 1)

        wh1 = np.random.uniform(0, 1)
        wh2 = np.random.uniform(0, 1)

        b1 = np.random.uniform(0, 1)
        b2 = np.random.uniform(0, 1)
        b3 = np.random.uniform(0, 1)

        for i in range(self.epochs):
            for j in range(len(self.inputs)):
                x1 = self.inputs[j][0]
                x2 = self.inputs[j][1]
                h1 = 1 / (1 + np.exp(- (x1 * w11 + x2 * w21 + b1)))
                h2 = 1 / (1 + np.exp(- (x1 * w12 + x2 * w22 + b2)))

                y = 1 / (1 + np.exp(- (h1 * wh1 + h2 * wh2 + b3)))

                error = self.outputs[j][0] - y

                derivative_y = y * (1 - y) * error
                derivative_h1 = h1 * (1 - h1) * wh1 * derivative_y
                derivative_h2 = h2 * (1 - h2) * wh2 * derivative_y

                delta_w11 = self.alpha * derivative_h1 * x1
                delta_w21 = self.alpha * derivative_h1 * x2
                delta_w12 = self.alpha * derivative_h2 * x1
                delta_w22 = self.alpha * derivative_h2 * x2

                delta_b1 = self.alpha * derivative_h1
                delta_b2 = self.alpha * derivative_h2
                delta_b3 = self.alpha * derivative_y

                delta_wh1 = self.alpha * derivative_y * h1
                delta_wh2 = self.alpha * derivative_y * h2

                w11 += delta_w11
                w21 += delta_w21
                w12 += delta_w12
                w22 += delta_w22

                b1 += delta_b1
                b2 += delta_b2
                b3 += delta_b3

                wh1 += delta_wh1
                wh2 += delta_wh2

        self.weights = (w11, w12, w21, w22, wh1, wh2, b1, b2, b3)
        return self.weights

    def predict(self, x1, x2):
        w11, w12, w21, w22, wh1, wh2, b1, b2, b3 = self.weights

        hidden1 = 1 / (1 + np.exp(- (x1 * w11 + x2 * w21 + b1)))
        hidden2 = 1 / (1 + np.exp(- (x1 * w12 + x2 * w22 + b2)))

        out = 1 / (1 + np.exp(- (hidden1 * wh1 + hidden2 * wh2 + b3)))

        return 1 if out > 0.5 else 0


if __name__ == "__main__":
    
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0], [1], [1], [0]]

    mlp = MLP()

    weights = mlp.train(inputs, outputs, alpha=0.05, epochs=10000)

    print("Pesos aprendidos:", weights)

    print("Resultados de predições após treinamento:")
    for x1, x2 in inputs:
        y_pred = mlp.predict(x1, x2)
        print(f"Entrada: {x1},{x2} => Predição: {y_pred}")

