import numpy as np

class AdamW():
    def __init__(self, parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, weight_decay=0.0001):
        self.parameters = parameters
        
        self.lr = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.weight_decay = weight_decay

        self.t = 0
        self.m = {}
        self.v = {}

    def update(self, param, grad, name):
        if grad is None:
            return
        
        if name not in self.m:
            self.m[name] = np.zeros_like(param)
            self.v[name] = np.zeros_like(param)

        self.m[name] = (self.beta1 * self.m[name] + (1.0 - self.beta1) * grad)
        self.v[name] = (self.beta2 * self.v[name] + (1.0 - self.beta2) * (grad ** 2))
        m_hat = self.m[name] / (1.0 - self.beta1 ** self.t)
        v_hat = self.v[name] / (1.0 - self.beta2 ** self.t)
        param -= (self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon))
        if "bias" not in name and "bn" not in name:
            param -= (
                self.lr
                * self.weight_decay
                * param
            )
    def step(self):
        self.t += 1
        for name, param, grad in self.parameters:
            self.update(param, grad, name)