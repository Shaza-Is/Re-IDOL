class FirstNet(Model):

    def __init__(self):
        super(FirstNet, self).__init__()
        self.input_layer = layers.Flatten(input_shape=(28,28))
        self.hidden_1 = layers.Dense(20, activation="sigmoid")
        self.hidden_2 = layers.Dense(20, activation="sigmoid")
        self.output_layer = layers.Dense(10)

    def call(self, inputs: Input):
        X = self.input_layer(inputs)
        X = self.hidden_1(X)
        X = self.hidden_2(X)
        X = self.output_layer(X)
