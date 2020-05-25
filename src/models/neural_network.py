from keras.models import Model
from keras.layers import Dense, BatchNormalization, Activation, Dropout, Input
from keras import initializers
import keras


class SimpleNeuralNetwork:
    def __init__(
            self,
            number_of_inputs,
            number_of_layers=1,
            neurons_per_layer=100,
            dropout_rate_per_layer=0.1,
            learning_rate=0.01,
            epochs=20,
            batch_size=256
    ):
        """
        Class for creating different network architectures to be used for Bayesian hyperparameter search.

        :param number_of_inputs: Number of input features
        :param number_of_layers: Number of hidden layers
        :param neurons_per_layer: Number of neurons per layer (same for every layer for simplicity)
        :param dropout_rate_per_layer: Dropout rate per layer (same for every layer for simplicity)
        :param learning_rate: Learning rate
        :param epochs: Number of epochs
        :param batch_size: Batch size
        """
        self.number_of_inputs = number_of_inputs
        self.number_of_layers = number_of_layers
        self.neurons_per_layer = neurons_per_layer
        self.dropout_rate_per_layer = dropout_rate_per_layer
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

    def create_model(self):
        X_input = Input(shape=(self.number_of_inputs,))

        # 1st layer
        X = self.add_layer(X_input)

        # Additional layers
        for layer in range(self.number_of_layers - 1):
            X = self.add_layer(X)

        output = Dense(units=1, activation='sigmoid')(X)

        model = Model(inputs=X_input, outputs=output)

        return model

    def add_layer(self, X):
        X = Dropout(rate=self.dropout_rate_per_layer)(X)
        X = Dense(units=self.neurons_per_layer, kernel_initializer=initializers.glorot_normal())(X)
        X = BatchNormalization()(X)
        X = Activation('relu')(X)

        return X

    def compile_model(self):
        simple_neural_network = self.create_model()

        rocauc = keras.metrics.AUC(num_thresholds=200, curve='ROC', summation_method='interpolation', name='roc_auc')
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        simple_neural_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[rocauc])

        self.compiled_model = simple_neural_network

    def fit(self, X, y, validation_data):
        self.compiled_model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=0
        )

    def predict_proba(self, X):
        return self.compiled_model.predict(X)

    def summary(self):
        return self.compiled_model.summary()
