from numpy import array, linspace, zeros, dot, where, arange, square, mean
from matplotlib.pyplot import figure, scatter, show, plot, title, xlabel, ylabel
from datetime import datetime


class LMSClasifier:
    def __init__(self, learning_rate=0.01, training_epochs=50):
        self.lr = learning_rate
        self.epochs = training_epochs
        self.activation_function = self.sign
        self.weight = None
        self.mean_square_error_threshold = 1e-6

    # Defining training function
    def train(self, input_vector, output):
        print(50 * "_")
        print("Training LMS")
        print(40 * "=")

        sample_size, feature_size = input_vector.shape  # Getting the input dataset shape
        mean_square_error = zeros(self.epochs)

        self.weight = zeros(feature_size)  # Initializing weights as 0

        epoch_counter = 0
        start_time = datetime.now()
        for epoch in range(self.epochs):
            epoch_counter += 1
            error = zeros(sample_size)
            for index, input_vector_index in enumerate(input_vector):
                # Calculating current error
                c_error = output[index] - \
                    (dot(input_vector_index, self.weight))
                error[index] = c_error
                # Updating weight based on instantaneous estimate of the gradient vector
                self.weight = self.weight + self.lr * \
                    (input_vector_index * c_error)

            mean_square_error[epoch] = square(
                mean(error))  # Updating mean_square_error

            if mean_square_error[epoch] <= self.mean_square_error_threshold:
                break

        print(f"Iterations: {epoch_counter}")
        print(f"Points trained: {sample_size}")
        print(
            f"Time taken to train: {(datetime.now() - start_time).total_seconds()}s")

        plot(arange(self.epochs), mean_square_error)
        title("Learning Curve")
        xlabel("Epochs")
        ylabel("Mean Squared Error")
        show()

    # Defining testing function
    def test(self, input_vector):
        print(50 * "_")
        print("Testing LMSClassifier with testing data")
        print(40 * "=")

        # Getting the output dataset shape
        sample_size, feature_size = input_vector.shape
        # Initializing predicted output matrix with 0
        predicted_class = zeros(sample_size)

        start_time = datetime.now()
        # Testing dataset in similar fashion of training data
        for index, input_vector_index in enumerate(input_vector):
            linear_output = dot(self.weight, input_vector_index)
            predicted_class[index] = self.activation_function(linear_output)

        print(f"Points tested: {sample_size}")
        print(
            f"Time taken to test: {(datetime.now() - start_time).total_seconds()}s")

        return predicted_class

    def sign(self, v):  # Activation function
        return where(v >= 0, 1, -1)
