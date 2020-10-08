from numpy import array, genfromtxt, newaxis, nan_to_num, mean, zeros, max, abs
from matplotlib.pyplot import figure, scatter, show

from lms import LMSClasifier

TRAIN_TEST_RATIO = 0.75  # Defining training to testing size ratio


def norm(vec):  # Normalization function for vector
    print(50 * "_")
    print("Normalizing dataset")

    rows, cols = vec.shape
    norm = zeros((rows, cols))
    means = mean(vec, axis=0)
    for i in range(cols):
        norm[:, i] = vec[:, i] - means[i]
    max_num = max(abs(norm))
    for i in range(cols):
        norm[:, i] = norm[:, i] / max_num

    return norm


def main():
    print(50 * "_")
    print("Reading from dataset")

    # Getting dataset from file
    dataset = genfromtxt("data.csv", delimiter=",", skip_header=1)
    rows, columns = dataset.shape

    # Separating input vectors and output labels from the dataset
    dataset_inputs = dataset[:, 1:-1]
    dataset_outputs = dataset[:, columns - 1]

    normalized_dataset_inputs = norm(dataset_inputs)

    # Adjusting output dataset by merging non-seizure instances (labels 2, 3, 4 and 5 in the dataset) to 1 and seizure instance (label 1 in the dataset) to -1
    adjusted_dataset_outputs = array(
        [1 if index > 1 else -1 for index in dataset_outputs])

    training_size = int(TRAIN_TEST_RATIO * rows)
    if training_size % 2 != 0:
        training_size + 1

    # Separating training input and output
    training_input = normalized_dataset_inputs[:training_size, :]
    training_output = adjusted_dataset_outputs[:training_size]

    # Separating testing input and output
    testing_input = normalized_dataset_inputs[training_size:, :]
    testing_output = adjusted_dataset_outputs[training_size:]

    # Using LMSClassifier class defined in lms.py to create new instance of LMSClassifier network
    seizure_lms = LMSClasifier(training_epochs=200)
    seizure_lms.train(training_input, training_output)
    # Getting all predicted outputs of seizures from LMS classifier
    seizure_test_output = seizure_lms.test(testing_input)

    # Calculating accuracy and errors in misclassifications
    seizure_error = 0
    adjusted_test_positive = 0
    seizure_test_positive = 0
    misclassified_instances = []
    for output_index in range(len(testing_output)):
        if testing_output[output_index] != seizure_test_output[output_index]:
            seizure_error += 1
            misclassified_instances.append(output_index)
        if testing_output[output_index] == -1:
            adjusted_test_positive += 1
        if seizure_test_output[output_index] == -1:
            seizure_test_positive += 1

    print(f"Actual seizure subjects: {adjusted_test_positive}")
    print(f"Predicted seizure subjects: {seizure_test_positive}")
    print(f"Classification errors: {seizure_error}")
    print(
        f"Accuracy: {100 - (seizure_error / len(testing_output) * 100)}%")


if __name__ == "__main__":
    main()


'''
__________________________________________________
Reading from dataset
__________________________________________________
Normalizing dataset
__________________________________________________
Training LMS
========================================
Iterations: 200
Points trained: 8625
Time taken to train: 12.759003s
__________________________________________________
Testing LMSClassifier with testing data
========================================
Points tested: 2875
Time taken to test: 0.021033s
Actual seizure subjects: 569
Predicted seizure subjects: 1151
Classification errors: 1130
Accuracy: 60.69565217391305%
'''
