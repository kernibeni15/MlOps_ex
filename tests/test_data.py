import unittest
from data import mnist
import torch

N_train = 25000
N_test = 5000


class TestData(unittest.TestCase):
    def test_data_shapes(self):
        train_set, test_set = mnist()

        # Check the length of the datasets
        self.assertEqual(len(train_set), N_train)
        self.assertEqual(len(test_set), N_test)

        # Check shapes of train data
        for data, label in train_set:
            self.assertEqual(data.shape, torch.Size([1, 28, 28]))  # Assuming you format it as [1, 28, 28]
            self.assertEqual(label.shape, torch.Size([]))  # Assuming labels are scalars

        # Check shapes of test data
        for data, label in test_set:
            self.assertEqual(data.shape, torch.Size([1, 28, 28]))
            self.assertEqual(label.shape, torch.Size([]))

    def test_label_representation(self):
        train_set, test_set = mnist()

        # Get unique labels from the datasets
        unique_labels_train = set(label.item() for _, label in train_set)
        unique_labels_test = set(label.item() for _, label in test_set)

        # Check that all labels are represented in both train and test sets
        self.assertEqual(unique_labels_train, set(range(10)))  # Assuming labels range from 0 to 9
        self.assertEqual(unique_labels_test, set(range(10)))


if __name__ == '__main__':
    unittest.main()