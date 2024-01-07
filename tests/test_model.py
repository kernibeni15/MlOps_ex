import unittest
import pytest
import torch
from torch.testing import assert_allclose
from model import myawesomemodel  # Replace with the actual module name

def test_error_on_wrong_shape():
    with pytest.raises(ValueError, match='Expected input to a 4D tensor'):
        myawesomemodel(torch.randn(1,2,3))

class TestMyAwesomeModel(unittest.TestCase):
    def test_output_shape(self):
        # Define the input shape (replace with your actual input shape)
        input_shape = (1, 1, 28, 28)  # Assuming input shape [B, C, H, W]

        # Generate a random input tensor with the specified shape
        input_tensor = torch.randn(input_shape)

        # Forward pass through the model
        output_tensor = myawesomemodel(input_tensor)

        # Define the expected output shape (replace with your actual expected output shape)
        expected_output_shape = (input_shape[0], 10)  # Assuming output shape [B, 10]

        # Check if the output shape matches the expected shape
        self.assertEqual(output_tensor.shape, torch.Size(expected_output_shape))

    def test_output_values(self):
        # Define the input shape (replace with your actual input shape)
        input_shape = (1, 1, 28, 28)  # Assuming input shape [B, C, H, W]

        # Generate a random input tensor with the specified shape
        input_tensor = torch.randn(input_shape)

        # Forward pass through the model
        output_tensor = myawesomemodel(input_tensor)

        # Perform additional checks on the output values (replace with your specific checks)
        assert_allclose(output_tensor.sum().item(), 0.0, rtol=1e-5, atol=1e-8)
        # Add more assertions based on your specific criteria

if __name__ == '__main__':
    unittest.main()
