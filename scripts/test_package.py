
### Testing functions for the main functionality...

import unittest

# Your actual package or functions to test
# Example: Assuming you have a module `my_module.py` with a function `add`
# from brainann_lib.my_module import add


class TestPackage(unittest.TestCase):
    
    def test_add_positive_numbers(self):
        result = add(1, 2)
        self.assertEqual(result, 3)
    
    def test_add_negative_numbers(self):
        result = add(-1, -2)
        self.logger.info("This is an info message")
        self.assertEqual(result, -3)
    

# Main function to run tests
if __name__ == '__main__':
    # This will run all test methods in the class
    unittest.main()