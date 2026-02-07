Using the python library num2words, I generated a dataset with 10,000 samples, 90% of which were used to train an AI. <br>
This specific AI will attempt to turn textual numbers ("two") to numerical numbers ("2").
<p></p>
The model only sees smaller numbers during training, and only larger numbers during testing, which is something that could be trained to improve the generalization. <br>
The number of epochs was also explicitly set to 10, which can be increased for better accuracy.


Number Translation Machine Learning
Converting text-based numbers to numerical digits using neural networks

A lightweight machine learning project that trains a neural network to translate written numbers (e.g., "two hundred forty-three") into their numerical equivalents (e.g., "243"). Built with TensorFlow/Keras and trained on synthetic data generated using the num2words library.

Overview
This project explores sequence-to-sequence learning by training a neural network to recognize patterns in textual number representations. The model learns from smaller numbers during training and is tested on larger numbers to evaluate generalization capabilities.

Key Highlights:

10,000 synthetically generated training samples
90/10 train-test split
Character-level sequence processing
Demonstrates model generalization to unseen number ranges
Project Structure
Number-Translation-Machine-Learning/
├── Number_Translation.py    # Main training script
├── Evaluate Model.py         # Model evaluation script
└── README.md                # This file
How It Works
Data Generation: Uses num2words to create pairs of textual and numerical representations
Training: A sequence-to-sequence model learns character-level patterns in a 90/10 split
Evaluation: Tests on larger numbers not seen during training to assess generalization
The model intentionally trains on smaller numbers and tests on larger ones to demonstrate whether it can learn the underlying translation logic beyond simple memorization.

Getting Started
Prerequisites
Python 3.x
TensorFlow 2.x
NumPy
num2words
Installation
Clone the repository and install dependencies:

bash
git clone https://github.com/shanghanyan/Number-Translation-Machine-Learning.git
cd Number-Translation-Machine-Learning
pip install tensorflow numpy num2words
Usage
Train the model:

bash
python Number_Translation.py
Evaluate the model:

bash
python "Evaluate Model.py"
Configuration
You can adjust these parameters in Number_Translation.py to experiment:

Dataset size: Currently 10,000 samples
Train/test split: 90/10 split
Number of epochs: Set to 10 (increase for better accuracy)
Number ranges: Train on smaller numbers, test on larger ones
Current Limitations & Future Work
Training range: Model currently sees only smaller numbers during training
Generalization: Testing on larger numbers shows room for improvement
Epochs: Limited to 10 epochs; more training could improve accuracy
Architecture: Basic sequence-to-sequence model could be enhanced with attention mechanisms
Potential improvements:

Expand training to include larger number ranges
Increase training epochs for better convergence
Implement data augmentation strategies
Add attention mechanisms for better long-range dependencies
Results
The model demonstrates the fundamental ability to translate text numbers to digits, with performance varying based on the complexity and size of the input numbers. Generalization to larger numbers shows the challenge of learning mathematical patterns from limited examples.
