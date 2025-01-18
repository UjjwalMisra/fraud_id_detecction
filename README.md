Fake Account Detection Using Neural Network

Overview

This project aims to detect fake accounts using a neural network model built with TensorFlow and Keras. The model predicts whether an account is genuine or fake based on various features. The project includes data preprocessing, feature scaling, and model evaluation using key performance metrics.

Dataset

Training Dataset: train.csv

Testing Dataset: test.csv

Target Variable: fake

0: Genuine account

1: Fake account

Features Used

Profile picture availability

Account privacy status

Username length

Numerical values in username

Other account-specific metadata

Tools and Libraries

Programming Language: Python

Libraries:

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: TensorFlow, Scikit-learn

Neural Network Architecture

Input Layer: 11 features

Hidden Layers:

Dense layer with 64 neurons (ReLU activation)

Dense layer with 128 neurons (ReLU activation)

Dropout layer (20%)

Dense layer with 64 neurons (ReLU activation)

Dropout layer (20%)

Output Layer: 2 neurons (Softmax activation for classification)

Data Preprocessing

Handling Missing Values: Checked and imputed missing values.

Feature Scaling: Standardized features using StandardScaler.

Encoding Target Variable: One-hot encoding of the fake column.

Training the Model

Optimizer: Adam

Loss Function: Categorical Crossentropy

Metrics: Accuracy

Epochs: 50

Validation Split: 10%

Training Results

The model was trained over 50 epochs, achieving high accuracy on the validation set.

Visualization

Training vs Validation Loss:

Visualized the loss curves to ensure the model is not overfitting.

Confusion Matrix:

Evaluated the model’s performance on the test set.

Results

Evaluation Metrics:

Accuracy: High classification accuracy on test data.

Classification Report: Detailed metrics including precision, recall, and F1-score.

Confusion Matrix

The confusion matrix provided insights into the number of correctly and incorrectly classified accounts.

How to Use

Clone the Repository:

git clone https://github.com/your-username/fake-account-detection.git
cd fake-account-detection

Install Dependencies:

pip install -r requirements.txt

Run the Script:

python fake_account_detection.py

View Results:

Training and validation loss curves.

Confusion matrix and classification report.

File Structure

train.csv: Training dataset

test.csv: Testing dataset

fake_account_detection.py: Main script for preprocessing, training, and evaluation

README.md: Documentation for the project

Future Improvements

Hyperparameter tuning to optimize the model’s performance.

Feature engineering to include additional meaningful features.

Explore other classification models like Random Forest or Gradient Boosting.

Acknowledgments

Special thanks to the contributors of the dataset and the open-source community for providing the tools and resources to make this project possible.

