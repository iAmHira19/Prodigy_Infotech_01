House Price Prediction Using Linear Regression
This project implements a linear regression model to predict house prices based on square footage, number of bedrooms, and number of bathrooms. The model is trained on the "House Prices: Advanced Regression Techniques" dataset from Kaggle.

Requirements
Python 3.x
pandas
numpy
scikit-learn
matplotlib
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your_username/House-prediction.git
cd house-price-prediction
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Usage
Place your train.csv file (from Kaggle) in the project directory.

Run the house_prediction.py script:

bash
Copy code
python house_prediction.py
The script will train the linear regression model, evaluate its performance, generate a scatter plot comparing actual vs. predicted prices, and output the predicted price for a sample house.

File Descriptions
house_prediction.py: Main Python script implementing the model.
train.csv: Dataset used for training and testing the model.
License
This project is licensed under the MIT License - see the LICENSE file for details.

