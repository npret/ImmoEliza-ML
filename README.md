# ImmoEliza-ML | Regression

## Description

This repository, **ImmoEliza-ML**, focuses on applying machine learning models to predict real estate prices in Belgium for the company "ImmoEliza." The project is a **consolidation challenge** designed to put theory into practice, specifically around using **Decision Trees and Random Forests** to perform regression analysis.

The project involves several key steps, such as data cleaning, feature engineering, model selection, and evaluation. The goal is to produce a reliable model that accurately predicts property prices using preprocessed data.

**Type of Challenge:** Consolidation 
**Duration:** 6 days 
**Deadline:** 09/12/2024 16:00 
**Team challenge:** Solo

### Learning Objectives
- Apply regression models (Decision Tree + Random Forest) in a real-world context.
- Preprocess data effectively for machine learning tasks.

### The Mission
The real estate company **"ImmoEliza"** has tasked us with creating a machine learning model to predict real estate prices in Belgium. 

Key steps include:
- Handling missing values (NANs).
- Encoding categorical data.
- Feature selection and reducing strong correlation between features.

The ultimate goal is to predict property prices accurately using **Decision Tree and Random Forest** models.

## Installation
To install the necessary packages for this project, please ensure you have Python installed. Then, run:

```bash
pip install -r requirements.txt
```
The main dependencies include:
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

## Usage
To run the project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone https://github.com/npret/ImmoEliza-ML.git
   ```
2. **Navigate to the project folder**:
   ```bash
   cd ImmoEliza-ML
   ```
3. **Run the preprocessing script**:
   ```bash
   python preprocess_data.py
   ```
4. **Train and evaluate the model**:
   ```bash
   python train_model.py
   ```
5. **Visualize Results**:
   Visualizations will be saved in the `visuals/` directory.

## Visuals
This project includes several visualizations of the data exploration and model performance:
- **Feature Correlation Heatmap**: Helps identify which features to remove based on multicollinearity.
- **Model Performance Plots**: Comparison of actual vs predicted prices, which are helpful to assess the model's accuracy.

## Contributors
This project is a solo challenge:
- **Nicole Pretorius**: Responsible for all aspects including data preprocessing, model training, and evaluation.

## Timeline
- **Day 1-2**: Data cleaning and preprocessing.
- **Day 3-4**: Feature engineering and selection.
- **Day 5**: Model implementation using **Decision Trees and Random Forest**.
- **Day 6**: Model evaluation and README enhancement.

## Personal Situation
This project is part of my learning journey at **BeCode**, where I am developing my skills in machine learning.

Through this challenge, I aim to demonstrate my ability to work independently, solve complex problems, and apply theoretical knowledge in a practical setting. It represents an important step in my progression as a data scientist.

