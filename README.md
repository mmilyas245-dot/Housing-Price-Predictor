# 🏠 Housing Price Prediction

> A machine learning project that predicts house prices based on property features using a complete data science workflow, from data preprocessing to model evaluation.

---

# 📖 Overview

Buying or selling a house involves many factors, and estimating the right price can be challenging. This project applies machine learning techniques to predict housing prices using features such as area, number of bedrooms, bathrooms, parking spaces, furnishing status, and other property characteristics.

The project was developed to practice the complete machine learning pipeline while creating a reusable and well-documented regression model.

---

# 🎯 Problem Statement

House prices are influenced by multiple factors, making manual estimation difficult and often inconsistent. The goal of this project is to build a regression model that learns the relationship between housing features and selling price, enabling accurate price predictions for unseen properties.

---

# 🚀 Features

- Complete data preprocessing pipeline
- Data cleaning and duplicate removal
- Exploratory Data Analysis (EDA)
- Outlier detection using the IQR method
- Correlation analysis and visualization
- Categorical feature encoding
- Train-test split for unbiased evaluation
- Machine Learning model training
- Performance evaluation using regression metrics
- Easy-to-understand notebook with step-by-step workflow

---

# 🛠 Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python 3.x |
| IDE | Jupyter Notebook |
| Data Analysis | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-learn |
| Version Control | Git & GitHub |

---

# ⚙️ Environment Setup

## Prerequisites

- Python 3.10 or later
- Git
- Jupyter Notebook

## Installation

Clone the repository

```bash
git clone https://github.com/mmilyas245-dot/Housing-Price-Predictor.git
```

Move into the project directory

```bash
cd Housing-Price-Prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook

```bash
jupyter notebook
```

Open

```
House_Price_Prediction.ipynb
```

and run all cells.

---

# 📂 Dataset

The dataset contains information about residential properties.

### Features

- Area
- Bedrooms
- Bathrooms
- Stories
- Parking
- Main Road
- Guest Room
- Basement
- Hot Water Heating
- Air Conditioning
- Preferred Area
- Furnishing Status

### Target

- Price

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Loaded the dataset using Pandas
- Checked data types
- Removed duplicate records
- Explored missing values
- Encoded categorical variables
- Detected outliers using the Interquartile Range (IQR)
- Visualized feature distributions
- Generated correlation heatmap
- Prepared features for machine learning

---

# 📊 Exploratory Data Analysis

EDA included:

- Summary statistics
- Histograms
- Boxplots
- Correlation Heatmap
- Distribution Plots
- Pairwise Feature Analysis

The analysis helped identify relationships between variables, understand feature distributions, and detect potential outliers before model training.

---

# 🤖 Model Development

The dataset was divided into:

- Training Set (80%)
- Testing Set (20%)

The project uses a **Linear Regression** model as the baseline approach for predicting housing prices.

The workflow includes:

1. Feature Selection
2. Data Splitting
3. Model Training
4. Prediction
5. Performance Evaluation

---

# 📏 Evaluation Metrics

Model performance was measured using:

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

These metrics provide insight into prediction accuracy and overall model performance.

---

# ▶️ Running the Project

Run the notebook directly:

```bash
jupyter notebook
```

or

```bash
jupyter lab
```

Execute all notebook cells in order.

---

# 🔍 Example Prediction

### Input

| Feature | Value |
|---------|------:|
| Area | 7420 |
| Bedrooms | 4 |
| Bathrooms | 2 |
| Stories | 3 |
| Parking | 2 |
| Furnishing | Furnished |

### Output

```
Predicted House Price:

8,230,000
```

*(Example only. Your output depends on the trained model.)*

---

# 📈 Results

The trained regression model successfully captured the relationship between housing features and property prices.

### Baseline Model

- Linear Regression

### Future Improvements

- Random Forest Regressor
- XGBoost Regressor
- Gradient Boosting
- Hyperparameter Tuning
- Cross Validation
- Feature Selection

These improvements can further enhance prediction accuracy and model robustness.

---

# 📁 Project Structure

```
Housing-Price-Prediction/

│
├── Housing.csv
├── House_Price_Prediction.ipynb
├── README.md
├── requirements.txt
└── images/
```

---

# 🤝 Contributing

Contributions are welcome!

If you would like to improve this project:

1. Fork the repository.
2. Create a new branch.
3. Make your changes.
4. Commit your updates with clear commit messages.
5. Submit a Pull Request.

Please ensure your code is well-documented and follows Python best practices.

---

# 📜 License

This project is licensed under the MIT License.

You are free to use, modify, and distribute this project with proper attribution.

---

# 👨‍💻 Author
**Muhammad Waseem**

Aspiring AI & Data Science Engineer with a strong interest in Machine Learning, Data Analysis, and Artificial Intelligence.

I created this project to strengthen my understanding of the complete machine learning workflow—from exploring raw data to training predictive models. Every project is an opportunity to learn something new, and this repository reflects that journey.

**GitHub:** https://github.com/mmilyas245-dot


---

# 🙏 Acknowledgements

Thanks to the open-source Python community and the developers of Pandas, NumPy, Matplotlib, Seaborn, and Scikit-learn for providing the tools that made this project possible.

---

⭐ **If you found this project useful or learned something from it, consider giving the repository a star. Your support is always appreciated!**# house-price-prediction
Machine learning project to predict house prices using Python,Pandas,Numpy,scikit-learn,and Regression
