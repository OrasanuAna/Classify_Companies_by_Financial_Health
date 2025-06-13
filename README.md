# 🏢 Classify Companies by Financial Health

This project uses **machine learning** to classify companies as **financially healthy** or **risky** based on financial metrics such as revenue, profit, debt, and market capitalization. It showcases a practical application of supervised learning in the domain of financial risk assessment.

## 🧠 Objective

Automatically determine whether a company is **safe to invest** in or **risky**, based on selected financial indicators.

## 🚀 Features

* Load and preprocess company financial data
* Apply feature scaling and label generation logic
* Train a classification model (e.g., Random Forest or Logistic Regression)
* Evaluate model accuracy and visualize the classification results
* Plot safe vs. risky companies using a 2D scatter chart

## 🛠️ Technologies Used

* Python
* pandas
* scikit-learn
* matplotlib
* NumPy

## 📂 Project Structure

```
Classify_Companies_by_Financial_Health/
├── main.py                                            # Main script for classification
├── financial_data.csv                                 # Company data file
├── Problema6 - Classify Companies by Financial Health.jpg  # Output image (classification plot)
└── README.md
```

## 📊 Sample Output

![Classification Output](https://github.com/OrasanuAna/Classify_Companies_by_Financial_Health/blob/master/Problema6%20-%20Classify%20Companies%20by%20Financial%20Health.jpg)

*Figure: Companies classified as financially "Safe" or "Risky" based on performance metrics.*

## 🧪 Model Workflow

1. **Load the CSV** file containing financial metrics.
2. **Label companies** based on a rule or threshold (e.g., profit margin, debt ratio).
3. **Train a classifier** (e.g., Random Forest, SVM, Logistic Regression).
4. **Evaluate accuracy** on test data.
5. **Visualize results** in a scatter plot.

## ⚙️ How to Use

1. **Clone the Repository**

   ```bash
   git clone https://github.com/OrasanuAna/Classify_Companies_by_Financial_Health.git
   cd Classify_Companies_by_Financial_Health
   ```

2. **Install Dependencies**

   ```bash
   pip install pandas scikit-learn matplotlib numpy
   ```

3. **Run the Script**

   ```bash
   python main.py
   ```

4. **Output**

   * Prints model accuracy in the console
   * Displays a chart of companies labeled as "Safe" or "Risky"

