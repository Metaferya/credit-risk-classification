# **Credit Risk Analysis Report**  

## **Overview of the Analysis**  

In this analysis, we used a machine learning approach to assess credit risk based on historical lending data. The goal was to build a predictive model that determines whether a loan is **high risk** (1) or **healthy** (0). A logistic regression model was employed to classify loan statuses using financial features.

The dataset contained **financial information** regarding various loans, and our objective was to **predict loan status** based on these attributes. The key variable we were trying to predict was the `loan_status`, which had two possible values:
- **0 (Healthy Loan):** The loan is expected to be repaid successfully.
- **1 (High-Risk Loan):** The loan has a higher chance of defaulting.

### **Data Overview**  
To understand the distribution of the target variable, we examined the value counts:

```python
y.value_counts()
```

The dataset was **highly imbalanced**, with far more healthy loans (0) than high-risk loans (1), which could affect model performance.

### **Machine Learning Process**  
The following stages were followed in the analysis:

1. **Data Preprocessing:**  
   - Read the dataset and separated the target variable (`loan_status`) from the feature set.
   - Split the dataset into **training (80%)** and **testing (20%)** subsets using `train_test_split`.

2. **Model Selection:**  
   - Used **Logistic Regression** as the machine learning model to predict credit risk.

3. **Model Training & Prediction:**  
   - Trained the `LogisticRegression` model on the training data.
   - Used the trained model to make predictions on the test set.

4. **Model Evaluation:**  
   - Evaluated performance using **confusion matrix** and **classification report** (precision, recall, and accuracy).

---

## **Results**  

- **Machine Learning Model 1: Logistic Regression**  
    - **Accuracy:** **99%**  
    - **Precision and Recall Scores:**  
      - **Healthy Loan (0):**  
        - **Precision:** **1.00** (Very high precision, meaning almost all predicted 0’s were correct)  
        - **Recall:** **0.99** (Most actual 0’s were correctly identified)  
      - **High-Risk Loan (1):**  
        - **Precision:** **0.86** (Some high-risk loans were misclassified)  
        - **Recall:** **0.94** (Most high-risk loans were correctly identified)  
    - **Confusion Matrix:**  
      ```
      [[14924    77]
       [   31   476]]
      ```  
      - **False Positives (Type I Error):** **77**  
      - **False Negatives (Type II Error):** **31**  
      - Most of the misclassifications were false positives (loans predicted as high-risk but were actually healthy).  

---

## **Summary and Recommendation**  

### **Which model performed best?**  
- The logistic regression model performed **very well**, achieving **99% accuracy**.  
- It had a **high precision** for healthy loans (`0`), meaning almost no incorrect high-risk predictions.  
- For high-risk loans (`1`), recall was **94%**, meaning it correctly identified most actual high-risk loans.  

### **Considerations for Decision-Making**  
- If **avoiding high-risk loans is critical**, the model is quite effective because it correctly identifies most high-risk loans (high recall of 94%).  
- However, there were **77 false positives**, meaning some loans that were actually healthy were incorrectly classified as high-risk, which could lead to **missed lending opportunities**.  

### **Final Recommendation**  
- This logistic regression model is **suitable** for **minimizing high-risk loans** while still allowing loan approvals.  
- If the primary goal is to **minimize false positives** (reduce incorrectly labeled high-risk loans), further improvements (e.g., resampling techniques or alternative models like Random Forest) may be beneficial.  
- If the goal is **ensuring no high-risk loans get approved**, this model is a **strong candidate** given its high recall.  