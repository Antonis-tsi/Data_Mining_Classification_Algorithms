# Machine Learning Classification 

This repository contains a collection of Python scripts implementing various Machine Learning classification algorithms. The project focuses on end-to-end data pipelines, including data preprocessing, feature engineering, and the application of different models on real-world datasets from the UCI Machine Learning Repository. Performance is rigorously evaluated using techniques such as Percentage Split, k-Fold Cross Validation, and Leave-One-Out Cross Validation (LOOCV).

## Projects Included

### 1. Mushroom Edibility Classification
*   **Goal:** Classify mushrooms as edible or poisonous based on nominal physical characteristics.
*   **Algorithms Used:** Naïve Bayes, Multilayer Perceptron (MLP - 1, 2, and 3 hidden layers), k-Nearest Neighbors (KNN for k=1, 2, 5, 9).
*   **Preprocessing:** Missing value imputation (mode), duplicate removal, Label Encoding.
*   **Script:** `mushroom_classification.py`

### 2. Building Energy Efficiency Prediction
*   **Goal:** Predict the cooling load (Y2) of buildings by grouping continuous values into three discrete classes (Low, Medium, High).
*   **Algorithms Used:** Support Vector Machines (SVM with RBF kernel), Decision Tree, Random Forest.
*   **Preprocessing:** Dimensionality reduction (removed Y1 feature), discretization (binning), stratified resampling to 20%, Standard Scaling.
*   **Script:** `energy_efficiency_model.py`

### 3. Car Evaluation Modeling
*   **Goal:** Evaluate cars based on structure, price, and technical specifications into four categories (unacc, acc, good, vgood).
*   **Algorithms Used:** Linear Discriminant Analysis (LDA), One-R (Decision Stump), C4.5 (Decision Tree).
*   **Preprocessing:** Data shuffling, Label Encoding for the target variable, One-Hot Encoding for input features to optimize LDA performance.
*   **Script:** `car_evaluation_model.py`

## Installation and Execution

1. Clone the repository:
   ```bash
   git clone https://github.com/Antonis-tsi/Data_Mining_Classification_Algorithms.git
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the scripts individually:
   ```bash
   python mushroom_classification.py
   python energy_efficiency_model.py
   python car_evaluation_model.py
   ```

*Note: The datasets are automatically fetched via the `ucimlrepo` library during execution.*


## Author
**Antonis Tsingeris**
