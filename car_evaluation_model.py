import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Φόρτωση Δεδομένων (Car Evaluation)
print("---  Φόρτωση και Προεπεξεργασία Car Evaluation Dataset ---")

# Car Evaluation Database
car_evaluation = fetch_ucirepo(id=19)

X_raw = car_evaluation.data.features
y_raw = car_evaluation.data.targets

# Συνένωση σε ένα DataFrame για την επεξεργασία
df = pd.concat([X_raw, y_raw], axis=1)

print(f"Αρχικό μέγεθος dataset: {df.shape}")
print("Κλάσεις:", df['class'].unique()) 

# Προεπεξεργασία 
# Τυχαιοποίηση Δεδομένων
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Μετατροπή Nominal -> Numeric
y = df['class']
X = df.drop('class', axis=1)

# Κωδικοποίηση  y  / LabelEncoder (0, 1, 2, 3)
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Αντί για Label Encoding στο X, κάνουμε One-Hot Encoding.
# βοηθάει τον LDA.
X = pd.get_dummies(X, drop_first=True)

# Ταξινόμηση (Αλγόριθμοι)

print("\n--- Έναρξη Ταξινόμησης ---")

models = []

# 1) Linear Discriminant Analysis (LDA)
models.append(('LDA', LinearDiscriminantAnalysis()))

# 2) One-R 
# Στο sklearn δεν υπάρχει "OneR". Το OneR είναι ισοδύναμο με ένα Δέντρο Απόφασης
# βάθους 1 (Decision Stump), που παίρνει απόφαση βάσει ενός μόνο χαρακτηριστικού.
models.append(('One-R (Decision Stump)', DecisionTreeClassifier(max_depth=1, criterion='entropy')))

# 3) C4.5 
# Ο C4.5 είναι αλγόριθμος δέντρου που χρησιμοποιεί Entropy/Information Gain.
# Στο sklearn χρησιμοποιούμε DecisionTreeClassifier με criterion='entropy'.
models.append(('C4.5 (Decision Tree)', DecisionTreeClassifier(criterion='entropy', random_state=42)))

# Αξιολόγηση (Split & Cross Validation)

for name, model in models:
    print(f"\n>> Αλγόριθμος: {name}")
    
    # Percentage Split (66% Train - 34% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc_split = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   [Percentage Split 66%] Accuracy: {acc_split:.4f}")
    print(f"   Confusion Matrix:\n{cm}")
    
    # 10-Fold Cross Validation 
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    print(f"   [10-Fold Cross Val]    Mean Accuracy: {cv_scores.mean():.4f}")

print("\nΤέλος Άσκησης 3.")
