import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
#Αντώνης Τσίγγερης 2026

# Φόρτωση και Προεπεξεργασία
print("--- 1. Φόρτωση και Προεπεξεργασία (Energy Efficiency) ---")

# Λήψη δεδομένων
dataset = fetch_ucirepo(id=242)
X_raw = dataset.data.features
y_raw = dataset.data.targets

# Συνένωση
df = pd.concat([X_raw, y_raw], axis=1)

# Αφαίρεση Υ1 
if 'Y1' in df.columns:
    df.drop('Y1', axis=1, inplace=True)

# Μετατροπή Υ2 σε 3 κατηγορίες (Binning) 
# Δημιουργία 3 κλάσεων: Low, Medium, High
df['class'] = pd.cut(df['Y2'], bins=3, labels=['Low', 'Medium', 'High'])
# Αφαίρεση της αρχικής numeric στήλης Y2
df.drop('Y2', axis=1, inplace=True)

# Resampling στο 20% με Stratified Sampling 
X_temp = df.drop('class', axis=1)
y_temp = df['class']

# train_size=0.20 -> Κρατάμε το 20% του δείγματος
X_final, _, y_final, _ = train_test_split(
    X_temp, y_temp, train_size=0.20, stratify=y_temp, random_state=42
)

print(f"   Αρχικά δείγματα: {len(df)}")
print(f"   Δείγματα μετά το Resampling (20%): {len(X_final)}")

# Τελική μορφή X και y για τους αλγορίθμους
# Encoding του στόχου y (Low=0, Medium=1, High=2)
le = LabelEncoder()
y = le.fit_transform(y_final)

# Scaling των χαρακτηριστικών X (για SVM)
scaler = StandardScaler()
X = scaler.fit_transform(X_final)


# Ταξινόμηση

print("\n--- 2. Αποτελέσματα Ταξινόμησης ---")

# Λίστα Μοντέλων
models = []
# 1) SVM
models.append(('SVM', SVC(kernel='rbf', random_state=42)))
# 2) Random Tree (Decision Tree)
models.append(('Random Tree', DecisionTreeClassifier(random_state=42)))
# 3) Random Forest
models.append(('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42)))

for name, model in models:
    print(f"\n>> Αλγόριθμος: {name}")
    
    # Α. Percentage Split (50%) 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=42, stratify=y)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc_split = accuracy_score(y_test, y_pred)
    
    print(f"   [Split 50%] Accuracy: {acc_split:.4f}")
    print(confusion_matrix(y_test, y_pred))
    
    # Β. 10-Fold Cross Validation 
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"   [10-Fold CV] Mean Accuracy: {cv_scores.mean():.4f}")
    
    # Γ. Leave One Out Cross Validation (LOOCV) 
    loo = LeaveOneOut()
    #    Το cross_val_score με LOO επιστρέφει 1 για σωστό, 0 για λάθος σε κάθε δείγμα.
    #    Ο μέσος όρος αυτών είναι το Accuracy.
    loo_scores = cross_val_score(model, X, y, cv=loo, scoring='accuracy')
    print(f"   [LOOCV]      Mean Accuracy: {loo_scores.mean():.4f}")

print("\nΤέλος Εκτέλεσης.")
