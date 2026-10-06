import pandas as pd
import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.naive_bayes import CategoricalNB

# Φόρτωση Δεδομένων


# Λήψη του συνόλου δεδομένων Mushroom από το UCI Machine Learning Repository
print("--- Φόρτωση Δεδομένων από UCI ---")
mushroom = fetch_ucirepo(id=73)

# Ανάκτηση των χαρακτηριστικών (X) και των στόχων/κλάσεων (y)
X_raw = mushroom.data.features
y_raw = mushroom.data.targets

# Συνένωση X και y σε ένα ενιαίο DataFrame για την ορθή αφαίρεση διπλότυπων 
df = pd.concat([y_raw, X_raw], axis=1)

print(f"Αρχικό μέγεθος συνόλου δεδομένων: {df.shape}")

# Προεπεξεργασία Δεδομένων (1ο Ερώτημα)
print("\n--- Έναρξη Προεπεξεργασίας ---")

# Αντικατάσταση του συμβόλου '?' με NaN
df.replace('?', np.nan, inplace=True)

# Συμπλήρωση ελλειπουσών τιμών με την πιο συχνή τιμή (Mode)
# Χρήση του SimpleImputer με 'most_frequent'
imputer = SimpleImputer(strategy='most_frequent')
df_imputed_array = imputer.fit_transform(df)

# Επαναδημιουργία DataFrame με τα ονόματα των στηλών, καθώς το imputer επιστρέφει numpy array
df_imputed = pd.DataFrame(df_imputed_array, columns=df.columns)

# Αφαίρεση διπλότυπων εγγραφών (Duplicates)
df_imputed.drop_duplicates(inplace=True)
print(f"Μέγεθος συνόλου δεδομένων μετά την αφαίρεση διπλών εγγραφών: {df_imputed.shape}")

# Διαχωρισμός Χαρακτηριστικών (X) και Κλάσης (y)
# Η 1η στήλη περιέχει την κλάση (poisonous/edible)
y = df_imputed.iloc[:, 0]
X = df_imputed.iloc[:, 1:]

# Κωδικοποίηση Κατηγορικών Δεδομένων (Label Encoding)
# Μετατροπή των nominal τιμών σε numeric για να είναι συμβατές με τους αλγορίθμους (KNN, MLP)

# Κωδικοποίηση της κλάσης y
le_y = LabelEncoder()
y = le_y.fit_transform(y)

# Κωδικοποίηση των χαρακτηριστικών X (κάθε στήλη ξεχωριστά)
X = X.apply(LabelEncoder().fit_transform)

print("Ολοκλήρωση Προεπεξεργασίας και Κωδικοποίησης.")

# Ορισμός Μοντέλων Ταξινόμησης

# Δημιουργία λίστας για την αποθήκευση των μοντέλων προς αξιολόγηση
models = []

# 1. Naïve Bayes
models.append(('Naïve Bayes', CategoricalNB()))

# 2. Multilayer Perceptron (MLP) με 1, 2, 3 κρυμμένα επίπεδα
# Ορίζεται max_iter για επάρκεια εκπαίδευσης
models.append(('MLP (1 hidden layer)', MLPClassifier(hidden_layer_sizes=(10,), max_iter=800, random_state=42)))
models.append(('MLP (2 hidden layers)', MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=800, random_state=42)))
models.append(('MLP (3 hidden layers)', MLPClassifier(hidden_layer_sizes=(10, 10, 10), max_iter=800, random_state=42)))

# 3. KNN για k=1, 2, 5, 9
for k in [1, 2, 5, 9]:
    models.append((f'KNN (k={k})', KNeighborsClassifier(n_neighbors=k)))

# Αξιολόγηση Μοντέλων (Split & Cross Validation)

print("\n--- Αποτελέσματα Ταξινόμησης ---")

for name, model in models:
    print(f"\n>> Αλγόριθμος: {name}")
    
    # Α. Percentage Split (66% Training - 34% Testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.34, random_state=42)
    
    # Εκπαίδευση του μοντέλου
    model.fit(X_train, y_train)
    
    # Πρόβλεψη στο σύνολο ελέγχου
    y_pred = model.predict(X_test)
    
    # Υπολογισμός Ακρίβειας (Accuracy)
    acc_split = accuracy_score(y_test, y_pred)
    
    # Δημιουργία Πίνακα Σύγχυσης (Confusion Matrix)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"   [Percentage Split 66%] Accuracy: {acc_split:.4f}")
    print(f"   Confusion Matrix:\n{cm}")
    
    
    # Β. 10-Fold Cross Validation
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    # Υπολογισμός σκορ για κάθε fold
    cv_scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    
    # Υπολογισμός μέσης τιμής ακρίβειας
    acc_cv = cv_scores.mean()
    
    print(f"   [10-Fold Cross Val]    Mean Accuracy: {acc_cv:.4f}")
    print("-" * 50)

print("\nΤέλος εκτέλεσης.")
