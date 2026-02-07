import sys
import subprocess
import os

# ---------------------------------------------------------
# Step 0: Auto-Install Dependencies
# ---------------------------------------------------------
def install(package):
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    except subprocess.CalledProcessError as e:
        print(f"Error installing {package}: {e}")

required_packages = ["scikit-learn", "pandas", "matplotlib", "numpy"]

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"Installing {package}...")
        install(package)

# ---------------------------------------------------------
# Step 1: Imports
# ---------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------
# Step 2: Load Local Data
# ---------------------------------------------------------
workspace_folder = os.getcwd()
csv_path = os.path.join(workspace_folder, "nyc_housing_base.csv")

if not os.path.exists(csv_path):
    print(f"ERROR: Could not find {csv_path}")
    sys.exit(1)

print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path, encoding="latin1", engine="python")
print(f"Original Shape: {df.shape}")

# ---------------------------------------------------------
# Step 3: Robust Cleaning
# ---------------------------------------------------------
# 1. Drop duplicates
df = df.drop_duplicates()

# 2. Force numeric types (Variable defined here as 'cols_to_numeric')
cols_to_numeric = ['sale_price', 'bldgarea', 'yearbuilt']

for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Filter out "Impossible" Data
if 'sale_price' in df.columns:
    df = df[df['sale_price'] > 10000]
if 'bldgarea' in df.columns:
    df = df[df['bldgarea'] > 0]
if 'yearbuilt' in df.columns:
    df = df[df['yearbuilt'] > 1800]

# 4. Drop rows with missing values in critical columns
# FIX: Use 'cols_to_numeric' here instead of 'cols_to_fix'
critical_cols = cols_to_numeric + ['latitude', 'longitude']
existing_cols = [c for c in critical_cols if c in df.columns]
df = df.dropna(subset=existing_cols)

# Clear all values that are null
df.dropna(inplace = True)

# Filter outliers with sales price (family transfers/inheritances)
for x in df.index:
  if df.loc[x, "sale_price"] <= 10:
    df.drop(x, inplace = True)


print(f"Cleaned Shape: {df.shape}")


# ---------------------------------------------------------
# Step 4: Run Linear Regression
# ---------------------------------------------------------
if not df.empty:
    X = df[['bldgarea']]
    y = df['sale_price']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n--- Model Results ---")
    print("Model trained successfully!")
    print(f"Coefficient (Price per sqft): ${model.coef_[0]:.2f}")
    
    # Visualization
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='black', alpha=0.5, label='Actual Sales')
    plt.plot(X_test, model.predict(X_test), color='blue', linewidth=3, label='Regression Line')
    plt.title('Building Area vs Sale Price')
    plt.xlabel('Building Area (sq ft)')
    plt.ylabel('Sale Price ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    output_img = "housing_regression.png"
    plt.savefig(output_img)
    print(f"Plot saved to '{output_img}'")
    
else:
    print("Error: Dataset is empty after cleaning.")