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

required_packages = ["scikit-learn", "pandas", "matplotlib", "numpy", "seaborn"]

print("Checking environment...")
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
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------------------------------
# Step 2: Load Data
# ---------------------------------------------------------
workspace_folder = os.getcwd()
csv_path = os.path.join(workspace_folder, "nyc_housing_base.csv")

if not os.path.exists(csv_path):
    print(f"ERROR: Could not find {csv_path}")
    sys.exit(1)

print(f"Loading data from: {csv_path}")
df = pd.read_csv(csv_path, encoding="latin1", engine="python")

# ---------------------------------------------------------
# Step 3: Robust Cleaning
# ---------------------------------------------------------
# 1. Drop duplicates
df = df.drop_duplicates()

# 2. Force numeric types
cols_to_numeric = ['sale_price', 'bldgarea', 'yearbuilt', 'unitsres', 'latitude', 'longitude']
for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. CRITICAL FILTER: Single Family Homes Only
# This fixes the "Weird Graph" issue (Apartments vs Whole Buildings)
if 'unitsres' in df.columns:
    df = df[df['unitsres'] == 1]

# 4. Logic Filters (Remove Garbage Data)
if 'sale_price' in df.columns:
    df = df[df['sale_price'] > 10000] # Remove transfers < $10k
if 'bldgarea' in df.columns:
    df = df[df['bldgarea'] > 0]       # Remove 0 size
if 'yearbuilt' in df.columns:
    df = df[df['yearbuilt'] > 1800]   # Remove Year 0

# 5. Drop NaNs
df = df.dropna(subset=cols_to_numeric + ['borough_x'])

print(f"Cleaned Data Shape: {df.shape}")

# ---------------------------------------------------------
# Step 4: Feature Engineering (Making the Model Smart)
# ---------------------------------------------------------
# 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island
df['borough_x'] = df['borough_x'].astype(int).astype(str)

# One-Hot Encoding: Turn "Borough" into columns (Is_Manhattan, Is_Bronx...)
df_model = pd.get_dummies(df, columns=['borough_x'], drop_first=True)

# Select Features: Size + Age + Location
features = ['bldgarea', 'yearbuilt'] + [c for c in df_model.columns if 'borough_x_' in c]
X = df_model[features]
y = df_model['sale_price']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------------------------------------
# Step 5: Train Gradient Boosting
# ---------------------------------------------------------
if not df.empty:
    print("\nTraining Gradient Boosting Model (this may take a moment)...")
    gb_model = GradientBoostingRegressor(
        n_estimators=200,  # More trees = smarter
        learning_rate=0.05,
        max_depth=5,       # Deeper trees
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    # Predictions
    y_pred = gb_model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Accuracy (R2 Score): {r2:.4f} (1.0 is perfect)")
    print(f"Average Error: ${mae:,.0f}")

    # ---------------------------------------------------------
    # Step 6: Find "Steals" & Save Results
    # ---------------------------------------------------------
    results = pd.DataFrame({
        'Actual_Price': y_test,
        'Predicted_Value': y_pred,
        'Bldg_Area': X_test['bldgarea'],
        'Year_Built': X_test['yearbuilt']
    })
    
    # "Potential Profit" = Predicted Value - Actual Price
    results['Potential_Profit'] = results['Predicted_Value'] - results['Actual_Price']
    
    # Sort by biggest "Deals"
    steals = results.sort_values(by='Potential_Profit', ascending=False).head(5)
    
    print("\n--- TOP 5 UNDEVALUED PROPERTIES (POTENTIAL STEALS) ---")
    print(steals[['Actual_Price', 'Predicted_Value', 'Potential_Profit']])

    # ---------------------------------------------------------
    # Step 7: Visualizations
    # ---------------------------------------------------------
    
    # Graph 1: Actual vs Predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5, color='purple', s=10)
    p_min, p_max = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    plt.plot([p_min, p_max], [p_min, p_max], 'k--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Price ($)')
    plt.ylabel('Predicted Price ($)')
    plt.title('Accuracy Check: Actual vs Predicted')
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True, alpha=0.3)
    plt.savefig("1_accuracy_check.png")
    print("\nSaved '1_accuracy_check.png'")

    # Graph 2: Feature Importance (What matters?)
    importance = gb_model.feature_importances_
    feat_names = X_train.columns
    sorted_idx = np.argsort(importance)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_idx)), importance[sorted_idx], align='center', color='teal')
    plt.yticks(range(len(sorted_idx)), feat_names[sorted_idx])
    plt.xlabel("Importance Score")
    plt.title("What Drives House Prices in NYC?")
    plt.tight_layout()
    plt.savefig("2_feature_importance.png")
    print("Saved '2_feature_importance.png'")

    # Graph 3: Geospatial Heatmap
    plt.figure(figsize=(10, 10))
    sc = plt.scatter(df['longitude'], df['latitude'], 
                     c=df['sale_price'], cmap='coolwarm', 
                     s=10, alpha=0.7, 
                     vmax=df['sale_price'].quantile(0.95)) # Cap at 95% to handle outliers
    plt.colorbar(sc, label='Sale Price ($)')
    plt.title("NYC Housing Price Map")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, alpha=0.3)
    plt.savefig("3_price_map.png")
    print("Saved '3_price_map.png'")

    # ---------------------------------------------------------
    # Step 8: Interactive Price Predictor Tool
    # ---------------------------------------------------------
    def predict_my_house(sqft, year, borough_code):
        """
        Predicts price for a custom house.
        borough_code: 1=Manhattan, 2=Bronx, 3=Brooklyn, 4=Queens, 5=Staten Island
        """
        # Create a dummy row with all zeros
        input_data = pd.DataFrame(columns=X_train.columns)
        input_data.loc[0] = 0 
        
        # Fill in user stats
        input_data['bldgarea'] = sqft
        input_data['yearbuilt'] = year
        
        # Handle Borough One-Hot Encoding
        # Manhattan (1) is the baseline (all 0s). 
        # For others, we set the specific column to 1.
        col_name = f"borough_x_{borough_code}"
        if col_name in input_data.columns:
            input_data[col_name] = 1
            
        pred = gb_model.predict(input_data)[0]
        
        boro_names = {1: "Manhattan", 2: "Bronx", 3: "Brooklyn", 4: "Queens", 5: "Staten Island"}
        b_name = boro_names.get(borough_code, "Unknown")
        
        print(f"\n--- CUSTOM PREDICTION ---")
        print(f"House: {sqft} sqft | Year: {year} | Location: {b_name}")
        print(f"Estimated Market Value: ${pred:,.2f}")

    # --- RUN A TEST PREDICTION ---
    # Example: 2,500 sqft house in Brooklyn (3), built in 1940
    predict_my_house(sqft=2500, year=1940, borough_code=3)

else:
    print("Error: Dataset empty after cleaning.")