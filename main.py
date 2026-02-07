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
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# ---------------------------------------------------------
# Step 3: Enhanced Data Cleaning & Exploration
# ---------------------------------------------------------
def explore_data(df):
    """Exploratory data analysis"""
    print("=== Data Exploration ===")
    print(f"Number of rows: {df.shape[0]}")
    print(f"Number of columns: {df.shape[1]}")
    
    # Check data types
    print("\nData Types:")
    print(df.dtypes.value_counts())
    
    # Check for missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'missing_count': missing, 'missing_percentage': missing_pct})
    print("\nMissing Values (top 10):")
    print(missing_df[missing_df['missing_count'] > 0].head(10))
    
    # Basic statistics
    print("\nNumerical Statistics:")
    
    print(df[numeric_cols].describe().T[['mean', 'std', 'min', 'max']].head(10))

explore_data(df)

# 1. Drop duplicates
initial_rows = len(df)
df = df.drop_duplicates()
print(f"\nRemoved {initial_rows - len(df)} duplicate rows")

# 2. Force numeric types for key columns
cols_to_numeric = ['sale_price', 'bldgarea', 'yearbuilt', 'lotarea', 
                   'resarea', 'comarea', 'unitsres', 'unitstotal', 
                   'numfloors', 'latitude', 'longitude', 'building_age']

for col in cols_to_numeric:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Enhanced filtering
print("\n=== Data Filtering ===")

# Price filtering - remove extreme outliers
if 'sale_price' in df.columns:
    price_q1 = df['sale_price'].quantile(0.01)
    price_q99 = df['sale_price'].quantile(0.99)
    before = len(df)
    df = df[(df['sale_price'] >= price_q1) & (df['sale_price'] <= price_q99)]
    print(f"Filtered sale prices: removed {before - len(df)} extreme outliers")
    print(f"Price range after filtering: ${df['sale_price'].min():,.0f} - ${df['sale_price'].max():,.0f}")

# Area filtering
if 'bldgarea' in df.columns:
    before = len(df)
    df = df[(df['bldgarea'] > 100) & (df['bldgarea'] < 1000000)]  # Reasonable building area
    print(f"Filtered building area: removed {before - len(df)} unrealistic values")

# Year filtering
if 'yearbuilt' in df.columns:
    before = len(df)
    df = df[(df['yearbuilt'] >= 1800) & (df['yearbuilt'] <= 2024)]
    print(f"Filtered year built: removed {before - len(df)} unrealistic years")

# 4. Handle missing values strategically
print("\n=== Handling Missing Values ===")

# Track total missing values
total_missing_filled = 0
missing_details = []

for col in numeric_cols:
    if col in df.columns:
        missing_before = df[col].isnull().sum()
        if missing_before > 0:
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            total_missing_filled += missing_before
            missing_details.append((col, missing_before, median_val))
            print(f"  - {col}: {missing_before} values → median({median_val:.2f})")

print(f"\n✅ Total missing values filled: {total_missing_filled}")

# Also handle categorical columns if needed
categorical_cols = ['borough_y', 'bldgclass', 'landuse']
for col in categorical_cols:
    if col in df.columngit and df[col].isnull().sum() > 0:
        missing_count = df[col].isnull().sum()
        mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
        df[col].fillna(mode_val, inplace=True)
        print(f"  - {col}: {missing_count} values → mode('{mode_val}')")

# 5. Create new features for better modeling
print("\n=== Feature Engineering ===")

# Price per square foot
if all(col in df.columns for col in ['sale_price', 'bldgarea']):
    df['price_per_sqft'] = df['sale_price'] / df['bldgarea']
    # Remove infinite values
    df['price_per_sqft'] = df['price_per_sqft'].replace([np.inf, -np.inf], np.nan)
    df['price_per_sqft'].fillna(df['price_per_sqft'].median(), inplace=True)
    print(f"Created 'price_per_sqft': ${df['price_per_sqft'].median():.2f} median")

# Age categories
if 'building_age' in df.columns:
    df['age_category'] = pd.cut(df['building_age'], 
                                 bins=[0, 10, 25, 50, 100, 200, 500],
                                 labels=['New', 'Young', 'Middle', 'Old', 'Very Old', 'Historic'])
    print("Created 'age_category' feature")

# Borough dummies for categorical analysis
if 'borough_y' in df.columns:
    borough_dummies = pd.get_dummies(df['borough_y'], prefix='borough', drop_first=True)
    df = pd.concat([df, borough_dummies], axis=1)
    print(f"Added borough dummy variables: {list(borough_dummies.columns)}")

# 6. Drop rows with missing values in critical columns
critical_cols = ['sale_price', 'bldgarea', 'latitude', 'longitude']
existing_critical = [c for c in critical_cols if c in df.columns]
before = len(df)
df = df.dropna(subset=existing_critical)
print(f"\nRemoved {before - len(df)} rows with missing critical values")

print(f"\nFinal Cleaned Shape: {df.shape}")
print(f"Remaining rows: {len(df)}")

# ---------------------------------------------------------
# Step 4: Advanced Visualizations
# ---------------------------------------------------------
def create_visualizations(df):
    """Create comprehensive visualizations"""
    print("\n=== Creating Visualizations ===")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('NYC Housing Data Analysis', fontsize=16, fontweight='bold')
    
    # 1. Price distribution
    axes[0, 0].hist(df['sale_price'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title('Sale Price Distribution')
    axes[0, 0].set_xlabel('Sale Price ($)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].ticklabel_format(style='plain', axis='x')
    
    # 2. Price vs Building Area
    axes[0, 1].scatter(df['bldgarea'], df['sale_price'], alpha=0.5, s=10)
    axes[0, 1].set_title('Building Area vs Sale Price')
    axes[0, 1].set_xlabel('Building Area (sq ft)')
    axes[0, 1].set_ylabel('Sale Price ($)')
    axes[0, 1].ticklabel_format(style='plain', axis='both')
    
    # 3. Price per sqft by borough
    if 'borough_y' in df.columns and 'price_per_sqft' in df.columns:
        borough_avg = df.groupby('borough_y')['price_per_sqft'].mean().sort_values()
        axes[0, 2].bar(borough_avg.index, borough_avg.values)
        axes[0, 2].set_title('Average Price per SqFt by Borough')
        axes[0, 2].set_xlabel('Borough')
        axes[0, 2].set_ylabel('Price per SqFt ($)')
        axes[0, 2].tick_params(axis='x', rotation=45)
    
    # 4. Building age distribution
    if 'building_age' in df.columns:
        axes[1, 0].hist(df['building_age'], bins=30, edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('Building Age Distribution')
        axes[1, 0].set_xlabel('Building Age (years)')
        axes[1, 0].set_ylabel('Frequency')
    
    # 5. Correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 1:
        corr_matrix = numeric_df.corr()
        im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto')
        axes[1, 1].set_title('Correlation Heatmap')
        axes[1, 1].set_xticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_yticks(range(len(corr_matrix.columns)))
        axes[1, 1].set_xticklabels(corr_matrix.columns, rotation=90, fontsize=8)
        axes[1, 1].set_yticklabels(corr_matrix.columns, fontsize=8)
        plt.colorbar(im, ax=axes[1, 1])
    
    # 6. Price by year built
    if 'yearbuilt' in df.columns:
        year_price = df.groupby(df['yearbuilt'] // 10 * 10)['sale_price'].mean()
        axes[1, 2].plot(year_price.index, year_price.values, marker='o')
        axes[1, 2].set_title('Average Price by Decade Built')
        axes[1, 2].set_xlabel('Decade Built')
        axes[1, 2].set_ylabel('Average Sale Price ($)')
        axes[1, 2].ticklabel_format(style='plain', axis='y')
    
    plt.tight_layout()
    plt.savefig('nyc_housing_analysis.png', dpi=150, bbox_inches='tight')
    print("Saved visualization: 'nyc_housing_analysis.png'")

create_visualizations(df)

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