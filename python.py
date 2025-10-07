# Analyzing Data with Pandas and Visualizing Results with Matplotlib
# Dataset: Iris Dataset

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

print("="*70)
print("IRIS DATASET ANALYSIS")
print("="*70)

# ============================================================================
# TASK 1: LOAD AND EXPLORE THE DATASET
# ============================================================================

print("\n" + "="*70)
print("TASK 1: LOAD AND EXPLORE THE DATASET")
print("="*70)

try:
    # Load the Iris dataset
    iris_data = load_iris()
    
    # Create a DataFrame
    df = pd.DataFrame(
        data=iris_data.data,
        columns=iris_data.feature_names
    )
    
    # Add the target variable (species)
    df['species'] = iris_data.target
    df['species_name'] = df['species'].map({
        0: 'setosa',
        1: 'versicolor',
        2: 'virginica'
    })
    
    print("\n✓ Dataset loaded successfully!")
    print(f"\nDataset shape: {df.shape} (rows, columns)")
    
    # Display first few rows
    print("\n1. First 5 rows of the dataset:")
    print("-" * 70)
    print(df.head())
    
    # Explore the structure
    print("\n2. Dataset Information:")
    print("-" * 70)
    print(f"\nColumn Names and Data Types:")
    print(df.dtypes)
    
    print(f"\nDataset Dimensions: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # Check for missing values
    print("\n3. Missing Values Check:")
    print("-" * 70)
    missing_values = df.isnull().sum()
    print(missing_values)
    
    if missing_values.sum() == 0:
        print("\n✓ No missing values found in the dataset!")
    else:
        print("\n⚠ Missing values detected. Cleaning in progress...")
        # Option 1: Drop rows with missing values
        # df_cleaned = df.dropna()
        
        # Option 2: Fill missing values with mean (for numerical columns)
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col].fillna(df[col].mean(), inplace=True)
        print("✓ Missing values handled successfully!")
    
    print("\n4. Dataset Summary:")
    print("-" * 70)
    print(f"Total number of samples: {len(df)}")
    print(f"\nSpecies distribution:")
    print(df['species_name'].value_counts())
    
except FileNotFoundError:
    print("Error: Dataset file not found!")
except Exception as e:
    print(f"An error occurred: {e}")

# ============================================================================
# TASK 2: BASIC DATA ANALYSIS
# ============================================================================

print("\n\n" + "="*70)
print("TASK 2: BASIC DATA ANALYSIS")
print("="*70)

# 1. Compute basic statistics
print("\n1. Basic Statistical Summary:")
print("-" * 70)
print(df.describe())

# 2. Group by species and compute mean
print("\n2. Mean Values by Species:")
print("-" * 70)
species_means = df.groupby('species_name')[iris_data.feature_names].mean()
print(species_means)

# 3. Additional analysis
print("\n3. Key Findings and Patterns:")
print("-" * 70)

# Correlation analysis
print("\nCorrelation Matrix:")
correlation_matrix = df[iris_data.feature_names].corr()
print(correlation_matrix)

# Key insights
print("\n📊 INSIGHTS:")
print("-" * 70)
print("• Petal length and petal width show strong positive correlation (0.96)")
print("• Sepal length and petal length are also highly correlated (0.87)")
print("• Virginica species has the largest average measurements")
print("• Setosa species has the smallest petal dimensions")
print("• All features show distinct differences across species")

# Find max and min values
print("\n• Maximum sepal length:", df['sepal length (cm)'].max(), "cm")
print("• Minimum sepal length:", df['sepal length (cm)'].min(), "cm")
print("• Maximum petal length:", df['petal length (cm)'].max(), "cm")
print("• Minimum petal length:", df['petal length (cm)'].min(), "cm")

# ============================================================================
# TASK 3: DATA VISUALIZATION
# ============================================================================

print("\n\n" + "="*70)
print("TASK 3: DATA VISUALIZATION")
print("="*70)
print("\nGenerating visualizations...")

# Create a figure with subplots
fig = plt.figure(figsize=(16, 12))

# -------------------------
# 1. LINE CHART - Trends
# -------------------------
plt.subplot(2, 2, 1)

# Create a trend line showing average measurements over sample index
# This simulates a time-series view
for feature in ['sepal length (cm)', 'petal length (cm)']:
    rolling_mean = df[feature].rolling(window=10).mean()
    plt.plot(rolling_mean, label=feature, linewidth=2)

plt.title('Rolling Average of Flower Measurements', fontsize=14, fontweight='bold')
plt.xlabel('Sample Index', fontsize=11)
plt.ylabel('Measurement (cm)', fontsize=11)
plt.legend(loc='best')
plt.grid(True, alpha=0.3)

# -------------------------
# 2. BAR CHART - Comparison
# -------------------------
plt.subplot(2, 2, 2)

# Average petal length by species
avg_petal_length = df.groupby('species_name')['petal length (cm)'].mean()
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = plt.bar(avg_petal_length.index, avg_petal_length.values, color=colors, edgecolor='black', linewidth=1.5)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.2f} cm',
             ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title('Average Petal Length by Species', fontsize=14, fontweight='bold')
plt.xlabel('Species', fontsize=11)
plt.ylabel('Average Petal Length (cm)', fontsize=11)
plt.xticks(rotation=0)
plt.grid(axis='y', alpha=0.3)

# -------------------------
# 3. HISTOGRAM - Distribution
# -------------------------
plt.subplot(2, 2, 3)

plt.hist(df['sepal width (cm)'], bins=20, color='#95E1D3', edgecolor='black', alpha=0.7)
plt.axvline(df['sepal width (cm)'].mean(), color='red', linestyle='--', 
            linewidth=2, label=f"Mean: {df['sepal width (cm)'].mean():.2f} cm")
plt.axvline(df['sepal width (cm)'].median(), color='blue', linestyle='--', 
            linewidth=2, label=f"Median: {df['sepal width (cm)'].median():.2f} cm")

plt.title('Distribution of Sepal Width', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Width (cm)', fontsize=11)
plt.ylabel('Frequency', fontsize=11)
plt.legend()
plt.grid(axis='y', alpha=0.3)

# -------------------------
# 4. SCATTER PLOT - Relationship
# -------------------------
plt.subplot(2, 2, 4)

species_colors = {'setosa': '#FF6B6B', 'versicolor': '#4ECDC4', 'virginica': '#45B7D1'}
for species in df['species_name'].unique():
    species_data = df[df['species_name'] == species]
    plt.scatter(species_data['sepal length (cm)'], 
                species_data['petal length (cm)'],
                label=species.capitalize(),
                color=species_colors[species],
                s=80,
                alpha=0.6,
                edgecolors='black',
                linewidth=0.5)

plt.title('Sepal Length vs Petal Length by Species', fontsize=14, fontweight='bold')
plt.xlabel('Sepal Length (cm)', fontsize=11)
plt.ylabel('Petal Length (cm)', fontsize=11)
plt.legend(title='Species', loc='best')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_analysis_plots.png', dpi=300, bbox_inches='tight')
print("\n✓ All visualizations created successfully!")
print("✓ Plots saved as 'iris_analysis_plots.png'")
plt.show()

# ============================================================================
# ADDITIONAL VISUALIZATIONS (BONUS)
# ============================================================================

print("\n\n" + "="*70)
print("BONUS: ADDITIONAL VISUALIZATIONS")
print("="*70)

# Create additional visualization: Correlation Heatmap
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Heatmap
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=axes[0])
axes[0].set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold')

# Box plot for all features by species
df_melted = df.melt(id_vars=['species_name'], 
                     value_vars=iris_data.feature_names,
                     var_name='Feature', value_name='Value')
sns.boxplot(data=df_melted, x='Feature', y='Value', hue='species_name', ax=axes[1])
axes[1].set_title('Feature Distributions by Species', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Feature', fontsize=11)
axes[1].set_ylabel('Measurement (cm)', fontsize=11)
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(title='Species', loc='upper left')

plt.tight_layout()
plt.savefig('iris_additional_plots.png', dpi=300, bbox_inches='tight')
print("\n✓ Additional visualizations created!")
print("✓ Plots saved as 'iris_additional_plots.png'")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n\n" + "="*70)
print("ANALYSIS COMPLETE - FINAL SUMMARY")
print("="*70)

print("""
📋 SUMMARY OF FINDINGS:

1. DATASET OVERVIEW:
   • 150 samples with 4 numerical features and 3 species classes
   • No missing values - dataset is clean and ready for analysis
   • Balanced distribution: 50 samples per species

2. STATISTICAL INSIGHTS:
   • Strong correlation between petal dimensions (r=0.96)
   • Clear separation between species based on measurements
   • Virginica has the largest measurements across all features
   • Setosa is distinctly smaller, especially in petal dimensions

3. VISUAL INSIGHTS:
   • Line chart shows measurement trends across samples
   • Bar chart reveals significant differences in petal length by species
   • Histogram shows normal distribution of sepal width
   • Scatter plot demonstrates clear species clustering

4. KEY PATTERNS:
   • Petal measurements are more discriminative than sepal measurements
   • Species can be effectively distinguished using combinations of features
   • Linear relationships exist between several feature pairs

5. RECOMMENDATIONS:
   • This dataset is ideal for classification tasks
   • Petal length and width should be prioritized as features
   • The clear separation makes this suitable for beginner ML projects
""")

print("="*70)
print("Thank you for reviewing this analysis!")
print("="*70)