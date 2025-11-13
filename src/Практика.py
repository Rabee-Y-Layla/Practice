

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import catboost as cb
from itertools import product, islice
import warnings, gc
warnings.filterwarnings('ignore')


df = pd.read_csv('Final1.csv')
print(" Original data shape:", df.shape)

def preprocess_data(df):
    df_clean = df.copy()

    def handle_range_values(value):
        if isinstance(value, str) and '-' in value:
            try:
                parts = list(map(float, value.split('-')))
                return np.mean(parts)
            except:
                return np.nan
        else:
            try:
                return float(value)
            except:
                return value

    numeric_columns = ['temperature', 'humidity', 'wind_speed']
    for col in numeric_columns:
        df_clean[col] = df_clean[col].apply(handle_range_values)

    numeric_columns_to_fill = [
        'flight_height', 'flight_speed', 'spray_volume',
        'temperature', 'humidity', 'wind_speed'
    ]
    for col in numeric_columns_to_fill:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())

    categorical_columns = [
        'crop_type', 'uav_model', 'genotype',
        'atomization_diameter', 'canopy_position', 'study'
    ]
    label_encoders = {}

    for col in categorical_columns:
        if col in df_clean.columns:
            le = LabelEncoder()
            df_clean[col] = df_clean[col].fillna('unknown')
            df_clean[col] = le.fit_transform(df_clean[col].astype(str))
            label_encoders[col] = le

    return df_clean, label_encoders

df_clean, label_encoders = preprocess_data(df)
print(f"✅ Data cleaned. Final shape: {df_clean.shape}")

######################################
# 2. Data Augmentation
###################################################3
def augment_dataset(df, augmentation_factor=3):
    augmented_data = []
    for _, row in df.iterrows():
        for _ in range(augmentation_factor):
            new_row = row.copy()
            if pd.notna(new_row['flight_height']):
                new_row['flight_height'] = max(1, min(6, new_row['flight_height'] + np.random.normal(0, 0.2)))
            if pd.notna(new_row['flight_speed']):
                new_row['flight_speed'] = max(1, min(25, new_row['flight_speed'] + np.random.normal(0, 0.5)))
            if pd.notna(new_row['spray_volume']):
                new_row['spray_volume'] = max(5, min(200, new_row['spray_volume'] + np.random.normal(0, 3)))
            if pd.notna(new_row['temperature']):
                new_row['temperature'] = max(10, min(40, new_row['temperature'] + np.random.normal(0, 1.5)))
            if pd.notna(new_row['humidity']):
                new_row['humidity'] = max(30, min(90, new_row['humidity'] + np.random.normal(0, 4)))
            if pd.notna(new_row['wind_speed']):
                new_row['wind_speed'] = max(0, min(8, new_row['wind_speed'] + np.random.normal(0, 0.4)))
            if pd.notna(new_row['coverage']):
                new_row['coverage'] = max(0, min(100, new_row['coverage'] + np.random.normal(0, new_row['coverage'] * 0.08)))
            if pd.notna(new_row['droplet_size']):
                new_row['droplet_size'] = max(50, new_row['droplet_size'] + np.random.normal(0, 12))
            if pd.notna(new_row['droplet_density']):
                new_row['droplet_density'] = max(0, new_row['droplet_density'] + np.random.normal(0, 4))
            augmented_data.append(new_row)
    return pd.DataFrame(augmented_data)

print(" Augmenting data...")
df_augmented = augment_dataset(df_clean, augmentation_factor=3)
print(f" Data after augmentation: {df_augmented.shape}")

###################################################################
# 3. Prepare Training Sets
#############################################################33
def prepare_training_data(df):
    feature_columns = [
        'crop_type', 'uav_model', 'flight_height', 'flight_speed',
        'spray_volume', 'atomization_diameter', 'temperature',
        'humidity', 'wind_speed'
    ]

    coverage_data = df.dropna(subset=['coverage']).copy()
    X_cov, y_cov = coverage_data[feature_columns], coverage_data['coverage']

    size_data = df.dropna(subset=['droplet_size']).copy()
    X_size, y_size = size_data[feature_columns], size_data['droplet_size']

    density_data = df.dropna(subset=['droplet_density']).copy()
    X_dens, y_dens = density_data[feature_columns], density_data['droplet_density']

    print(f"\n Training datasets:")
    print(f"Coverage: {X_cov.shape}")
    print(f"Droplet size: {X_size.shape}")
    print(f"Droplet density: {X_dens.shape}")

    return X_cov, y_cov, X_size, y_size, X_dens, y_dens, feature_columns

X_cov, y_cov, X_size, y_size, X_dens, y_dens, feature_columns = prepare_training_data(df_augmented)

##############################################3
# 4. Train CatBoost Models
###################################################
def train_catboost_models(X_cov, y_cov, X_size, y_size, X_dens, y_dens):
    def split_and_train(X, y, label):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = cb.CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, random_state=42, verbose=False)
        print(f"Training {label} model...")
        model.fit(X_train, y_train, eval_set=(X_test, y_test))
        return model

    return (
        split_and_train(X_cov, y_cov, "coverage"),
        split_and_train(X_size, y_size, "droplet size"),
        split_and_train(X_dens, y_dens, "droplet density")
    )

print(" Starting model training...")
model_coverage, model_droplet_size, model_droplet_density = train_catboost_models(
    X_cov, y_cov, X_size, y_size, X_dens, y_dens
)

####################################################
# 5. Memory-efficient parameter grid generation
########################################################3
def get_parameter_grid_structure(search_scale=3):
    print(f"\n⚙ Preparing scalable grid structure (scale={search_scale})...")
    return {
        'crop_type': [0, 1, 2],
        'uav_model': [0, 1, 2, 3],
        'flight_height': np.round(np.linspace(0.5, 6.0, 10 * search_scale), 2),
        'flight_speed': np.round(np.linspace(0.5, 20.0, 15 * search_scale), 2),
        'spray_volume': np.round(np.linspace(5, 200, 15 * search_scale), 2),
        'atomization_diameter': [0, 1, 2],
        'temperature': np.round(np.linspace(10, 40, 8 * search_scale), 1),
        'humidity': np.round(np.linspace(30, 90, 10 * search_scale), 1),
        'wind_speed': np.round(np.linspace(0, 8, 10 * search_scale), 2),
    }

############################################################
# 6. Streamed batch processing (no memory explosion)
##############################################################
def process_parameter_grid_in_batches(models, batch_size=50000, search_scale=3):
    model_coverage, model_droplet_size, model_droplet_density = models
    grid = get_parameter_grid_structure(search_scale)
    keys = list(grid.keys())

    total_combinations = np.prod([len(v) for v in grid.values()])
    print(f" Estimated total combinations: {total_combinations:,}")
    print(f" Batch size: {batch_size:,}")

    best_results = {
        'top_configurations': pd.DataFrame(),
        'best_coverage': 0,
        'best_optimality_score': 0,
        'total_processed': 0
    }

    generator = product(*grid.values())
    batch_number, processed = 0, 0

    print("\n Streaming through parameter space...")
    while True:
        batch_combos = list(islice(generator, batch_size))
        if not batch_combos:
            break

        batch_number += 1
        batch_df = pd.DataFrame(batch_combos, columns=keys)
        best_results = process_batch(batch_df, models, best_results, batch_number, total_combinations)
        processed += len(batch_df)

        del batch_df, batch_combos
        if batch_number % 5 == 0:
            gc.collect()

    print(f"\n COMPLETED streaming search ({processed:,} combinations).")
    print(f" Best coverage: {best_results['best_coverage']:.2f}%")
    print(f" Best optimality score: {best_results['best_optimality_score']:.2f}")
    return best_results['top_configurations']

def process_batch(batch_df, models, best_results, batch_number, total_combinations):
    model_coverage, model_droplet_size, model_droplet_density = models
    feature_columns = batch_df.columns

    print(f"🔨 Processing batch {batch_number} ({len(batch_df):,} combos)...")
    coverage_predictions = model_coverage.predict(batch_df)
    droplet_size_predictions = model_droplet_size.predict(batch_df)
    droplet_density_predictions = model_droplet_density.predict(batch_df)

    batch_results = batch_df.copy()
    batch_results['predicted_coverage'] = coverage_predictions
    batch_results['predicted_droplet_size'] = droplet_size_predictions
    batch_results['predicted_droplet_density'] = droplet_density_predictions

    coverage_score = batch_results['predicted_coverage']
    droplet_size_score = 100 - np.abs(batch_results['predicted_droplet_size'] - 250) / 2.5
    droplet_density_score = np.minimum(batch_results['predicted_droplet_density'], 80)
    batch_results['optimality_score'] = (
        coverage_score * 0.5 +
        droplet_size_score * 0.3 +
        droplet_density_score * 0.2
    )

    batch_top_10 = batch_results.nlargest(10, 'optimality_score')
    if best_results['top_configurations'].empty:
        best_results['top_configurations'] = batch_top_10
    else:
        combined = pd.concat([best_results['top_configurations'], batch_top_10], ignore_index=True)
        best_results['top_configurations'] = combined.nlargest(10, 'optimality_score')

    best_results['best_coverage'] = max(best_results['best_coverage'], batch_results['predicted_coverage'].max())
    best_results['best_optimality_score'] = max(best_results['best_optimality_score'], batch_results['optimality_score'].max())
    best_results['total_processed'] += len(batch_df)

    progress = (best_results['total_processed'] / total_combinations) * 100
    print(f"📊 Progress: {progress:.2f}% | Best coverage: {best_results['best_coverage']:.2f}%")

    return best_results

#####################################
# 7. Run full search
##########################################3
print("\n🚀 STARTING FULL EXHAUSTIVE SEARCH...")
print("=" * 70)
search_scale = 3  # Adjustable parameter
final_top_configurations = process_parameter_grid_in_batches(
    (model_coverage, model_droplet_size, model_droplet_density),
    batch_size=50000,
    search_scale=search_scale
)

################################################
# 8. Save and display final results
##################################################
print("\n FINAL TOP 10 OPTIMAL CONFIGURATIONS:")
print("=" * 80)
cols = [
    'crop_type', 'uav_model', 'flight_height', 'flight_speed',
    'spray_volume', 'predicted_coverage',
    'predicted_droplet_size', 'predicted_droplet_density',
    'optimality_score'
]
print(final_top_configurations[cols].round(2))

final_top_configurations.to_csv('FINAL_optimal_parameters.csv', index=False)
print("\n Results saved to FINAL_optimal_parameters.csv")
print(" Done!")

