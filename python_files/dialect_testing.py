import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import gensim.downloader as api
import seaborn as sns
from pathlib import Path
from feature_extractor import FeatureExtractor

def load_dialect_samples(sae_path: str, aae_path: str) -> tuple:
    """Load parallel dialect samples"""
    with open(sae_path) as f:
        sae_samples = [line.strip() for line in f]
    with open(aae_path) as f:
        aae_samples = [line.strip() for line in f]
    return sae_samples, aae_samples

def compute_bias_metrics(preds_sae: np.ndarray, preds_aae: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute various bias metrics"""
    preds_sae = preds_sae.flatten()
    preds_aae = preds_aae.flatten()
    
    metrics = {
        'mean_sae': np.mean(preds_sae),
        'mean_aae': np.mean(preds_aae),
        'toxic_rate_sae': np.mean(preds_sae >= threshold),
        'toxic_rate_aae': np.mean(preds_aae >= threshold),
        'score_differences': preds_aae - preds_sae,
    }
    
    metrics['mean_difference'] = np.mean(metrics['score_differences'])
    metrics['percent_aae_higher'] = np.mean(metrics['score_differences'] > 0) * 100
    
    # Add confidence analysis
    metrics['high_conf_sae'] = np.mean((preds_sae >= 0.8) | (preds_sae <= 0.2))
    metrics['high_conf_aae'] = np.mean((preds_aae >= 0.8) | (preds_aae <= 0.2))
    
    return metrics

def plot_bias_results(metrics: dict, raw_preds_sae: np.ndarray, raw_preds_aae: np.ndarray, 
                     save_path: str = None):
    """Enhanced visualization of bias metrics"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Mean scores comparison
    means = [metrics['mean_sae'], metrics['mean_aae']]
    ax1.bar(['SAE', 'AAE'], means)
    ax1.set_title('Mean Toxicity Scores by Dialect')
    ax1.set_ylabel('Mean Score')
    
    # Plot 2: Score differences distribution
    sns.histplot(metrics['score_differences'], ax=ax2)
    ax2.axvline(x=0, color='r', linestyle='--')
    ax2.set_title('Distribution of Score Differences (AAE - SAE)')
    
    # Plot 3: SAE score distribution
    sns.histplot(raw_preds_sae, ax=ax3, bins=20)
    ax3.set_title('SAE Score Distribution')
    
    # Plot 4: AAE score distribution
    sns.histplot(raw_preds_aae, ax=ax4, bins=20)
    ax4.set_title('AAE Score Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()

def verify_features(X_sae, X_aae):
    """Debug helper to verify feature extraction"""
    print(f"\nFeature Statistics:")
    print(f"SAE features - Shape: {X_sae.shape}")
    print(f"AAE features - Shape: {X_aae.shape}")
    print(f"SAE features range: [{X_sae.min():.3f}, {X_sae.max():.3f}]")
    print(f"AAE features range: [{X_aae.min():.3f}, {X_aae.max():.3f}]")
    
    # Add statistics per feature
    print("\nFeature-wise statistics:")
    feature_means = np.mean(X_sae, axis=0)
    feature_stds = np.std(X_sae, axis=0)
    print(f"Mean range: [{feature_means.min():.3f}, {feature_means.max():.3f}]")
    print(f"Std range: [{feature_stds.min():.3f}, {feature_stds.max():.3f}]")

def verify_predictions(preds):
    """Debug helper to analyze model predictions"""
    print(f"\nPrediction Statistics:")
    print(f"Shape: {preds.shape}")
    print(f"Range: [{preds.min():.3f}, {preds.max():.3f}]")
    print(f"Mean: {preds.mean():.3f}")
    print(f"Distribution:")
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    hist, _ = np.histogram(preds, bins=bins)
    for i in range(len(bins)-1):
        print(f"{bins[i]:.1f}-{bins[i+1]:.1f}: {hist[i]}")

def predict_with_threshold(model, X, threshold=0.5):  # Lower threshold
    """Make predictions with an adjusted confidence threshold"""
    raw_preds = model.predict(X, verbose=0)
    return raw_preds, (raw_preds >= threshold).astype(int)

def normalize_features(X):
    """Normalize features to zero mean and unit variance"""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    return (X - mean) / std

def main():
    # Load GloVe embeddings
    print("Loading GloVe embeddings...")
    glove_model = api.load("glove-wiki-gigaword-300")
    
    # Load model and initialize feature extractor
    print("Loading model...")
    model = load_model('/Users/crownedprinz/Documents/Projects/Python/dual-stage-toxic-comment-detection-system/models/dnnModel.keras')
    print("\nModel Summary:")
    model.summary()
    print("\nModel Config:")
    print(model.get_config())
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
    extractor = FeatureExtractor(glove_model)
    
    # Load dialect samples
    sae_samples, aae_samples = load_dialect_samples('/Users/crownedprinz/Documents/Projects/Python/dual-stage-toxic-comment-detection-system/raw_data/Dialect Dataset/sae_samples.txt', '/Users/crownedprinz/Documents/Projects/Python/dual-stage-toxic-comment-detection-system/raw_data/Dialect Dataset/aave_samples.txt')
    
    # Check feature dimensions
    X_sae_sample = extractor.get_combined_features(sae_samples[0])
    print(f"Feature vector shape: {X_sae_sample.shape}")
    
    # Extract and normalize features
    X_sae = np.vstack([extractor.get_combined_features(text) for text in sae_samples])
    X_aae = np.vstack([extractor.get_combined_features(text) for text in aae_samples])
    
    # Normalize features
    X_sae = normalize_features(X_sae)
    X_aae = normalize_features(X_aae)
    
    # Verify normalized features
    verify_features(X_sae, X_aae)
    
    # Adjust threshold for more balanced predictions
    raw_preds_sae, preds_sae = predict_with_threshold(model, X_sae, threshold=0.5)
    raw_preds_aae, preds_aae = predict_with_threshold(model, X_aae, threshold=0.5)
    
    print("\nRaw SAE Predictions:")
    verify_predictions(raw_preds_sae)
    print("\nThresholded SAE Predictions:")
    verify_predictions(preds_sae)
    
    print("\nRaw AAE Predictions:")
    verify_predictions(raw_preds_aae)
    print("\nThresholded AAE Predictions:")
    verify_predictions(preds_aae)
    
    # Compute metrics
    metrics = compute_bias_metrics(preds_sae, preds_aae)
    
    # Print enhanced results
    print("\n=== Enhanced Dialect Bias Analysis ===")
    print(f"Average toxicity scores:")
    print(f"  SAE: {metrics['mean_sae']:.3f}")
    print(f"  AAE: {metrics['mean_aae']:.3f}")
    print(f"\nHigh confidence predictions (>0.8 or <0.2):")
    print(f"  SAE: {metrics['high_conf_sae']*100:.1f}%")
    print(f"  AAE: {metrics['high_conf_aae']*100:.1f}%")
    
    # Plot enhanced results
    plot_bias_results(metrics, raw_preds_sae, raw_preds_aae, 'dialect_bias_results.png')

if __name__ == "__main__":
    main()