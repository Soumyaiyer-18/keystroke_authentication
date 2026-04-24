"""
COMPLETE DEMO FOR MILESTONE 3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import extract_keystroke_features, prepare_binary_classification_data
from classifiers import KeystrokeAuthenticator
from r_rcm import RobustRecurrentConfidenceModel

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("KEYSTROKE AUTHENTICATION SYSTEM DEMO")
print("CS5100 Capstone Milestone 3")


# Generate synthetic keystroke data
print("\nGenerating synthetic keystroke data.")

np.random.seed(42)

def generate_user_keystrokes(user_id, n_keystrokes=1000, typing_speed='normal'):
    """Generate synthetic keystroke data"""
    
    if typing_speed == 'fast':
        hold_mean, hold_std = 60, 15
        inter_mean, inter_std = 100, 25
    elif typing_speed == 'slow':
        hold_mean, hold_std = 120, 30
        inter_mean, inter_std = 200, 50
    else:
        hold_mean, hold_std = 90, 20
        inter_mean, inter_std = 150, 35
    
    keystrokes = []
    current_time = 0
    
    for i in range(n_keystrokes):
        key = np.random.choice(list('abcdefghijklmnopqrstuvwxyz '))
        down_time = current_time
        hold_duration = max(20, np.random.normal(hold_mean, hold_std))
        up_time = down_time + hold_duration
        
        keystrokes.append({
            'UserID': user_id,
            'SessionID': 1,
            'KeyCode': key,
            'DownTime': int(down_time),
            'UpTime': int(up_time)
        })
        
        current_time = up_time + max(20, np.random.normal(inter_mean, inter_std))
    
    return pd.DataFrame(keystrokes)

user1_data = generate_user_keystrokes(1, 800, 'fast')
user2_data = generate_user_keystrokes(2, 800, 'normal')
user3_data = generate_user_keystrokes(3, 800, 'slow')

all_data = pd.concat([user1_data, user2_data, user3_data], ignore_index=True)

print(f"Generated {len(all_data)} keystrokes for 3 users")
print(f"User 1: Fast typer ({len(user1_data)} keystrokes)")
print(f"User 2: Normal typer ({len(user2_data)} keystrokes)")
print(f"User 3: Slow typer ({len(user3_data)} keystrokes)")

# Extract features
print("\nExtracting keystroke features.")

features = extract_keystroke_features(all_data)

print(f"Extracted {len(features)} feature vectors")
print(f"\nFeature statistics:")
print(features[['Hold', 'DD', 'DU', 'UD', 'UU']].describe())

# Visualize typing patterns
print("\nVisualizing typing patterns.")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, user_id in enumerate([1, 2, 3]):
    user_features = features[features['UserID'] == user_id]
    axes[i].hist(user_features['Hold'], bins=30, alpha=0.7, color=f'C{i}', edgecolor='black')
    axes[i].set_title(f'User {user_id} - Hold Time Distribution')
    axes[i].set_xlabel('Hold Time (ms)')
    axes[i].set_ylabel('Frequency')
    axes[i].axvline(user_features['Hold'].mean(), color='red', linestyle='--', 
                    label=f'Mean: {user_features["Hold"].mean():.1f}ms')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('typing_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: typing_patterns.png")
plt.close()

# Train classifiers
print("\nTraining ensemble classifiers for User 1.")

X_train, y_train = prepare_binary_classification_data(features, genuine_user_id=1, split_type='train')
X_test, y_test = prepare_binary_classification_data(features, genuine_user_id=1, split_type='test')
X_val, y_val = prepare_binary_classification_data(features, genuine_user_id=1, split_type='val')

print(f"\nData split sizes:")
print(f"Training: {len(X_train)} samples")
print(f"Testing: {len(X_test)} samples")
print(f"Validation: {len(X_val)} samples\n")

authenticator = KeystrokeAuthenticator()
authenticator.train(X_train, y_train, verbose=True)

# Evaluate classifiers
print("\nEvaluating classifiers.")

results = authenticator.evaluate(X_test, y_test)

print("CLASSIFIER PERFORMANCE ON TEST SET")

for name in ['svm', 'rf', 'xgb', 'ensemble']:
    print(f"\n{name.upper()}:")
    print(f"Accuracy: {results[name]['accuracy']:.2%}")
    print(f"Confusion Matrix:\n{results[name]['confusion_matrix']}")

# Plot comparison
accuracies = {name: results[name]['accuracy'] for name in ['svm', 'rf', 'xgb', 'ensemble']}

plt.figure(figsize=(10, 6))
bars = plt.bar(accuracies.keys(), accuracies.values(), color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'])
plt.ylabel('Accuracy', fontsize=12)
plt.title('Classifier Performance Comparison (User 1)', fontsize=14, fontweight='bold')
plt.ylim([0.7, 1.0])

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1%}', ha='center', va='bottom', fontweight='bold')

plt.grid(True, alpha=0.3, axis='y')
plt.savefig('classifier_comparison.png', dpi=300, bbox_inches='tight')
print("\nSaved: classifier_comparison.png")
plt.close()

# R-RCM authentication
print("\nRunning R-RCM authentication.")

genuine_indices = y_val == 1
impostor_indices = y_val == 0

genuine_probs = authenticator.ensemble_fusion(X_val[genuine_indices])
impostor_probs = authenticator.ensemble_fusion(X_val[impostor_indices])

rcm = RobustRecurrentConfidenceModel(H=0.5, Z=0.1, M=0.05, N=0.05, D=0.82, T_f=0.75)

ANGA = rcm.calculate_ANGA(genuine_probs)
ANIA = rcm.calculate_ANIA(impostor_probs)


print("R-RCM AUTHENTICATION RESULTS")
print(f"\nANGA (Genuine User): {ANGA:.2%}")
print(f"ANIA (Impostor):     {ANIA:.2%}")

if ANGA > 0.8 and ANIA < 0.4:
    print("\nEXCELLENT PERFORMANCE (meets paper's criteria)")
elif ANGA > 0.7 and ANIA < 0.5:
    print("\nGOOD PERFORMANCE")

# Plot confidence trajectories
print("\nGenerating confidence trajectory visualizations.")
rcm.plot_confidence_trajectory(genuine_probs, impostor_probs, save_path='confidence_trajectory.png')

# Summary

print("DEMO COMPLETE")
print("\nFiles created:")
print("typing_patterns.png")
print("classifier_comparison.png")
print("confidence_trajectory.png")
