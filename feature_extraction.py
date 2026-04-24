"""
Feature Extraction for Keystroke Dynamics
"""

import pandas as pd
import numpy as np

def extract_keystroke_features(data):
    """Extract 7 features from raw keystroke data"""
    
    features_list = []
    
    for (user_id, session_id), group in data.groupby(['UserID', 'SessionID']):
        group = group.sort_values('DownTime').reset_index(drop=True)
        
        for i in range(1, len(group)):
            current = group.iloc[i]
            previous = group.iloc[i-1]
            
            keycode_curr = current['KeyCode']
            keycode_prev = previous['KeyCode']
            hold_time = current['UpTime'] - current['DownTime']
            
            dd_time = current['DownTime'] - previous['DownTime']
            du_time = current['UpTime'] - previous['DownTime']
            ud_time = current['DownTime'] - previous['UpTime']
            uu_time = current['UpTime'] - previous['UpTime']
            
            if all(0 < t < 2000 for t in [hold_time, dd_time, du_time, ud_time, uu_time]):
                features_list.append({
                    'UserID': user_id,
                    'SessionID': session_id,
                    'KeyCode_curr': keycode_curr,
                    'KeyCode_prev': keycode_prev,
                    'Hold': hold_time,
                    'DD': dd_time,
                    'DU': du_time,
                    'UD': ud_time,
                    'UU': uu_time
                })
    
    return pd.DataFrame(features_list)


def create_train_test_split(features, user_id, train_ratio=0.6, test_ratio=0.2):
    """Create train/test/validation split for a specific user"""
    
    user_data = features[features['UserID'] == user_id].copy()
    n_samples = len(user_data)
    
    train_end = int(n_samples * train_ratio)
    test_end = int(n_samples * (train_ratio + test_ratio))
    
    return {
        'train': user_data.iloc[:train_end],
        'test': user_data.iloc[train_end:test_end],
        'val': user_data.iloc[test_end:]
    }


def prepare_binary_classification_data(features, genuine_user_id, split_type='train'):
    """Prepare binary classification dataset (genuine vs impostor)"""
    
    splits = create_train_test_split(features, genuine_user_id)
    genuine_data = splits[split_type]
    
    impostor_data = features[features['UserID'] != genuine_user_id].copy()
    impostor_sample = impostor_data.sample(n=len(genuine_data), random_state=42)
    
    genuine_data['Label'] = 1
    impostor_sample['Label'] = 0
    
    combined = pd.concat([genuine_data, impostor_sample]).sample(frac=1, random_state=42)
    
    X = combined[['Hold', 'DD', 'DU', 'UD', 'UU']].values
    y = combined['Label'].values
    
    return X, y