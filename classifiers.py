"""
Ensemble Classifiers for Keystroke Authentication
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb

class KeystrokeAuthenticator:
    """Ensemble classifier for keystroke authentication"""
    
    def __init__(self):
        self.svm = None
        self.rf = None
        self.xgb = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def train(self, X_train, y_train, verbose=True):
        """Train all three classifiers"""
        
        X_scaled = self.scaler.fit_transform(X_train)
        
        if verbose:
            print("Training SVM.")
        self.svm = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42)
        self.svm.fit(X_scaled, y_train)
        
        if verbose:
            print("Training Random Forest.")
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        self.rf.fit(X_scaled, y_train)
        
        if verbose:
            print("Training XGBoost.")
        self.xgb = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        self.xgb.fit(X_scaled, y_train)
        
        self.is_trained = True
        if verbose:
            print("Training complete!")
        
    def predict_proba_single(self, X):
        """Get probability predictions from each classifier"""
        
        if not self.is_trained:
            raise ValueError("Classifiers not trained yet!")
        
        X_scaled = self.scaler.transform(X)
        
        svm_prob = self.svm.predict_proba(X_scaled)[:, 1]
        rf_prob = self.rf.predict_proba(X_scaled)[:, 1]
        xgb_prob = self.xgb.predict_proba(X_scaled)[:, 1]
        
        return {'svm': svm_prob, 'rf': rf_prob, 'xgb': xgb_prob}
    
    def ensemble_fusion(self, X, weights=[0.33, 0.33, 0.33]):
        """Weighted ensemble fusion"""
        
        probs = self.predict_proba_single(X)
        w0, w1, w2 = weights
        fused = (w0 * probs['svm'] + w1 * probs['rf'] + w2 * probs['xgb']) / (w0 + w1 + w2)
        
        return fused
    
    def evaluate(self, X_test, y_test):
        """Evaluate all classifiers and ensemble"""
        
        results = {}
        X_scaled = self.scaler.transform(X_test)
        
        # SVM
        svm_pred = self.svm.predict(X_scaled)
        results['svm'] = {
            'accuracy': accuracy_score(y_test, svm_pred),
            'confusion_matrix': confusion_matrix(y_test, svm_pred)
        }
        
        # Random Forest
        rf_pred = self.rf.predict(X_scaled)
        results['rf'] = {
            'accuracy': accuracy_score(y_test, rf_pred),
            'confusion_matrix': confusion_matrix(y_test, rf_pred)
        }
        
        # XGBoost
        xgb_pred = self.xgb.predict(X_scaled)
        results['xgb'] = {
            'accuracy': accuracy_score(y_test, xgb_pred),
            'confusion_matrix': confusion_matrix(y_test, xgb_pred)
        }
        
        # Ensemble
        ensemble_prob = self.ensemble_fusion(X_test)
        ensemble_pred = (ensemble_prob > 0.5).astype(int)
        results['ensemble'] = {
            'accuracy': accuracy_score(y_test, ensemble_pred),
            'confusion_matrix': confusion_matrix(y_test, ensemble_pred)
        }
        
        return results