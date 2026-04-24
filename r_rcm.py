"""
Robust Recurrent Confidence Model (R-RCM)
"""

import numpy as np
import matplotlib.pyplot as plt

class RobustRecurrentConfidenceModel:
    """R-RCM: Maintains confidence score updated after each keystroke"""
    
    def __init__(self, H=0.5, Z=0.1, M=0.05, N=0.05, D=0.82, T_f=0.75):
        self.H = H
        self.Z = Z
        self.M = M
        self.N = N
        self.D = D
        self.T_f = T_f
        
    def compute_confidence_change(self, confidence, prob):
        """Compute change in confidence"""
        
        if confidence >= self.D:
            delta_conf = -self.N + (2 * self.N) / (1 + np.exp(-(prob - self.H) / self.Z))
        elif confidence < self.D and prob < self.H:
            delta_conf = -self.N + (2 * self.N * (1 - self.H)) / (1 + np.exp(-(prob - self.H) / self.Z))
        else:
            delta_conf = self.N + (3 * self.N) / (1 + np.exp(-(prob - self.H) / self.Z))
        
        delta_conf = min(delta_conf, self.M)
        return delta_conf
    
    def authenticate_sequence(self, prob_sequence):
        """Authenticate a sequence of actions"""
        
        confidence = 1.0
        confidence_trajectory = [confidence]
        lockout_points = []
        
        for t, prob in enumerate(prob_sequence):
            delta_conf = self.compute_confidence_change(confidence, prob)
            confidence = max(0.0, min(1.0, confidence + delta_conf))
            confidence_trajectory.append(confidence)
            
            if confidence < self.T_f:
                lockout_points.append(t)
                confidence = 1.0
        
        return {
            'confidence_trajectory': confidence_trajectory,
            'lockout_points': lockout_points,
            'final_confidence': confidence,
            'num_lockouts': len(lockout_points)
        }
    
    def calculate_ANGA(self, genuine_prob_sequence):
        """Calculate Average Number of Genuine Actions"""
        
        result = self.authenticate_sequence(genuine_prob_sequence)
        total_actions = len(genuine_prob_sequence)
        lockouts = result['lockout_points']
        
        if len(lockouts) == 0:
            return 1.0
        
        lockout_indices = [0] + lockouts + [total_actions]
        action_counts = np.diff(lockout_indices)
        ANGA = np.sum(action_counts) / (len(lockouts) * total_actions)
        
        return ANGA
    
    def calculate_ANIA(self, impostor_prob_sequence):
        """Calculate Average Number of Impostor Actions"""
        
        result = self.authenticate_sequence(impostor_prob_sequence)
        total_actions = len(impostor_prob_sequence)
        lockouts = result['lockout_points']
        
        if len(lockouts) == 0:
            return 1.0
        
        lockout_indices = [0] + lockouts + [total_actions]
        action_counts = np.diff(lockout_indices)
        ANIA = np.sum(action_counts) / (len(lockouts) * total_actions)
        
        return ANIA
    
    def plot_confidence_trajectory(self, genuine_probs, impostor_probs, save_path=None):
        """Plot confidence trajectories"""
        
        genuine_result = self.authenticate_sequence(genuine_probs)
        impostor_result = self.authenticate_sequence(impostor_probs)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Genuine user
        ax1.plot(genuine_result['confidence_trajectory'], color='green', linewidth=2)
        ax1.axhline(y=self.D, color='orange', linestyle='--', label=f'Alert ({self.D})')
        ax1.axhline(y=self.T_f, color='red', linestyle='--', label=f'Final ({self.T_f})')
        ax1.set_xlabel('Action Index')
        ax1.set_ylabel('Confidence')
        ax1.set_title('Genuine User - Confidence Trajectory')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # Impostor user
        ax2.plot(impostor_result['confidence_trajectory'], color='red', linewidth=2)
        ax2.axhline(y=self.D, color='orange', linestyle='--', label=f'Alert ({self.D})')
        ax2.axhline(y=self.T_f, color='red', linestyle='--', label=f'Final ({self.T_f})')
        for i, lockout in enumerate(impostor_result['lockout_points'][:10]):  # Show first 10
            ax2.axvline(x=lockout, color='darkred', alpha=0.5, linewidth=2)
        ax2.set_xlabel('Action Index')
        ax2.set_ylabel('Confidence')
        ax2.set_title('Impostor User - Confidence Trajectory')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f" Plot saved: {save_path}")
        
        plt.show()
        return fig