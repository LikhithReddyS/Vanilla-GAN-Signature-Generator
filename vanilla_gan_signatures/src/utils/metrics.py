"""
Metrics
=======
Evaluation metrics for verification.
"""

import numpy as np
from sklearn.metrics import roc_curve, auc

def compute_eer(y_true, y_scores):
    """
    Computes Equal Error Rate (EER), FAR, FRR at EER threshold.
    
    Args:
        y_true: True binary labels (0 or 1).
        y_scores: Predicted probabilities or scores.
        
    Returns:
        eer: Equal Error Rate.
        thresh: Threshold at EER.
        far: False Acceptance Rate at EER.
        frr: False Rejection Rate at EER.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    # EER is where FAR = FRR
    # FAR = FPR
    # FRR = 1 - TPR
    
    fnr = 1 - tpr
    
    # Find index where absolute difference is minimal
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    
    # Theoretically EER is the point where FPR == FNR
    # We can interpolate or just take the closest point
    EER = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    
    return EER, eer_threshold
