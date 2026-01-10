# ============================================================
# FIXED MATH LOGIC FOR DASHBOARD ALIGNMENT
# ============================================================

def weighted_brand_salience(rs8_series: pd.Series, aware_series: pd.Series, wts: np.ndarray) -> float:
    """
    MATCHES PURPLE GRAPH: Calculates % among BRAND AWARE respondents only.
    """
    # 1. Identify only those who are Aware of TFM (RS3_1NET == 1)
    aware_mask = (aware_series == 1).to_numpy()
    
    if aware_mask.sum() == 0:
        return 0.0
    
    # 2. Identify those within the Aware group who associate TFM with the CEP
    # 1 = Selected, 2 = Not Selected. We treat anything else as 0.
    success = (rs8_series == 1).astype(int).to_numpy()
    
    # 3. Calculate: (Weighted Successes) / (Weighted Aware Total)
    numerator = np.sum(success[aware_mask] * wts[aware_mask])
    denominator = np.sum(wts[aware_mask])
    
    return float(numerator / denominator)

def weighted_category_prevalence(rs1_series: pd.Series, wts: np.ndarray) -> float:
    """
    MATCHES PURPLE GRAPH: Calculates % of total survey respondents selecting the CEP.
    """
    # 1 = Selected, 2 = Not Selected. 
    x = (rs1_series == 1).astype(int).to_numpy()
    return float(np.sum(x * wts) / np.sum(wts))
