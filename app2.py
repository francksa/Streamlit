# ============================================================
# UPDATED UTILITIES FOR DASHBOARD ALIGNMENT
# ============================================================

def weighted_brand_salience(rs8_series: pd.Series, aware_series: pd.Series, w: np.ndarray) -> float:
    """
    FIXED: Calculates salience among TOTAL BRAND AWARE respondents 
    to match dashboard 'Banner' logic.
    """
    # Base: Anyone who is aware of TFM (S2_1 == 1 in your codebook)
    # We filter for the MSA universe outside this function.
    aware_mask = (aware_series == 1).to_numpy()
    
    if aware_mask.sum() == 0:
        return 0.0
    
    # Success: Respondent selected TFM for this CEP (RS8_i_1NET == 1)
    # Note: We treat NaNs/2s as 0 (not selected) within the Aware base
    success = (rs8_series == 1).astype(int).to_numpy()
    
    # Return weighted mean: (Successes * Weights) / (Total Aware * Weights)
    return float(np.sum(success[aware_mask] * w[aware_mask]) / np.sum(w[aware_mask]))

# ============================================================
# MODIFIED MAIN LOOP SECTION
# ============================================================
# In your main() function, replace the salience_w calculation with this:

# 1. Identify Aware Column (Based on your codebook S2_1 = TFM Awareness)
aware_col = "S2_1" 

# 2. Apply MSA Filter (Already in your code)
raw = apply_msa_filter(raw_all, use_msa_only)
w = raw[WEIGHT_COL].to_numpy()

# 3. Calculate the fixed Baseline Salience
salience_w = np.array([
    weighted_brand_salience(raw[f"RS8_{cep}_1NET"], raw[aware_col], w) 
    for cep in cep_idx
])

# 4. Display in Sidebar (using the new salience_w)
for cep in ordered_ceps:
    j = cep_idx.index(cep)
    # This will now display the 7.1% or similar MSA-specific number
    uplifts[j] = st.sidebar.slider(
        f"{labels[cep]} (Current: {salience_w[j]*100:.1f}%)",
        min_value=0,
        max_value=20,
        value=0
    )


