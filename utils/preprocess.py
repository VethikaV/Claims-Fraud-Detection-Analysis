import numpy as np

def preprocess_input(user_input: dict):
    # Ensure order matches training features
    feature_order = [
    'DiagnosisDiversity_perProvider',
    'ProcedureDiversity_perProvider',
    'NumBeneficiaries_perProvider',
    'AvgClaimAmount_perProvider',
    'State_freq',
    'County_freq',
    'LengthOfStay',
    'ClmDiagnosisCode_10_freq',
    'ClmDiagnosisCode_8_freq',
    'TotalClaimAmount_perBene',
    'Race_freq',
    'ClmDiagnosisCode_7_freq',
    'AvgChronicCond_perBene',
    'ClmProcedureCode_3_freq',
    'ClmProcedureCode_1_freq',
    'ClmDiagnosisCode_9_freq',
    'ClmDiagnosisCode_2_freq',
    'TotalChronicCond_perBene',
    'AvgClaimAmount_perBene',
    'ClmDiagnosisCode_6_freq'
]

    values = [float(user_input[f]) for f in feature_order]
    return np.array(values)
