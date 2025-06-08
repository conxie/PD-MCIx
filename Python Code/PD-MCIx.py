import pandas as pd
import numpy as np
from typing import Optional, Union, List, Dict

# PD-MCIx - Subjective Optional

def handle_type(df: pd.DataFrame, type_: str, base_name: str, 
                standalone: Optional[str] = None, standalone_cutoff: Optional[float] = None, 
                cutoff_direction: str = "less",
                series: Optional[List[str]] = None, series_array: Optional[List[Union[str, int, float]]] = None, 
                ogsame: Optional[str] = None) -> pd.DataFrame:
    
    impairment_col = f"{base_name}_impairment"
    
    if type_ == "standalone":
        if cutoff_direction == "less":
            df[impairment_col] = np.where(df[standalone] <= standalone_cutoff, 1,
                                         np.where(df[standalone] > standalone_cutoff, 0, np.nan))
        elif cutoff_direction == "greater":
            df[impairment_col] = np.where(df[standalone] >= standalone_cutoff, 1,
                                         np.where(df[standalone] < standalone_cutoff, 0, np.nan))
    elif type_ == "series":
        # series is list of column names (strings)
        df[impairment_col] = np.where(
            df[series].apply(lambda row: all(item in row.values for item in series_array), 1, 0
        )
    elif type_ == "OGSame":
        # ogsame is a boolean or numeric vector with impairment flags
        df[impairment_col] = df[ogsame]
    else:
        # Unknown type or no data
        df[impairment_col] = np.nan
        
    return df


def PD_MCIx(
    df: pd.DataFrame,
    # Neuropsychological Assessment parameters
    attentionOne: str, attentionOne_cutoff: float, 
    attentionTwo: str, attentionTwo_cutoff: float,
    memoryOne: str, memoryOne_cutoff: float, 
    memoryTwo: str, memoryTwo_cutoff: float,
    execuFuncOne: str, execuFuncOne_cutoff: float, 
    execuFuncTwo: str, execuFuncTwo_cutoff: float,
    visuoSpatOne: str, visuoSpatOne_cutoff: float, 
    visuoSpatTwo: str, visuoSpatTwo_cutoff: float,
    languageOne: str, languageOne_cutoff: float, 
    languageTwo: str, languageTwo_cutoff: float,
    
    # Tertiary fallback
    attentionThree: Optional[str] = None, attentionThree_cutoff: Optional[float] = None,
    memoryThree: Optional[str] = None, memoryThree_cutoff: Optional[float] = None,
    execuFuncThree: Optional[str] = None, execuFuncThree_cutoff: Optional[float] = None,
    visuoSpatThree: Optional[str] = None, visuoSpatThree_cutoff: Optional[float] = None,
    languageThree: Optional[str] = None, languageThree_cutoff: Optional[float] = None,
    
    # Subjective Response parameters
    type_1: str, subjectiveOne_standalone: Optional[str] = None, subjectiveOne_standalone_cutoff: Optional[float] = None, 
    subjectiveOne_cutoff_direction: str = "less", subjectiveOne_Series: Optional[List[str]] = None, 
    subjectiveOne_Series_array: Optional[List[Union[str, int, float]]] = None, subjectiveOne_OGSame: Optional[str] = None,
    type_2: str, subjectiveTwo_standalone: Optional[str] = None, subjectiveTwo_standalone_cutoff: Optional[float] = None, 
    subjectiveTwo_cutoff_direction: str = "less", subjectiveTwo_Series: Optional[List[str]] = None, 
    subjectiveTwo_Series_array: Optional[List[Union[str, int, float]]] = None, subjectiveTwo_OGSame: Optional[str] = None,
    type_3: str, subjectiveThree_standalone: Optional[str] = None, subjectiveThree_standalone_cutoff: Optional[float] = None, 
    subjectiveThree_cutoff_direction: str = "less", subjectiveThree_Series: Optional[List[str]] = None, 
    subjectiveThree_Series_array: Optional[List[Union[str, int, float]]] = None, subjectiveThree_OGSame: Optional[str] = None,
    type_4: str, subjectiveFour_standalone: Optional[str] = None, subjectiveFour_standalone_cutoff: Optional[float] = None, 
    subjectiveFour_cutoff_direction: str = "less", subjectiveFour_Series: Optional[List[str]] = None, 
    subjectiveFour_Series_array: Optional[List[Union[str, int, float]]] = None, subjectiveFour_OGSame: Optional[str] = None,
    
    # Functional Response parameters
    type_1F: str, functionalOne_standalone: Optional[str] = None, functionalOne_standalone_cutoff: Optional[float] = None, 
    functionalOne_cutoff_direction: str = "less", functionalOne_Series: Optional[List[str]] = None, 
    functionalOne_Series_array: Optional[List[Union[str, int, float]]] = None, functionalOne_OGSame: Optional[str] = None,
    type_2F: str, functionalTwo_standalone: Optional[str] = None, functionalTwo_standalone_cutoff: Optional[float] = None, 
    functionalTwo_cutoff_direction: str = "less", functionalTwo_Series: Optional[List[str]] = None, 
    functionalTwo_Series_array: Optional[List[Union[str, int, float]]] = None, functionalTwo_OGSame: Optional[str] = None,
    type_3F: str, functionalThree_standalone: Optional[str] = None, functionalThree_standalone_cutoff: Optional[float] = None, 
    functionalThree_cutoff_direction: str = "less", functionalThree_Series: Optional[List[str]] = None, 
    functionalThree_Series_array: Optional[List[Union[str, int, float]]] = None, functionalThree_OGSame: Optional[str] = None,
    type_4F: str, functionalFour_standalone: Optional[str] = None, functionalFour_standalone_cutoff: Optional[float] = None, 
    functionalFour_cutoff_direction: str = "less", functionalFour_Series: Optional[List[str]] = None, 
    functionalFour_Series_array: Optional[List[Union[str, int, float]]] = None, functionalFour_OGSame: Optional[str] = None
) -> pd.DataFrame:
    
    # Add neuropsych vectors as columns
    df = df.assign(
        attentionOne_vec = df[attentionOne],
        attentionTwo_vec = df[attentionTwo],
        memoryOne_vec = df[memoryOne],
        memoryTwo_vec = df[memoryTwo],
        execuFuncOne_vec = df[execuFuncOne],
        execuFuncTwo_vec = df[execuFuncTwo],
        visuoSpatOne_vec = df[visuoSpatOne],
        visuoSpatTwo_vec = df[visuoSpatTwo],
        languageOne_vec = df[languageOne],
        languageTwo_vec = df[languageTwo]
    )
    
    # Tertiary fallback
    if attentionThree is not None:
        df['attentionThree_vec'] = df[attentionThree]
    if memoryThree is not None:
        df['memoryThree_vec'] = df[memoryThree]
    if execuFuncThree is not None:
        df['execuFuncThree_vec'] = df[execuFuncThree]
    if visuoSpatThree is not None:
        df['visuoSpatThree_vec'] = df[visuoSpatThree]
    if languageThree is not None:
        df['languageThree_vec'] = df[languageThree]
    
    # Neuropsychological impairments
    def calculate_impaired(row, primary_col, primary_cutoff, fallback_col=None, fallback_cutoff=None):
        if not pd.isna(row[primary_col]):
            return 1 if row[primary_col] <= primary_cutoff else 0
        elif fallback_col is not None and not pd.isna(row[fallback_col]):
            return 1 if row[fallback_col] <= fallback_cutoff else 0
        return np.nan
    
    df['attentionOne_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'attentionOne_vec', attentionOne_cutoff, 
                                      'attentionThree_vec', attentionThree_cutoff), axis=1)
    df['attentionTwo_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'attentionTwo_vec', attentionTwo_cutoff, 
                                      'attentionThree_vec', attentionThree_cutoff), axis=1)
    
    df['memoryOne_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'memoryOne_vec', memoryOne_cutoff, 
                                     'memoryThree_vec', memoryThree_cutoff), axis=1)
    df['memoryTwo_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'memoryTwo_vec', memoryTwo_cutoff, 
                                     'memoryThree_vec', memoryThree_cutoff), axis=1)
    
    df['execuFuncOne_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'execuFuncOne_vec', execuFuncOne_cutoff, 
                                      'execuFuncThree_vec', execuFuncThree_cutoff), axis=1)
    df['execuFuncTwo_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'execuFuncTwo_vec', execuFuncTwo_cutoff, 
                                      'execuFuncThree_vec', execuFuncThree_cutoff), axis=1)
    
    df['visuoSpatOne_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'visuoSpatOne_vec', visuoSpatOne_cutoff, 
                                      'visuoSpatThree_vec', visuoSpatThree_cutoff), axis=1)
    df['visuoSpatTwo_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'visuoSpatTwo_vec', visuoSpatTwo_cutoff, 
                                      'visuoSpatThree_vec', visuoSpatThree_cutoff), axis=1)
    
    df['languageOne_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'languageOne_vec', languageOne_cutoff, 
                                      'languageThree_vec', languageThree_cutoff), axis=1)
    df['languageTwo_impaired'] = df.apply(
        lambda row: calculate_impaired(row, 'languageTwo_vec', languageTwo_cutoff, 
                                      'languageThree_vec', languageThree_cutoff), axis=1)
    
    # Apply Subjective impairments
    df = handle_type(df, type_1, "SubjectiveOne", subjectiveOne_standalone, subjectiveOne_standalone_cutoff, 
                    subjectiveOne_cutoff_direction, subjectiveOne_Series, subjectiveOne_Series_array, subjectiveOne_OGSame)
    df = handle_type(df, type_2, "SubjectiveTwo", subjectiveTwo_standalone, subjectiveTwo_standalone_cutoff, 
                    subjectiveTwo_cutoff_direction, subjectiveTwo_Series, subjectiveTwo_Series_array, subjectiveTwo_OGSame)
    df = handle_type(df, type_3, "SubjectiveThree", subjectiveThree_standalone, subjectiveThree_standalone_cutoff, 
                    subjectiveThree_cutoff_direction, subjectiveThree_Series, subjectiveThree_Series_array, subjectiveThree_OGSame)
    df = handle_type(df, type_4, "SubjectiveFour", subjectiveFour_standalone, subjectiveFour_standalone_cutoff, 
                    subjectiveFour_cutoff_direction, subjectiveFour_Series, subjectiveFour_Series_array, subjectiveFour_OGSame)
    
    # Apply Functional impairments
    df = handle_type(df, type_1F, "FunctionalOne", functionalOne_standalone, functionalOne_standalone_cutoff, 
                    functionalOne_cutoff_direction, functionalOne_Series, functionalOne_Series_array, functionalOne_OGSame)
    df = handle_type(df, type_2F, "FunctionalTwo", functionalTwo_standalone, functionalTwo_standalone_cutoff, 
                    functionalTwo_cutoff_direction, functionalTwo_Series, functionalTwo_Series_array, functionalTwo_OGSame)
    df = handle_type(df, type_3F, "FunctionalThree", functionalThree_standalone, functionalThree_standalone_cutoff, 
                    functionalThree_cutoff_direction, functionalThree_Series, functionalThree_Series_array, functionalThree_OGSame)
    df = handle_type(df, type_4F, "FunctionalFour", functionalFour_standalone, functionalFour_standalone_cutoff, 
                    functionalFour_cutoff_direction, functionalFour_Series, functionalFour_Series_array, functionalFour_OGSame)
    
    # Final calculations
    df['Attention_impaired'] = df['attentionOne_impaired'] + df['attentionTwo_impaired']
    df['Memory_impaired'] = df['memoryOne_impaired'] + df['memoryTwo_impaired']
    df['Executive_impaired'] = df['execuFuncOne_impaired'] + df['execuFuncTwo_impaired']
    df['Visuospatial_impaired'] = df['visuoSpatOne_impaired'] + df['visuoSpatTwo_impaired']
    df['Language_impaired'] = df['languageOne_impaired'] + df['languageTwo_impaired']
    
    def calculate_NP_impaired(row):
        conditions = [
            row['Attention_impaired'] >= 2,
            row['Memory_impaired'] >= 2,
            row['Executive_impaired'] >= 2,
            row['Visuospatial_impaired'] >= 2,
            row['Language_impaired'] >= 2,
            row['Attention_impaired'] >= 1 and row['Memory_impaired'] >= 1,
            row['Attention_impaired'] >= 1 and row['Executive_impaired'] >= 1,
            row['Attention_impaired'] >= 1 and row['Visuospatial_impaired'] >= 1,
            row['Attention_impaired'] >= 1 and row['Language_impaired'] >= 1,
            row['Memory_impaired'] >= 1 and row['Executive_impaired'] >= 1,
            row['Memory_impaired'] >= 1 and row['Visuospatial_impaired'] >= 1,
            row['Memory_impaired'] >= 1 and row['Language_impaired'] >= 1,
            row['Executive_impaired'] >= 1 and row['Visuospatial_impaired'] >= 1,
            row['Executive_impaired'] >= 1 and row['Language_impaired'] >= 1,
            row['Visuospatial_impaired'] >= 1 and row['Language_impaired'] >= 1
        ]
        return 1 if any(conditions) else 0
    
    df['NP_impaired'] = df.apply(calculate_NP_impaired, axis=1)
    
    # Calculate Subjective and Functional totals
    subjective_cols = [col for col in df.columns if col.startswith('Subjective') and col.endswith('impairment')]
    df['Subjective_total'] = df[subjective_cols].sum(axis=1)
    df['Subjective_impaired'] = np.where(df['Subjective_total'] >= 1, 1, 0)
    
    functional_cols = [col for col in df.columns if col.startswith('Functional') and col.endswith('impairment')]
    df['Functional_total'] = df[functional_cols].sum(axis=1)
    df['Functional_impaired'] = np.where(df['Functional_total'] >= 1, 1, 0)
    
    # Missing values code
    def get_missing_values(row):
        cols_to_check = [
            'attentionOne_impaired', 'attentionTwo_impaired',
            'memoryOne_impaired', 'memoryTwo_impaired',
            'execuFuncOne_impaired', 'execuFuncTwo_impaired',
            'visuoSpatOne_impaired', 'visuoSpatTwo_impaired',
            'languageOne_impaired', 'languageTwo_impaired'
        ] + subjective_cols + functional_cols
        
        missing_cols = [col for col in cols_to_check if pd.isna(row[col])]
        if missing_cols:
            return f"Patient is missing values in {', '.join(missing_cols)}"
        return "No missing values"
    
    df['missingValues'] = df.apply(get_missing_values, axis=1)
    
    # AutoDx calculation
    conditions = [
        (df['NP_impaired'] == 0) & (df['Subjective_impaired'] == 0) & (df['Functional_impaired'] == 0),
        (df['NP_impaired'] == 1) & (df['Subjective_impaired'] == 0) & (df['Functional_impaired'] == 0),
        (df['NP_impaired'] == 0) & (df['Subjective_impaired'] == 1) & (df['Functional_impaired'] == 0),
        (df['NP_impaired'] == 0) & (df['Subjective_impaired'] == 0) & (df['Functional_impaired'] == 1),
        (df['NP_impaired'] == 1) & (df['Subjective_impaired'] == 0) & (df['Functional_impaired'] == 1),
        (df['NP_impaired'] == 1) & (df['Subjective_impaired'] == 1) & (df['Functional_impaired'] == 0),
        (df['NP_impaired'] == 0) & (df['Subjective_impaired'] == 1) & (df['Functional_impaired'] == 1),
        (df['NP_impaired'] == 1) & (df['Subjective_impaired'] == 1) & (df['Functional_impaired'] == 1)
    ]
    choices = [0, 1, 0, 0, 0, 1, 0, 1]
    df['AutoDx'] = np.select(conditions, choices, default=np.nan)
    
    # Amnestic status
    df['amnesticStatus'] = np.where(
        (df['AutoDx'] == 1) & ((df['memoryOne_impaired'] == 1) | (df['memoryTwo_impaired'] == 1)),
        "amnestic",
        np.where(df['AutoDx'] == 1, "nonAmnestic", 
                np.where(df['AutoDx'] == 0, "noMCI", np.nan))
    )
    
    # Multiple/Single domain classification
    df['multipleSingle'] = np.where(
        df[['Attention_impaired', 'Memory_impaired', 'Executive_impaired', 
            'Visuospatial_impaired', 'Language_impaired']].sum(axis=1) > 1,
        "Multiple",
        np.where(
            df[['Attention_impaired', 'Memory_impaired', 'Executive_impaired', 
                'Visuospatial_impaired', 'Language_impaired']].sum(axis=1) < 1,
            "None",
            "Single"
        )
    )
    
    # Reliability measure
    def calculate_reliability(row):
        total_cols = len([
            'attentionOne_impaired', 'attentionTwo_impaired',
            'memoryOne_impaired', 'memoryTwo_impaired',
            'execuFuncOne_impaired', 'execuFuncTwo_impaired',
            'visuoSpatOne_impaired', 'visuoSpatTwo_impaired',
            'languageOne_impaired', 'languageTwo_impaired'
        ]) + len(subjective_cols) + len(functional_cols)
        
        non_missing = sum(1 for col in [
            'attentionOne_impaired', 'attentionTwo_impaired',
            'memoryOne_impaired', 'memoryTwo_impaired',
            'execuFuncOne_impaired', 'execuFuncTwo_impaired',
            'visuoSpatOne_impaired', 'visuoSpatTwo_impaired',
            'languageOne_impaired', 'languageTwo_impaired'
        ] + subjective_cols + functional_cols if not pd.isna(row[col]))
        
        return round(100 * (non_missing / total_cols), 1)
    
    df['Reliability'] = df.apply(calculate_reliability, axis=1)
    
    # Multiple/Single domain details
    def get_domain_details(row):
        domains = []
        if not pd.isna(row['Attention_impaired']) and row['Attention_impaired'] >= 1:
            domains.append("Attention")
        if not pd.isna(row['Memory_impaired']) and row['Memory_impaired'] >= 1:
            domains.append("Memory")
        if not pd.isna(row['Executive_impaired']) and row['Executive_impaired'] >= 1:
            domains.append("Executive")
        if not pd.isna(row['Visuospatial_impaired']) and row['Visuospatial_impaired'] >= 1:
            domains.append("Visuospatial")
        if not pd.isna(row['Language_impaired']) and row['Language_impaired'] >= 1:
            domains.append("Language")
        return ", ".join(domains) if domains else ""
    
    df['multipleSingleDomain'] = df.apply(get_domain_details, axis=1)
    
    # Reorder columns
    column_order = [col for col in df.columns if col not in ['NP_impaired', 'Subjective_impaired', 
                                                           'Functional_impaired', 'missingValues', 
                                                           'AutoDx', 'amnesticStatus', 
                                                           'multipleSingle', 'multipleSingleDomain']]
    column_order += ['NP_impaired', 'Subjective_impaired', 'Functional_impaired', 'missingValues',
                   'AutoDx', 'amnesticStatus', 'multipleSingle', 'multipleSingleDomain']
    
    return df[column_order]


# Comorbidity Function
def comorbidityIdentifier(df: pd.DataFrame, type_: str = "broad", base_name: str = "comorb") -> pd.DataFrame:
    # Identify all columns that match the base_name
    comorb_cols = [col for col in df.columns if col.startswith(base_name)]
    
    def get_comorbidities(row):
        unique_comorbs = row[comorb_cols].dropna().unique()
        if len(unique_comorbs) == 0:
            return "None"
        return "; ".join(unique_comorbs)
    
    df['comorbidities'] = df.apply(get_comorbidities, axis=1)
    
    if type_ == "strict":
        df['suggestions'] = np.where(
            df['comorbidities'] == "None",
            "No notable comorbidities.",
            "Refer to specialist."
        )
    elif type_ == "broad":
        df['suggestions'] = np.where(
            df['comorbidities'] == "None",
            "No action needed.",
            "Consider follow-up for: " + df['comorbidities']
        )
    else:
        raise ValueError("Invalid type. Use 'broad' or 'strict'.")
    
    return df