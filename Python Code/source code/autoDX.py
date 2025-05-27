import numpy as np
domains = ['Attention_impaired', 'Memory_impaired', 'Executive_impaired', 'Visuospatial_impaired', 'Language_impaired']


def PD_MCIx(df, neuropsych_domains, subjective_responses, functional_responses):
    """
    Classify MCI based on neuropsychological tests, subjective responses, and functional responses.

    Parameters
    ----------
    df : pd.DataFrame
        The patient-level dataframe containing raw test scores and responses.

    neuropsych_domains : list
        List where values are
        lists of tuples of the form (column_name, cutoff). There should be 10 tests total, 2 from each possible domain.

        Example:
            [("mem_test1", 20), ("mem_test2", 15), ("attn1", 12), ("attn2", 10), ...]

    subjective_responses : dict
        Dictionary of subjective input configurations. Keys can be 1 to 4.
        Each value must be a dict with:
            - 'col': Column name in df
            - 'type': one of {"Cutoff", "Series", "Binary"}
            - 'val': Value for the criteria type

    functional_responses : dict
        Same format as subjective_responses, but for functional questions (1 to 4).

    Returns
    -------
    pd.DataFrame
        Modified DataFrame with new impairment flags, missing value messages,
        AutoDx classification, and amnestic subtyping.
    """
    # --- Neuropsychological assessment ---
    df['tests_impaired'] = sum(df[col] < cutoff for col, cutoff in neuropsych_domains) >= 2


    # --- Subjective responses ---
    impairment_cols = []

    for key, config in subjective_responses.items():
        t = config['type']
        colname = f'Subjective{key}_impairment'

        if t == 'standalone':
            df[colname] = (df[config['col']] <= config['val']).astype(int)
        elif t == 'series':
            df[colname] = df[config['col']].isin(config['val']).astype(int)
        elif t == 'OGSame':
            df[colname] = df[config['val']]

        impairment_cols.append(colname)

    # Combine all subjective impairments
    df['Subjective_impaired'] = df[impairment_cols].any(axis=1).astype(int)


    # --- Functional responses ---
    impairment_cols = []

    for key, config in subjective_responses.items():
        t = config['type']
        colname = f'Subjective{key}_impairment'

        if t == 'standalone':
            df[colname] = (df[config['col']] <= config['val']).astype(int)
        elif t == 'series':
            df[colname] = df[config['col']].isin(config['val']).astype(int)
        elif t == 'OGSame':
            df[colname] = df[config['val']]

        impairment_cols.append(colname)

    # Combine all subjective impairments
    df['Subjective_impaired'] = df[impairment_cols].any(axis=1).astype(int)



    # --- Missing values check ---
    df['missingValues'] = df.apply(
        lambda row: f"Patient is missing values in {', '.join(row[row.isna()].index)}"
        if row.isna().any() else "No missing values", axis=1)

    # --- AutoDx logic ---
    df['AutoDx'] = np.where((df['NP_impaired'] == 0) | (df['Functional_impaired'] == 1), 0, 1)


    # --- Amnestic subtyping ---
    memory1_col = neuropsych_domains['Memory'][0][0]
    memory2_col = neuropsych_domains['Memory'][1][0]
    memory1_imp = f"memory1_impaired"
    memory2_imp = f"memory2_impaired"

    df['amnesticStatus'] = np.where(
        (df['AutoDx'] == 1) & ((df[memory1_imp] == 1) | (df[memory2_imp] == 1)), 'amnestic',
        np.where(df['AutoDx'] == 1, 'nonAmnestic', 'noMCI')
    )

    # --- Column reordering ---
    first_cols = ['AutoDx', 'NP_impaired', 'Subjective_impaired', 'Functional_impaired']
    remaining_cols = [col for col in df.columns if col not in first_cols]
    df = df[first_cols + remaining_cols]

    return df
