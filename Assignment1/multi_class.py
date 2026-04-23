def create_multiclass_target(df):
    # Definimos condiciones para tipos de fraude
    # 0: No Fraude, 1: Fraude tipo A (bajo monto), 2: Fraude tipo B (alto monto)
    conditions = [
        (df['fraud_label'] == 0),
        (df['fraud_label'] == 1) & (df['transaction_amount'] < 500),
        (df['fraud_label'] == 1) & (df['transaction_amount'] >= 500)
    ]
    choices = [0, 1, 2]
    df['fraud_category'] = np.select(conditions, choices, default=0)
    return df