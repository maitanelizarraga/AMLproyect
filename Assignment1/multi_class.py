def create_multiclass_target(df):
    # 0: No fraud, 1: fraud type A (low amount), 2: Fraud type B (high amount)
    conditions = [
        (df['fraud_label'] == 0),
        (df['fraud_label'] == 1) & (df['transaction_amount'] < 500),
        (df['fraud_label'] == 1) & (df['transaction_amount'] >= 500)
    ]
    choices = [0, 1, 2]
    df['fraud_category'] = np.select(conditions, choices, default=0)
    return df

def main():
    df = pd.read_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_imputed.csv")
    df_multi = create_multiclass_target(df)
    df_multi.to_csv("./Assignment1/datasets/Digital_Payment_Fraud_Detection_Dataset_multiclass.csv", index=False)
    print("Multiclass target created and saved to: datasets/Digital_Payment_Fraud_Detection_Dataset_multiclass.csv")