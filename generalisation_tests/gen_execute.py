import gen_preprocess
import gen_variable
import gen_fixed

def main():
    # 1. Load Data
    data_dir = 'mito_data/nets'
    groups, group_files = gen_preprocess.load_groups(data_dir)
    
    if not groups:
        print("No groups loaded. Please ensure 'trial.py' has been run to generate the data.")
        return

    # 2. Run Variable Strategy
    df_var = gen_variable.run_strategy(groups, group_files)
    df_var.to_csv('gen_results_variable.csv', index=False)
    print("Variable strategy complete. Saved to 'gen_results_variable.csv'.")

    # 3. Run Fixed Strategy
    df_fixed = gen_fixed.run_strategy(groups, group_files)
    df_fixed.to_csv('gen_results_fixed.csv', index=False)
    print("Fixed strategy complete. Saved to 'gen_results_fixed.csv'.")

if __name__ == "__main__":
    main()