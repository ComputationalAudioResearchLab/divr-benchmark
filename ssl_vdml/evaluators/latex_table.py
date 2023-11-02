import pandas as pd
def csv_to_latex_table(csv_path):
    """
    Convert a given CSV to a LaTeX table.
    
    Args:
    - csv_path (str): Path to the CSV file.
    
    Returns:
    - str: LaTeX table string.
    """
    # Read the CSV
    df = pd.read_csv('/home/workspace/ssl_vdml/evaluators/summrized.csv')
    
    # Convert the DataFrame to LaTeX
    latex_table = df.to_latex(index=False, float_format="%.2f", 
                              caption="Summary of Model Accuracies and Deviations", 
                              label="tab:summary")
    
    return latex_table

# Generate the LaTeX table for the provided CSV
latex_output = csv_to_latex_table('/home/workspace/ssl_vdml/evaluators/latex.txt')
print(latex_output)
