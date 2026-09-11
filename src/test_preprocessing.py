from data_loader import load_cicids2017
from preprocessing import clean_data


# Load the CICIDS2017 dataset
df = load_cicids2017()

# Clean the dataset
cleaned_df = clean_data(df)