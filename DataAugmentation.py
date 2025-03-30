import pandas as pd
import time
import os
import random
import nlpaug.augmenter.word as naw
import spacy
from tqdm import tqdm

# Load spaCy model for similarity calculation
nlp = spacy.load("en_core_web_md")

# Initialize BERT for contextual word embeddings-based augmentation using nlpaug
augmenter = naw.ContextualWordEmbsAug(
    model_path='bert-base-uncased',
    action='substitute',
    top_k=50  # Increases the pool of possible substitutions
)

# Function to calculate semantic similarity using spaCy
def calculate_similarity(text1, text2):
    doc1 = nlp(text1)
    doc2 = nlp(text2)
    return doc1.similarity(doc2)

# Function to augment text with validation
def augment_with_validation(text, existing_texts, num_augmentations=10, max_attempts_per_text=4, similarity_threshold=0.85):
    unique_augmentations = set()
    attempts = 0

    while len(unique_augmentations) < num_augmentations and attempts < num_augmentations * max_attempts_per_text:
        augmented_texts = augmenter.augment(text)

        # Ensure it's a list
        if not isinstance(augmented_texts, list):
            augmented_texts = [augmented_texts]  # Ensure it's always a list

        for augmented_text in augmented_texts:
            # Calculate similarity between original and augmented text
            similarity_score = calculate_similarity(text, augmented_text)

            # Validate semantic similarity
            if similarity_score >= similarity_threshold and augmented_text not in unique_augmentations and augmented_text not in existing_texts:
                unique_augmentations.add(augmented_text)

        attempts += 1
        if attempts % 5 == 0:  # Print progress every 5 attempts
            print(f"Attempt {attempts}: {len(unique_augmentations)} augmentations found so far.")

        if attempts > 50:  # Break if we've made too many attempts without enough unique augmentations
            print(f"Too many attempts made without enough unique augmentations.")
            break

    return list(unique_augmentations)

# Function to sanitize file names (replace spaces and backslashes with underscores)
def sanitize_filename(filename):
    return filename.replace(" ", "").replace("\\", "").replace("/", "_")


# Function to augment data for a range of groups
def augment_for_range(min_groups=1, max_groups=10, file_path_csv = 'total.csv'):
    # Load the input CSV file path
    data = pd.read_csv(file_path_csv)

    # Strip spaces and check for any column name issues
    data.columns = data.columns.str.strip()

    # Sort the data by the four columns in decreasing order
    sorted_data = data.sort_values(by=['product-category', 'hazard-category', 'product', 'hazard'], ascending=False)

    # Add a 'count' column that counts occurrences of each unique combination of product-category, hazard-category, product, and hazard
    sorted_data['count'] = sorted_data.groupby(['product-category', 'hazard-category', 'product', 'hazard'])['text'].transform('count')

    # Sort the data by 'count' (ascending order to start with the lowest occurrences)
    sorted_data = sorted_data.sort_values(by='count', ascending=True)

    # Group by the unique combinations of the four columns
    grouped = sorted_data.groupby(['product-category', 'hazard-category', 'product', 'hazard'])

    # Select the groups within the custom range
    groups_list = grouped.size().index
    selected_groups = groups_list[min_groups-1:max_groups]  # Slice the list based on the input range

    # Initialize a list to hold all augmented data
    all_augmented_rows = []

    for (product_category, hazard_category, product, hazard) in selected_groups:
        group = grouped.get_group((product_category, hazard_category, product, hazard))
        print(f"Processing combination: {product_category}, {hazard_category}, {product}, {hazard}")

        generated_rows = []
        existing_texts = set(group['text'])  # Avoid duplication of text

        # Generate exactly 50 augmentations for each group
        num_to_generate = 50  # Fixed number of augmentations

        while len(generated_rows) < num_to_generate:
            for idx, row in tqdm(group.iterrows(), total=len(group), desc=f"Augmenting {product_category}, {hazard_category}, {product}, {hazard}"):
                original_text = row['text']

                # Get augmentations for the current text
                augmented_texts = augment_with_validation(original_text, existing_texts, num_augmentations=num_to_generate - len(generated_rows), max_attempts_per_text=3)

                for augmented_text in augmented_texts:
                    new_row = row.copy()
                    new_row['text'] = augmented_text
                    generated_rows.append(new_row)
                    existing_texts.add(augmented_text)

                # Sleep every 5 rows to avoid hitting rate limits or to slow down processing
                if len(generated_rows) % 5 == 0:
                    time.sleep(2)  # 1 second delay after every 5 rows

                # Stop early if we have enough augmentations
                if len(generated_rows) >= num_to_generate:
                    break

        # If we have generated rows, save them to a file
        if generated_rows:
            augmented_df = pd.DataFrame(generated_rows)
            # Create a directory for saving individual CSV files
            os.makedirs('all_results', exist_ok=True)

            # Define the CSV file name based on the combination
            file_name = f"{product_category}{hazard_category}{product}_{hazard}.csv"
            # Sanitize the file name (replace spaces and backslashes with underscores)
            sanitized_file_name = sanitize_filename(file_name)

            sanitized_file_name = f"{sanitized_file_name}"

            # Save this specific augmented data to a CSV file
            augmented_df.to_csv(sanitized_file_name, index=False, quoting=1)  # quoting=1 ensures the text is correctly quoted
            print(f"Augmented data for {product_category}, {hazard_category}, {product}, {hazard} saved to: {sanitized_file_name}")
            
            # Add the generated rows to the cumulative list
            all_augmented_rows.extend(generated_rows)
        else:
            print(f"No augmentations generated for {product_category}, {hazard_category}, {product}, {hazard}")

    # Save the cumulative augmented data to a single file
    if all_augmented_rows:
        cumulative_df = pd.DataFrame(all_augmented_rows)
        os.makedirs('all_results', exist_ok=True)  # Ensure the output directory exists
        cumulative_file_name = 'cumulative_augmented_data.csv'
        # Sanitize the cumulative file name (replace spaces and backslashes with underscores)
        # sanitized_cumulative_file_name = sanitize_filename(cumulative_file_name)
        cumulative_df.to_csv(cumulative_file_name, index=False, quoting=1)
        print(f"Cumulative augmented data saved to: {cumulative_file_name}")

# Call the function to augment data for a custom range of groups
min_groups = 0 # Start from group 1
max_groups = 100  # End at group 5 (for example)
file_path_csv = 'final_cleaned_train.csv'  # Modify with your file path
augment_for_range(min_groups, max_groups, file_path_csv)




# Load the original and new files
original_file = 'final_cleaned_train.csv'  # Replace with your file path
new_file = 'total.csv'  # Replace with your file path

# Read the original and new CSV files into DataFrames
original_df = pd.read_csv(original_file)
new_df = pd.read_csv(new_file)

# Remove the last column of the new DataFrame
new_df = new_df.iloc[:, :-1]

# Check if both DataFrames have the same columns (this is an optional check)
if original_df.columns.tolist() == new_df.columns.tolist():
    # Append the rows from the new DataFrame to the original DataFrame
    combined_df = original_df.append(new_df, ignore_index=True)
else:
    print("Columns in the original file and the new file do not match.")

# Save the combined DataFrame back to a CSV file
combined_df.to_csv('data.csv', index=False)

print("Files have been successfully combined and saved.")


