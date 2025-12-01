import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import csv
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from collections import defaultdict

# Initialize BLIP processor and model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Directory wimaghere images are stored
image_folder = 'ship_images/'  # Ensure this is the correct folder path
output_csv = 'vqa_results.csv'  # Output CSV for storing results

# Your questions
questions = [
    "What is the name of the ship in this image?", 
    "Which country does this ship in this image belong to?", 
    "What type of ship is shown in this image?", 
    "What is the ship's hull number?"
]

# Initialize dictionaries for tracking results
accuracy_by_type = defaultdict(int)
accuracy_by_country = defaultdict(int)
accuracy_by_resize = defaultdict(int)
accuracy_by_prompt = defaultdict(int)
all_labels = []
all_preds = []

# Helper function to calculate metrics (precision, recall, F1)
def calculate_metrics(labels, preds):
    precision = precision_score(labels, preds, average='macro', zero_division=0)
    recall = recall_score(labels, preds, average='macro', zero_division=0)
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    return precision, recall, f1

# Open the CSV file for writing results
with open(output_csv, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    header = ['image_name'] + questions + ['type', 'country', 'resize_factor', 'prompt_method']
    csvwriter.writerow(header)

    # Loop through image files
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        
        if image_name.lower().endswith('.jpg') or image_name.lower().endswith('.jpeg'):  # Ensuring it's an image file
            row = [image_name]
            
            # Open image and convert it to RGB (safer image loading)
            try:
                image = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"Error processing {image_name}: {e}")
                continue

            # Get ground-truth labels (can be customized as per your data)
            type_label = 'Destroyer'  # Example, you will replace this with actual data
            country_label = 'USA'     # Example, you will replace this with actual data

            all_labels.append((type_label, country_label))

            # Loop through questions, resize factors, and prompt methods
            for resize_factor in [0.5, 0.25, 0.125]:
                resized_image = image.resize((int(image.width * resize_factor), int(image.height * resize_factor)))

                for prompt_method in ["genfill", "genfill_prompt", "manual"]:
                    try:
                        # Prepare inputs for the model
                        inputs = processor(images=resized_image, text=questions[0], return_tensors="pt")  # Using the first question as example
                        out = blip_model.generate(**inputs)
                        answer = processor.decode(out[0], skip_special_tokens=True)

                        row.append(answer)

                        # Accuracy tracking by type, country, resize factor, and prompt method
                        if prompt_method == "genfill":
                            accuracy_by_prompt['genfill'] += (answer == type_label)
                        elif prompt_method == "genfill_prompt":
                            accuracy_by_prompt['genfill_prompt'] += (answer == country_label)
                        else:
                            accuracy_by_prompt['manual'] += (answer == type_label)

                        # Store accuracy and predictions for final evaluation
                        all_preds.append(answer)

                        # Write results to CSV
                        csvwriter.writerow(row)

                    except Exception as e:
                        print(f"Error processing {image_name} for {question}: {e}")
                        row.append('ERROR')
                        csvwriter.writerow(row)
                        continue

    print("Done! Results saved to", output_csv)

# After looping through all the images, calculate precision, recall, and F1 scores
precision, recall, f1 = calculate_metrics(all_labels, all_preds)

# Print the final metrics
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Aggregated metrics for each category
df = pd.read_csv(output_csv)
type_accuracy = df.groupby('type')['accuracy'].mean()
country_accuracy = df.groupby('country')['accuracy'].mean()
resize_accuracy = df.groupby('resize_factor')['accuracy'].mean()
prompt_accuracy = df.groupby('prompt_method')['accuracy'].mean()

# Save aggregated metrics to Excel
with pd.ExcelWriter('vqa_metrics.xlsx') as writer:
    type_accuracy.to_excel(writer, sheet_name='Ship Type Accuracy')
    country_accuracy.to_excel(writer, sheet_name='Country Accuracy')
    resize_accuracy.to_excel(writer, sheet_name='Resize Accuracy')
    prompt_accuracy.to_excel(writer, sheet_name='Prompt Accuracy')

# Additional summary metrics (precision, recall, f1)
summary_df = pd.DataFrame({
    'Metric': ['Precision', 'Recall', 'F1 Score'],
    'Value': [precision, recall, f1]
})

# Save to Excel
with pd.ExcelWriter('vqa_metrics.xlsx', mode='a') as writer:
    summary_df.to_excel(writer, sheet_name='Summary Metrics')

print("Aggregated results and metrics saved to vqa_metrics.xlsx")
