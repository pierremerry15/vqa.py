import torch
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import csv
import pandas as pd
from collections import defaultdict

# Initialize BLIP processor and model for VQA
print("Loading BLIP model...")
from transformers import BlipForQuestionAnswering
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
blip_model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
device = "cuda" if torch.cuda.is_available() else "cpu"
blip_model = blip_model.to(device)
print(f"Using device: {device}")

# Base directory for images
base_folder = 'VQA_Ship_Images./TADMUS Image Augmentation/'

# Define resize factors and their corresponding folders
resize_configs = [
    (0.125, '0.125_factor_resized-ship-set-copy', 'Genfill'),
    (0.125, '0.125_ps_factor_resized-ship-set-copy', 'Manual'),
    (0.25, '0.25_factor_resized-ship-set-copy', 'Genfill'),
    (0.25, '0.25_ps_factor_resized-ship-set-copy', 'Manual'),
    (0.5, '0.5_factor_resized-ship-set-copy', 'Genfill'),
    (0.5, '0.5_ps_factor_resized-ship-set-copy', 'Manual'),
]

# Questions for VQA
questions = {
    'ship_name': "What is the name of the ship?",
    'country': "Which country does this ship belong to?",
    'ship_type': "What type of ship is this?",
    'hull_number': "What is the hull number?"
}

# Parse filename to extract ground truth
def parse_filename(filename):
    """
    Parse filename format: Country_ShipType_ShipName(HullNumber)_Index.jpeg
    Example: USA_Destroyer_USS Forrest Sherman (DDG-98)_0.jpeg
    """
    try:
        # Remove extension
        name = filename.replace('.jpeg', '').replace('.jpg', '')
        
        # Split by underscore
        parts = name.split('_')
        
        if len(parts) < 3:
            return None
        
        country = parts[0].strip()
        ship_type = parts[1].strip()
        
        # Join remaining parts for ship name (may contain underscores)
        ship_name_part = '_'.join(parts[2:])
        
        # Extract hull number if present (in parentheses)
        hull_number = ''
        ship_name = ship_name_part
        
        if '(' in ship_name_part and ')' in ship_name_part:
            start = ship_name_part.index('(')
            end = ship_name_part.index(')')
            hull_number = ship_name_part[start+1:end].strip()
            ship_name = ship_name_part[:start].strip()
            # Remove trailing index after parentheses
            remaining = ship_name_part[end+1:]
            if remaining and remaining.startswith('_'):
                pass  # This is just the index, ignore it
        
        return {
            'country': country,
            'ship_type': ship_type,
            'ship_name': ship_name,
            'hull_number': hull_number
        }
    except Exception as e:
        print(f"Error parsing filename {filename}: {e}")
        return None

# Store all results
all_results = []

print("\nProcessing images...")
total_images = 0

# Process each configuration
for resize_factor, folder_name, prompt_method in resize_configs:
    folder_path = os.path.join(base_folder, folder_name)
    
    if not os.path.exists(folder_path):
        print(f"Folder not found: {folder_path}")
        continue
    
    print(f"\nProcessing: {folder_name} (Resize: {resize_factor}, Method: {prompt_method})")
    
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]
    
    for idx, image_name in enumerate(image_files):
        if idx % 10 == 0:
            print(f"  Processing image {idx+1}/{len(image_files)}...")
        
        # Parse ground truth from filename
        ground_truth = parse_filename(image_name)
        if not ground_truth:
            continue
        
        image_path = os.path.join(folder_path, image_name)
        
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_name}: {e}")
            continue
        
        # Ask each question
        result = {
            'image_name': image_name,
            'resize_factor': resize_factor,
            'prompt_method': prompt_method,
            'gt_country': ground_truth['country'],
            'gt_ship_type': ground_truth['ship_type'],
            'gt_ship_name': ground_truth['ship_name'],
            'gt_hull_number': ground_truth['hull_number']
        }
        
        for q_key, question in questions.items():
            try:
                # Process inputs properly for VQA
                inputs = processor(image, question, return_tensors="pt")
                # Move inputs to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                out = blip_model.generate(**inputs, max_length=50)
                answer = processor.decode(out[0], skip_special_tokens=True)
                
                # Debug: print first few answers
                if idx == 0:
                    print(f"    Q: {question}")
                    print(f"    A: {answer}")
                
                result[f'pred_{q_key}'] = answer
            except Exception as e:
                print(f"Error processing question '{question}' for {image_name}: {e}")
                import traceback
                traceback.print_exc()
                result[f'pred_{q_key}'] = 'ERROR'
        
        all_results.append(result)
        total_images += 1

print(f"\nTotal images processed: {total_images}")

# Save raw results to CSV
df = pd.DataFrame(all_results)
df.to_csv('vqa_results.csv', index=False)
print("Raw results saved to vqa_results.csv")

# Calculate accuracy metrics
def check_match(pred, gt, fuzzy=True):
    """Check if prediction matches ground truth"""
    if pd.isna(pred) or pd.isna(gt) or pred == 'ERROR':
        return False
    
    pred_clean = str(pred).lower().strip()
    gt_clean = str(gt).lower().strip()
    
    if fuzzy:
        # Check if key terms match
        return gt_clean in pred_clean or pred_clean in gt_clean
    else:
        return pred_clean == gt_clean

# Add correctness columns
df['correct_country'] = df.apply(lambda row: check_match(row['pred_country'], row['gt_country']), axis=1)
df['correct_ship_type'] = df.apply(lambda row: check_match(row['pred_ship_type'], row['gt_ship_type']), axis=1)
df['correct_ship_name'] = df.apply(lambda row: check_match(row['pred_ship_name'], row['gt_ship_name']), axis=1)
df['correct_hull_number'] = df.apply(lambda row: check_match(row['pred_hull_number'], row['gt_hull_number']), axis=1)

print("\n" + "="*80)
print("ANALYSIS RESULTS")
print("="*80)

# (1) Overall accuracy by answer type
print("\n1. OVERALL ACCURACY BY ANSWER TYPE:")
print("-" * 50)
for answer_type in ['country', 'ship_type', 'ship_name', 'hull_number']:
    accuracy = df[f'correct_{answer_type}'].mean() * 100
    print(f"  {answer_type.replace('_', ' ').title()}: {accuracy:.2f}%")

# (2a) Accuracy by ship type
print("\n2a. ACCURACY BY SHIP TYPE:")
print("-" * 50)
ship_type_accuracy = df.groupby('gt_ship_type').agg({
    'correct_country': 'mean',
    'correct_ship_type': 'mean',
    'correct_ship_name': 'mean',
    'correct_hull_number': 'mean'
}).round(4) * 100

for ship_type in ship_type_accuracy.index:
    print(f"\n  {ship_type}:")
    for col in ship_type_accuracy.columns:
        answer_type = col.replace('correct_', '').replace('_', ' ').title()
        print(f"    {answer_type}: {ship_type_accuracy.loc[ship_type, col]:.2f}%")

# (2b) Accuracy by country
print("\n2b. ACCURACY BY COUNTRY:")
print("-" * 50)
country_accuracy = df.groupby('gt_country').agg({
    'correct_country': 'mean',
    'correct_ship_type': 'mean',
    'correct_ship_name': 'mean',
    'correct_hull_number': 'mean'
}).round(4) * 100

for country in country_accuracy.index:
    print(f"\n  {country}:")
    for col in country_accuracy.columns:
        answer_type = col.replace('correct_', '').replace('_', ' ').title()
        print(f"    {answer_type}: {country_accuracy.loc[country, col]:.2f}%")

# (3) Accuracy by resize factor
print("\n3. ACCURACY BY RESIZE FACTOR:")
print("-" * 50)
resize_accuracy = df.groupby('resize_factor').agg({
    'correct_country': 'mean',
    'correct_ship_type': 'mean',
    'correct_ship_name': 'mean',
    'correct_hull_number': 'mean'
}).round(4) * 100

for resize in resize_accuracy.index:
    print(f"\n  Resize Factor {resize}:")
    for col in resize_accuracy.columns:
        answer_type = col.replace('correct_', '').replace('_', ' ').title()
        print(f"    {answer_type}: {resize_accuracy.loc[resize, col]:.2f}%")

# (3a) Accuracy by ship type for each resize factor
print("\n3a. ACCURACY BY SHIP TYPE FOR EACH RESIZE FACTOR:")
print("-" * 50)
for resize in sorted(df['resize_factor'].unique()):
    print(f"\n  Resize Factor {resize}:")
    resize_df = df[df['resize_factor'] == resize]
    ship_type_resize = resize_df.groupby('gt_ship_type').agg({
        'correct_country': 'mean',
        'correct_ship_type': 'mean',
        'correct_ship_name': 'mean',
        'correct_hull_number': 'mean'
    }).round(4) * 100
    
    for ship_type in ship_type_resize.index:
        print(f"    {ship_type}:")
        for col in ship_type_resize.columns:
            answer_type = col.replace('correct_', '').replace('_', ' ').title()
            print(f"      {answer_type}: {ship_type_resize.loc[ship_type, col]:.2f}%")

# (3b) Accuracy by country for each resize factor
print("\n3b. ACCURACY BY COUNTRY FOR EACH RESIZE FACTOR:")
print("-" * 50)
for resize in sorted(df['resize_factor'].unique()):
    print(f"\n  Resize Factor {resize}:")
    resize_df = df[df['resize_factor'] == resize]
    country_resize = resize_df.groupby('gt_country').agg({
        'correct_country': 'mean',
        'correct_ship_type': 'mean',
        'correct_ship_name': 'mean',
        'correct_hull_number': 'mean'
    }).round(4) * 100
    
    for country in country_resize.index:
        print(f"    {country}:")
        for col in country_resize.columns:
            answer_type = col.replace('correct_', '').replace('_', ' ').title()
            print(f"      {answer_type}: {country_resize.loc[country, col]:.2f}%")

# (4) Accuracy by prompt method
print("\n4. ACCURACY BY PROMPT METHOD:")
print("-" * 50)
prompt_accuracy = df.groupby('prompt_method').agg({
    'correct_country': 'mean',
    'correct_ship_type': 'mean',
    'correct_ship_name': 'mean',
    'correct_hull_number': 'mean'
}).round(4) * 100

for method in prompt_accuracy.index:
    print(f"\n  {method}:")
    for col in prompt_accuracy.columns:
        answer_type = col.replace('correct_', '').replace('_', ' ').title()
        print(f"    {answer_type}: {prompt_accuracy.loc[method, col]:.2f}%")

# (4a) Accuracy by ship type for each prompt method
print("\n4a. ACCURACY BY SHIP TYPE FOR EACH PROMPT METHOD:")
print("-" * 50)
for method in sorted(df['prompt_method'].unique()):
    print(f"\n  {method}:")
    method_df = df[df['prompt_method'] == method]
    ship_type_method = method_df.groupby('gt_ship_type').agg({
        'correct_country': 'mean',
        'correct_ship_type': 'mean',
        'correct_ship_name': 'mean',
        'correct_hull_number': 'mean'
    }).round(4) * 100
    
    for ship_type in ship_type_method.index:
        print(f"    {ship_type}:")
        for col in ship_type_method.columns:
            answer_type = col.replace('correct_', '').replace('_', ' ').title()
            print(f"      {answer_type}: {ship_type_method.loc[ship_type, col]:.2f}%")

# (4b) Accuracy by country for each prompt method
print("\n4b. ACCURACY BY COUNTRY FOR EACH PROMPT METHOD:")
print("-" * 50)
for method in sorted(df['prompt_method'].unique()):
    print(f"\n  {method}:")
    method_df = df[df['prompt_method'] == method]
    country_method = method_df.groupby('gt_country').agg({
        'correct_country': 'mean',
        'correct_ship_type': 'mean',
        'correct_ship_name': 'mean',
        'correct_hull_number': 'mean'
    }).round(4) * 100
    
    for country in country_method.index:
        print(f"    {country}:")
        for col in country_method.columns:
            answer_type = col.replace('correct_', '').replace('_', ' ').title()
            print(f"      {answer_type}: {country_method.loc[country, col]:.2f}%")

# Save detailed results to Excel
print("\n" + "="*80)
print("Saving detailed results to Excel...")

with pd.ExcelWriter('vqa_detailed_results.xlsx', engine='openpyxl') as writer:
    # Raw results
    df.to_excel(writer, sheet_name='Raw Results', index=False)
    
    # Overall accuracy
    overall_acc = pd.DataFrame({
        'Answer Type': ['Country', 'Ship Type', 'Ship Name', 'Hull Number'],
        'Accuracy (%)': [
            df['correct_country'].mean() * 100,
            df['correct_ship_type'].mean() * 100,
            df['correct_ship_name'].mean() * 100,
            df['correct_hull_number'].mean() * 100
        ]
    })
    overall_acc.to_excel(writer, sheet_name='Overall Accuracy', index=False)
    
    # By ship type
    ship_type_accuracy.to_excel(writer, sheet_name='By Ship Type')
    
    # By country
    country_accuracy.to_excel(writer, sheet_name='By Country')
    
    # By resize factor
    resize_accuracy.to_excel(writer, sheet_name='By Resize Factor')
    
    # By prompt method
    prompt_accuracy.to_excel(writer, sheet_name='By Prompt Method')

print("Detailed results saved to vqa_detailed_results.xlsx")
print("\n" + "="*80)
print("ANALYSIS COMPLETE!")
print("="*80)
