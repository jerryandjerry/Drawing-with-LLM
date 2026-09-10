import json
import csv

# Open and load the JSON file
with open('annotations_trainval2017/annotations/captions_val2017.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Extract all captions one by one
captions = []
for annotation in data['annotations']:
    caption = annotation['caption']
    # Replace all quotes with empty string
    caption = caption.replace('"', '').replace("'", '').replace('\u201c', '').replace('\u201d', '').replace('\u2018', '').replace('\u2019', '')
    # Convert to lowercase
    caption = caption.lower()
    # Remove period at the end if present
    if caption.endswith('.'):
        caption = caption[:-1]
    # Strip any extra whitespace
    caption = caption.strip()
    captions.append(caption)

# Save captions to CSV file
with open('captions_val2017.csv', 'w', newline='', encoding='utf-8') as csvfile:
    writer = csv.writer(csvfile, quoting=csv.QUOTE_NONE, escapechar='\\')
    writer.writerow(['index', 'caption'])  # Header
    for i, caption in enumerate(captions, 1):
        writer.writerow([i, caption])

print(f"Total captions extracted: {len(captions)}")
print(f"Captions saved to: captions_val2017.csv")

