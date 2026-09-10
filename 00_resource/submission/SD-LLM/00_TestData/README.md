# GPT-5 SVG Generation and Evaluation

This directory contains scripts to generate SVG images using GPT-5 (or other LLMs) and evaluate them using the same competition metric as the diffusion model approach.

## Files

- **`llm_tool.py`**: LLM wrapper that supports multiple models (GPT-5, Claude, Gemini, DeepSeek, etc.)
- **`gpt5_svg_generator.py`**: Python script for batch SVG generation and evaluation
- **`gpt5_svg_evaluation.ipynb`**: Jupyter notebook for interactive generation and evaluation
- **`README.md`**: This file

## Setup

### 1. Install Requirements

Make sure you have all required packages installed (from parent directory):

```bash
pip install openai pandas numpy matplotlib jupyter tqdm
```

### 2. Configure API Keys

Edit `llm_tool.py` and add your API keys:

```python
self.models = {
    "gpt-5": {
        "api_key": "your-openai-api-key-here",  # ← Add your key
        "base_url": "https://api.openai.com/v1",
        "model_name": "gpt-5-2025-08-07",
        "provider": "openai"
    },
    # ... other models
}
```

### 3. Ensure Parent Directory Structure

The scripts expect the following files in the parent directory:
- `train.csv` - Training data with descriptions
- `questions.parquet` - Pre-defined Q&A for evaluation
- `metric.py` - Evaluation metric implementation

## Usage

### Option 1: Python Script (Automated)

Run the automated script:

```bash
cd 00_LLM_refinement
python gpt5_svg_generator.py
```

This will:
1. Load training data
2. Generate SVGs for all samples using GPT-5
3. Evaluate using competition metric
4. Save results to `gpt5_results.csv`

### Option 2: Jupyter Notebook (Interactive)

For interactive experimentation:

```bash
cd 00_LLM_refinement
jupyter notebook gpt5_svg_evaluation.ipynb
```

The notebook allows you to:
- Test single generations
- Visualize SVGs
- Analyze score distributions
- Compare with diffusion model results

## Code Structure

### GPT5SVGModel Class

```python
from gpt5_svg_generator import GPT5SVGModel

# Initialize model
model = GPT5SVGModel(model_name="gpt-5", max_retries=3)

# Generate SVG from description
description = "a purple forest at dusk"
svg_code = model.predict(description)
```

### Evaluation Process

The evaluation uses the **same metric** as the diffusion model:

```python
import metric

# Score = harmonic_mean(VQA, Aesthetic, β=0.5) × OCR
score = metric.score_instance(multiple_choice_qa, svg, random_seed=42)

# Returns dict:
# {
#     'competition_score': 0.75,
#     'vqa_score': 0.92,
#     'ocr_score': 1.0,
#     'aesthetic_score': 0.65
# }
```

**Key differences from diffusion approach:**
- ❌ No bitmap generation (SVG created directly)
- ❌ No spaCy Q&A generation (uses pre-defined Q&A)
- ❌ No bitmap→SVG conversion
- ✅ Direct text→SVG via LLM
- ✅ Same evaluation metric
- ✅ Same pre-defined Q&A from `questions.parquet`

## Workflow Comparison

### Diffusion Model Workflow
```
Text → Enhance Prompt → Generate 9 Bitmaps → Score Each → Pick Best → Convert to SVG
```

### GPT-5 LLM Workflow
```
Text → LLM Prompt → Generate SVG → Validate → Score → Done
```

## Using Different LLMs

The `llm_tool.py` supports multiple models. To use a different model:

### In Python Script:
```python
# Use Claude instead of GPT-5
model = GPT5SVGModel(model_name="claude-opus", max_retries=3)

# Or Gemini
model = GPT5SVGModel(model_name="gemini-pro", max_retries=3)

# Or DeepSeek
model = GPT5SVGModel(model_name="deepseek-v3", max_retries=3)
```

### Available Models:
- `gpt-5` - OpenAI GPT-5
- `claude-opus` - Anthropic Claude Opus 4
- `gemini-pro` - Google Gemini 2.5 Pro
- `gemini-flash` - Google Gemini 2.5 Flash
- `qwen-plus` - Alibaba Qwen Plus
- `deepseek-v3` - DeepSeek V3
- `deepseek-r1` - DeepSeek R1 (with reasoning)
- `glm-4.5` - GLM 4.5

## Prompt Engineering

The SVG generation uses carefully crafted prompts:

### System Prompt:
```
You are an expert SVG graphic designer. Your task is to create beautiful, 
accurate SVG code based on text descriptions.

Requirements:
1. SVG must be exactly 384x384 pixels with viewBox="0 0 384 384"
2. Total SVG code length must be under 10,000 characters
3. Use clean, simple geometric shapes (prefer polygons over complex paths)
4. Do NOT include any text, letters, or numbers in the image
5. Use vibrant, appropriate colors
6. Create a vector art style illustration
7. Output ONLY the SVG code, no explanation
```

### User Prompt Template:
```
Create an SVG illustration for: "{description}"

Style guidelines:
- Vector art aesthetic with clean shapes
- Watercolor-inspired color palette
- Vibrant and well-composed
- No text or lettering whatsoever
- Maximum 10,000 characters total

Output the complete SVG code:
```

You can modify these prompts in `gpt5_svg_generator.py` to experiment with different styles.

## Output Files

### `gpt5_results.csv`
Contains evaluation results:
```csv
id,description,svg,competition_score,vqa_score,ocr_score,aesthetic_score
02d892,"a purple forest at dusk","<svg...",0.7532,0.9234,1.0,0.6521
...
```

### `gpt5_submission.csv`
Competition submission format:
```csv
id,svg
02d892,"<svg width='384' height='384'..."
...
```

## Performance Notes

### Speed
- **GPT-5**: ~2-5 seconds per SVG (API dependent)
- **Diffusion**: ~2-3 minutes per description (9 candidates)

### Quality
- GPT-5 may struggle with complex spatial relationships
- Diffusion models better at photorealistic rendering
- LLMs better at following precise geometric constraints
- LLMs perfect for size constraints (always under 10KB)

### Cost
- GPT-5: ~$0.01-0.05 per image (API costs)
- Diffusion: Free (local GPU)

## Troubleshooting

### API Key Errors
```
Error: Invalid API key
```
**Solution**: Add your API key to `llm_tool.py`

### Import Errors
```
ModuleNotFoundError: No module named 'metric'
```
**Solution**: Ensure parent directory is in Python path and `metric.py` exists

### SVG Validation Errors
```
Invalid SVG: Contains text elements
```
**Solution**: LLM generated text in image. System prompt should prevent this, but may need adjustment.

### Large SVG Size
```
Invalid SVG: SVG too long: 15000 characters (max 10000)
```
**Solution**: Adjust prompt to emphasize simplicity, or increase max_retries

## Extending the Code

### Add Custom Validation
```python
def _validate_svg(self, svg_code: str) -> Tuple[bool, str]:
    # Add your custom checks
    if 'forbidden_element' in svg_code:
        return False, "Contains forbidden element"
    return True, ""
```

### Modify Scoring
```python
# Use different metric weights
score = metric.harmonic_mean(vqa_score, aesthetic_score, beta=0.7)  # More weight on aesthetic
```

### Batch Processing
```python
# Process in smaller batches
batch_size = 10
for i in range(0, len(train_df), batch_size):
    batch = train_df.iloc[i:i+batch_size]
    batch['svg'] = batch.description.apply(model.predict)
    # Evaluate and save intermediate results
```

## License

This code extends the original competition solution and follows the same licensing terms.

