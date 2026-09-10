# Quick Start Guide - GPT-5 SVG Generation

Get started with GPT-5 SVG generation in 3 simple steps!

## Prerequisites

1. **Python 3.8+** installed
2. **API Key** for GPT-5 or other LLM
3. **Data files** in parent directory:
   - `train.csv`
   - `questions.parquet`
   - `metric.py`

## Step 1: Configure API Key

Edit `llm_tool.py` line 10:

```python
"gpt-5": {
    "api_key": "sk-your-actual-api-key-here",  # ← Replace this
    "base_url": "https://api.openai.com/v1",
    "model_name": "gpt-5-2025-08-07",
    "provider": "openai"
},
```

## Step 2: Install Dependencies

```bash
pip install openai pandas numpy matplotlib jupyter tqdm
```

## Step 3: Run!

### Option A: Quick Test (Single Image)

```bash
cd 00_LLM_refinement
python example_usage.py --mode single
```

**Output:**
```
Generating SVG for: 'a purple forest at dusk'
✓ Generated valid SVG (3847 chars)

Results:
   Competition Score: 0.7234
   VQA Score:         0.9123
   OCR Score:         1.0000
   Aesthetic Score:   0.6345
```

### Option B: Batch Processing (Multiple Images)

```bash
python example_usage.py --mode batch
```

### Option C: Full Evaluation (All Training Data)

```bash
python gpt5_svg_generator.py
```

This will:
1. Load all training data
2. Generate SVGs for each description
3. Evaluate using competition metric
4. Save results to `gpt5_results.csv`

### Option D: Interactive Notebook

```bash
jupyter notebook gpt5_svg_evaluation.ipynb
```

## What You Get

### Generated Files

- **`gpt5_results.csv`** - Full evaluation results with scores
- **`gpt5_submission.csv`** - Competition submission format
- **`example_output.svg`** - Sample generated SVG

### Results Format

```csv
id,description,svg,competition_score,vqa_score,ocr_score,aesthetic_score
02d892,"a purple forest at dusk","<svg...",0.7234,0.9123,1.0,0.6345
```

## Understanding the Scores

### Competition Score (0.0 - 1.0)
Final metric = `harmonic_mean(VQA, Aesthetic, β=0.5) × OCR`
- **0.8+** = Excellent
- **0.6-0.8** = Good
- **0.4-0.6** = Fair
- **<0.4** = Poor

### VQA Score (0.0 - 1.0)
How well the image matches the description
- Based on answering questions about the image
- Higher = better alignment with description

### Aesthetic Score (0.0 - 1.0)
Visual quality and appeal
- Uses CLIP embeddings + trained predictor
- Higher = more aesthetically pleasing

### OCR Score (0.0 - 1.0)
Text penalty (1.0 = no text detected)
- Penalizes any letters/text in image
- `exp(-num_chars + 4)`

## Workflow Comparison

### Original Diffusion Approach
```
Description → Generate 9 bitmaps → Score each → 
Pick best → Convert to SVG → Final score
Time: ~2-3 minutes
```

### GPT-5 LLM Approach
```
Description → Generate SVG directly → Score → Done
Time: ~2-5 seconds
```

## Troubleshooting

### Problem: API Key Error
```
Error: Invalid API key
```
**Solution**: Add your real API key to `llm_tool.py`

### Problem: Module Not Found
```
ModuleNotFoundError: No module named 'metric'
```
**Solution**: 
```bash
# Make sure metric.py exists in parent directory
ls ../metric.py
```

### Problem: SVG Too Large
```
Invalid SVG: SVG too long: 12000 characters (max 10000)
```
**Solution**: Model will auto-retry with simpler prompt. If persists, increase `max_retries`

### Problem: Low Scores
**Possible causes:**
- LLM not following SVG format correctly
- Colors/shapes don't match description
- Unexpected text in image

**Solutions:**
- Adjust system prompt for clarity
- Try different LLM model
- Increase max_retries

## Using Different Models

### Try Claude Opus:
```python
model = GPT5SVGModel(model_name="claude-opus", max_retries=3)
```

### Try Gemini Pro:
```python
model = GPT5SVGModel(model_name="gemini-pro", max_retries=3)
```

### Compare Models:
```bash
python example_usage.py --mode compare
```

## Next Steps

### 1. Tune the Prompts
Edit `gpt5_svg_generator.py` to modify:
- `system_prompt` - Overall instructions
- `user_prompt_template` - Per-image prompt

### 2. Experiment with Models
Test different LLMs to find best quality/cost balance

### 3. Optimize for Speed
- Use faster models (gemini-flash)
- Reduce max_retries
- Batch API requests

### 4. Improve Quality
- Refine prompts
- Add validation rules
- Use multi-step generation

## Example Code Snippets

### Generate Single SVG
```python
from gpt5_svg_generator import GPT5SVGModel

model = GPT5SVGModel(model_name="gpt-5")
svg = model.predict("a red circle on blue background")
print(svg)
```

### Evaluate SVG
```python
import metric

qa = {
    'question': ['Is there a red circle?'],
    'choices': [['no', 'yes']],
    'answer': ['yes']
}

score = metric.score_instance(qa, svg, random_seed=42)
print(f"Score: {score['competition_score']:.4f}")
```

### Process Batch
```python
import pandas as pd

df = pd.read_csv('descriptions.csv')
df['svg'] = df.description.apply(model.predict)
df.to_csv('results.csv', index=False)
```

## Support

For issues or questions:
1. Check `README.md` for detailed documentation
2. Review `example_usage.py` for code examples
3. Inspect `gpt5_svg_generator.py` for implementation details

## Performance Tips

### Speed Optimization
- Use `gemini-flash` for faster generation
- Set `max_retries=1` if quality is acceptable
- Process in parallel (if API allows)

### Quality Optimization
- Use `gpt-5` or `claude-opus` for best quality
- Increase `max_retries=5` for better validation
- Refine prompts based on failure patterns

### Cost Optimization
- Use open-source models (qwen, deepseek)
- Cache common generations
- Batch requests when possible

---

**Ready to generate SVGs? Start with:**

```bash
python example_usage.py --mode single
```

Good luck! 🎨

