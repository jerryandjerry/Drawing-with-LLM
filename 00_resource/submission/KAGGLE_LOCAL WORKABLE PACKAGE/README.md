# Drawing with LLMs - Text-to-SVG Generation System

This project is a Kaggle competition solution that converts text descriptions into high-quality SVG images. The system generates multiple candidate images, evaluates them against visual question-answering criteria, and converts the best result to SVG format.

## Table of Contents
- [Overview](#overview)
- [Generation Workflow](#generation-workflow)
- [Evaluation Process](#evaluation-process)
- [Models Used](#models-used)
- [Requirements](#requirements)
- [Usage](#usage)

---

## Overview

The system takes a text description (e.g., "a purple forest at dusk") and produces an SVG image that accurately represents the description while maintaining aesthetic quality. The process involves:

1. **Multi-candidate generation**: Generate 9 bitmap images using Stable Diffusion
2. **Quality scoring**: Evaluate each candidate using VQA, aesthetic, and OCR metrics
3. **Bitmap-to-SVG conversion**: Convert the best bitmap to optimized SVG format
4. **Size optimization**: Ensure SVG stays under 10KB limit

---

## Generation Workflow

### Architecture Overview

The system uses a **multi-GPU, multi-process architecture**:

- **GPU 0 (Scorer Process)**: Runs PaliGemma-2 VQA model and CLIP aesthetic evaluator
- **GPU 1 (Generator Process)**: Runs SDXL-Flash diffusion model
- **Queue-based communication**: Processes communicate via shared queues

### Step-by-Step Generation Flow

```
Text Description
      ↓
1. Generate Temporary Q&A (spaCy NLP)
      ↓
2. Create Enhanced Prompt
      ↓
3. Generate 9 Candidate Images (SDXL-Flash + LoRA)
      ↓
4. Score Each Candidate (VQA + Aesthetic + OCR)
      ↓
5. Select Best Image
      ↓
6. Analyze Color Richness (determines SVG resolution)
      ↓
7. Convert to SVG (vtracer with binary search optimization)
      ↓
8. Inject Anti-OCR markers
      ↓
Final SVG Output
```

### Detailed Steps

#### 1. Temporary Q&A Generation (Internal Use Only)

For candidate selection, the system generates questions from the prompt using spaCy NLP:

```python
def generate_qa_spacy(text):
    # Extract noun phrases
    doc = nlp(text)
    descriptive_elements = set(chunk.text for chunk in doc.noun_chunks)
    
    # Generate yes/no questions
    questions = []
    for element in descriptive_elements:
        question = f'Are there {element} in the image? Answer yes or no.'
        questions.append(question)
    
    # Add final comprehensive question
    questions.append(f"Are there {text}? Answer yes or no.")
    
    return {'question': questions, 'choices': [["no", "yes"]] * len(questions), 'answer': ["yes"] * len(questions)}
```

**Note**: This temporary Q&A is only used for **selecting the best of 9 candidates**. The actual competition scoring uses pre-defined questions from Kaggle.

#### 2. Prompt Enhancement

```python
prompt = f"{prompt_prefix} {description}{prompt_suffix}"
# Example: "a stylized digital painting presenting a purple forest at dusk. 
#           The painting promote vector-art aesthetic, in watercolor art style..."
```

#### 3. Image Generation (GPU 1)

- **Model**: SDXL-Flash with Vector Illustration LoRA
- **Parameters**: 
  - Resolution: 768×768
  - Steps: 7
  - CFG Scale: 2.5
  - Batch: 9 images (sequentially)

#### 4. Scoring (GPU 0)

Each candidate is evaluated on three metrics:

**a) VQA Score** (Visual Question Answering)
- Model: PaliGemma-2-10B (4-bit quantized)
- Answers questions about image content
- Returns probability of correct answers

**b) Aesthetic Score**
- Model: CLIP ViT-L/14 + trained aesthetic predictor
- Measures visual quality and appeal
- Scale: 0.0 to 1.0

**c) OCR Score**
- Detects and penalizes text/lettering
- Formula: `min(1.0, exp(-num_chars + 4))`
- Allows up to 4 "free" characters

**Final Candidate Score**: `harmonic_mean(VQA, Aesthetic, β=0.5) × OCR`

Where β=0.5 weights VQA slightly higher than aesthetic score.

#### 5. Color Richness Analysis

```python
def compute_color_richness_entropy(image):
    # Convert to LAB color space
    # Compute entropy across L, A, B channels
    # Normalize to [0, 1]
    return normalized_entropy

def map_score_to_range(normalized_score):
    if normalized_score <= 0.5:
        return 384  # Simple images → high resolution
    elif normalized_score >= 0.7:
        return 128  # Complex images → low resolution (more colors)
    else:
        # Linear interpolation
        return int(384 - alpha * 256)
```

This adaptively adjusts SVG complexity based on image characteristics.

#### 6. Bitmap-to-SVG Conversion

Uses **vtracer** with binary search optimization:

```python
def bitmap_to_svg_layered(img, input_path, output_path, resolution):
    max_svg_length = 9996  # 10KB limit minus injection
    
    # Binary search for best layer_difference parameter
    low, high = 1, 200
    while low <= high:
        layer_difference = (low + high) // 2
        
        # Convert with current parameters
        vtracer.convert_image_to_svg_py(
            input_path, output_path,
            colormode='color',
            hierarchical='stacked',
            mode='polygon',
            layer_difference=layer_difference,
            # ... other parameters
        )
        
        # Check if within size limit
        if len(svg_code) + injection_length <= max_svg_length:
            best_svg_code = svg_code
            high = layer_difference - 1  # Try for more detail
        else:
            low = layer_difference + 1   # Simplify more
    
    return best_svg_code
```

**Key optimizations**:
- Converts paths to polygons (more compact)
- Removes XML declarations and namespaces
- Binary search finds maximum detail within size limit
- Falls back to higher layer_difference values if needed

#### 7. Anti-OCR Injection

To prevent hallucinated text detection:

```python
injection_a1 = '<path d="M20 364 L24 356 L28 364 M22 360 L26 360" stroke="#CCCCCC"/>'  # Bottom-left A
injection_a2 = '<path d="M364 28 L360 20 L356 28 M362 24 L358 24" stroke="#888888"/>'  # Top-right A
svg_code = svg_code.replace("</svg>", injection_a1 + injection_a2 + "</svg>")
```

These subtle letter shapes in corners help satisfy OCR models without being visually intrusive.

---

## Evaluation Process

### Local Evaluation vs Competition Evaluation

Both use the **same scoring function**, but different contexts:

| Aspect | Local Evaluation | Competition Evaluation |
|--------|------------------|----------------------|
| **Purpose** | Development/testing | Final submission scoring |
| **Q&A Source** | `questions.parquet` (local) | Kaggle's pre-defined Q&A |
| **Inputs** | `multiple_choice_qa` + `svg` | `multiple_choice_qa` + `svg` |
| **Prompt Used?** | ❌ No (only for generation) | ❌ No (only for generation) |

### Evaluation Inputs

```python
# Function signature
metric.score_instance(multiple_choice_qa, svg, random_seed=42)

# Where multiple_choice_qa is:
{
    'question': ['Is there a purple forest in the image?', 'Is there dusk in the image?', ...],
    'choices': [['no', 'yes'], ['no', 'yes'], ...],
    'answer': ['yes', 'yes', ...]
}
```

**Important**: The text description/prompt is **NOT** an input to evaluation. It's only used during generation.

### Scoring Components

#### 1. Visual Question Answering (VQA) Score

```python
def score(questions, choices, answers, image, n=1):
    scores = []
    for question, choice_list, answer in zip(questions, choices, answers):
        # Get probability distribution over choices
        choice_probabilities = model.get_choice_probability(image, question, choice_list)
        
        # Extract probability of correct answer
        answer_probability = choice_probabilities[answer]
        scores.append(answer_probability)
    
    return statistics.mean(scores)
```

- **Model**: PaliGemma-2-10B
- **Method**: Multiple-choice format with probability masking
- **Output**: Mean probability of correct answers (0.0 to 1.0)

#### 2. Aesthetic Score

```python
def score(image):
    # Extract CLIP features
    image_features = clip_model.encode_image(image)
    image_features = normalize(image_features)
    
    # Predict aesthetic score
    score = aesthetic_predictor(image_features)
    
    return score / 10.0  # Scale to [0, 1]
```

- **Model**: CLIP ViT-L/14 + trained linear predictor
- **Training**: Trained on SAC+Logos+AVA1 aesthetic dataset
- **Output**: Aesthetic quality score (0.0 to 1.0)

#### 3. OCR Score (Text Penalty)

```python
def ocr(image, free_chars=4):
    # Run OCR on image
    decoded = model.generate(image, prompt="<image>ocr\n")
    num_char = len(decoded)
    
    # Exponential decay penalty
    return min(1.0, math.exp(-num_char + free_chars))
```

- **Model**: PaliGemma-2 (OCR mode)
- **Penalty**: Exponential decay for detected characters
- **Free chars**: First 4 characters don't penalize (allows small artifacts)

#### 4. Image Preprocessing (Robustness Testing)

Before scoring, images undergo defensive transformations:

```python
def apply(self):
    return (
        self.apply_random_crop_resize(crop_percent=0.03)
        .apply_jpeg_compression(quality=95)
        .apply_median_filter(size=9)
        .apply_fft_low_pass(cutoff_frequency=0.5)
        .apply_bilateral_filter(d=5, sigma_color=75, sigma_space=75)
        .apply_jpeg_compression(quality=92)
    )
```

This tests robustness against:
- Cropping and resizing
- JPEG compression artifacts
- Noise filtering
- Frequency domain filtering

### Final Score Calculation

```python
def score_instance(multiple_choice_qa, svg, random_seed=42):
    # Convert SVG to PNG
    image = svg_to_png(svg)
    
    # Apply defensive transformations
    image_processor = ImageProcessor(image=image, seed=seed).apply()
    image = image_processor.image.copy()
    
    # Score components
    aesthetic_score = aesthetic_evaluator.score(image)
    vqa_score = vqa_evaluator.score(
        questions=multiple_choice_qa['question'],
        choices=multiple_choice_qa['choices'],
        answers=multiple_choice_qa['answer'],
        image=image
    )
    
    # Apply more transforms for OCR test
    image_processor.reset().apply_random_crop_resize().apply_jpeg_compression(quality=90)
    ocr_score = vqa_evaluator.ocr(image_processor.image)
    
    # Combine scores
    instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=0.5) * ocr_score
    
    return instance_score
```

**Score Formula**:

```
Final Score = harmonic_mean(VQA, Aesthetic, β=0.5) × OCR

where harmonic_mean(a, b, β) = (1 + β²) × (a × b) / (β² × a + b)
```

With β=0.5:
- VQA gets slightly higher weight (~55-60%)
- Aesthetic gets slightly lower weight (~40-45%)
- OCR acts as a multiplier (penalizes text heavily)

### Local Evaluation Workflow

```python
# 1. Load data with pre-defined Q&A
train_df = pd.read_csv('train.csv')
train_question_df = pd.read_parquet('questions.parquet')
train_df['multiple_choice_qa'] = create_qa_dict(train_df)

# 2. Generate SVGs from descriptions
train_df['raw_res'] = train_df.description.progress_apply(model.predict_impl)
train_df['svg'] = train_df.raw_res.apply(lambda x: x[0])
train_df['bitmap'] = train_df.raw_res.apply(lambda x: x[1])

# 3. Score both bitmap and SVG
train_df['bitmap_score'] = train_df.progress_apply(
    lambda r: bitmap_score_instance(r.multiple_choice_qa, r.bitmap),
    axis=1
)
train_df['svg_score'] = train_df.progress_apply(
    lambda r: metric.score_instance(r.multiple_choice_qa, r.svg),
    axis=1
)

# 4. Compare results
mean_bitmap_score = train_df['bitmap_score'].mean()
mean_svg_score = train_df['svg_score'].mean()
```

This allows comparing:
- **Bitmap score**: Quality of direct image generation
- **SVG score**: Quality after bitmap→SVG conversion (tests vtracer fidelity)

---

## Models Used

### Image Generation
- **SDXL-Flash**: Fast Stable Diffusion variant (7 steps)
- **Vector Illustration LoRA**: Fine-tuned weights for vector art style
- **DPMSolver**: Single-step scheduler for speed

### Evaluation Models
- **PaliGemma-2-10B** (4-bit NF4): Vision-language model for VQA and OCR
  - Path: `paligemma2-10b-mix-448/`
- **CLIP ViT-L/14**: Visual feature extraction
  - Path: `clip-vit-large-patch14/ViT-L-14.pt`
- **Aesthetic Predictor**: Linear model trained on aesthetic datasets
  - Path: `sac-logos-ava1-l14-linearmse/sac+logos+ava1-l14-linearMSE.pth`

### NLP & Conversion
- **spaCy en_core_web_lg**: Question generation from prompts
- **vtracer**: Bitmap-to-SVG conversion library

---

## Requirements

```txt
svgwrite
cairosvg
openai-clip
opencv-python 
scikit-image 
pillow
bitsandbytes
peft
gguf
pyarrow
pandas
transformers==4.49.0
diffusers==0.32.2
vtracer
spacy
more_itertools
```

### Model Files Required

```
├── paligemma2-10b-mix-448/          # PaliGemma-2 VQA model
├── clip-vit-large-patch14/          # CLIP model
├── sac-logos-ava1-l14-linearmse/    # Aesthetic predictor
├── sdxl-flash/                      # SDXL-Flash diffusion model
├── lora/                            # Vector illustration LoRA weights
├── en_core_web_lg/                  # spaCy English model
└── kaggle_evaluation/               # Competition evaluation package
```

---

## Usage

### Basic Generation

```python
from your_module import Model

# Initialize model (loads all components)
model = Model()

# Generate SVG from text description
description = "a purple forest at dusk"
svg_string = model.predict(description)

# Save SVG
with open('output.svg', 'w') as f:
    f.write(svg_string)
```

### Advanced Generation with Scoring

```python
# Get both SVG and best bitmap
svg_string, best_bitmap = model.predict_impl(description)

# Display results
from IPython.display import display, SVG
display(best_bitmap)
display(SVG(svg_string))
```

### Local Evaluation

```python
import pandas as pd
import metric

# Load data
df = pd.read_csv('test.csv')
df_questions = pd.read_parquet('questions.parquet')

# Merge to get Q&A
df = df.merge(df_questions, on='id')
df['multiple_choice_qa'] = create_qa_dict(df)

# Generate and score
df['svg'] = df.description.apply(model.predict)
df['score'] = df.apply(
    lambda r: metric.score_instance(r.multiple_choice_qa, r.svg),
    axis=1
)

# Get mean score
mean_score = df['score'].mean()
print(f"Average score: {mean_score:.4f}")
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Main Process                        │
│  - Manages worker processes                                 │
│  - Handles queue communication                              │
│  - Collects and compares results                            │
└────────────┬────────────────────────────┬───────────────────┘
             │                            │
             ▼                            ▼
┌──────────────────────────┐  ┌──────────────────────────┐
│   Generator (GPU 1)      │  │   Scorer (GPU 0)         │
│  - SDXL-Flash            │  │  - PaliGemma-2 VQA       │
│  - Vector LoRA           │  │  - CLIP Aesthetic        │
│  - Generates 9 images    │  │  - Scores candidates     │
│  - Compiled with Dynamo  │  │  - Bitmap→SVG conversion │
└────────────┬─────────────┘  └─────────────┬────────────┘
             │                               │
             │    prompt_q (text + qa)       │
             │──────────────────────────────►│
             │                               │
             │◄──────────────────────────────│
             │    image_q (bitmap + qa)      │
             │                               │
             │                               │
             │◄──────────────────────────────│
             │    result_q (scores + svg)    │
             ▼                               ▼
```

### Queue Flow

1. **prompt_q**: Main → Generator
   - Contains: `(tag, prompt, qa, num_attempts)`
   
2. **image_q**: Generator → Scorer
   - Contains: `(tag, png_bytes, qa)`
   
3. **result_q**: Scorer → Main
   - Contains: `(tag, instance_score, aesthetic_score, ocr_score, vqa_score, svg, png_bytes)`

---

## Performance Characteristics

### Timing (per image set)
- **Image Generation**: ~3-4s per image × 9 = ~27-36s
- **Scoring**: ~10-15s per image × 9 = ~90-135s
- **SVG Conversion**: ~5-10s
- **Total**: ~2-3 minutes per description

### Memory Usage
- **GPU 0 (Scorer)**: ~12GB (PaliGemma-2 4-bit + CLIP)
- **GPU 1 (Generator)**: ~8GB (SDXL-Flash compiled)
- **System RAM**: ~16GB (data loading, queues)

### Quality Metrics (typical ranges)
- **VQA Score**: 0.85-0.99 (higher is better)
- **Aesthetic Score**: 0.50-0.65 (higher is better)
- **OCR Score**: 0.95-1.0 (1.0 = no text detected)
- **Final Score**: 0.65-0.85 (competition metric)

---

## Key Design Decisions

### Why Multi-Process Architecture?
- **GPU isolation**: Each GPU runs one model without conflicts
- **Parallel scoring**: Can score previous images while generating new ones
- **Memory management**: Separate processes prevent VRAM fragmentation

### Why 9 Candidates?
- Balance between quality and speed
- Empirically found to provide good diversity
- Allows recovery from occasional bad generations

### Why spaCy Q&A for Selection?
- Fast NLP processing
- Generates relevant questions automatically
- Better than random selection or using only aesthetics

### Why Binary Search for SVG Optimization?
- Deterministic way to maximize detail within size limit
- Faster than trial-and-error
- Ensures consistent output size

---

## Troubleshooting

### Out of Memory (OOM)
- Reduce `num_images_per_prompt` in generator
- Increase `layer_difference` start value for vtracer
- Use smaller batch size for VQA scoring

### Low Scores
- Check if OCR is penalizing (look for text in images)
- Verify prompt engineering (try adjusting prefix/suffix)
- Ensure models are loaded correctly

### SVG Too Large
- vtracer will automatically simplify until size limit met
- If still failing, check injection length
- May need to adjust `max_svg_length`

### Slow Performance
- Enable model compilation (already on for SDXL)
- Check GPU utilization
- Ensure processes aren't blocking on queues

---

## License

This project was developed for the Kaggle "Drawing with LLMs" competition. Model weights and datasets have their respective licenses:
- **SDXL**: CreativeML Open RAIL++-M License
- **PaliGemma**: Gemma Terms of Use
- **CLIP**: MIT License
- **Competition Data**: Kaggle Competition Rules

---

## Acknowledgments

- Kaggle competition organizers for the challenge
- Stability AI for SDXL models
- Google for PaliGemma and CLIP models
- vtracer contributors for bitmap-to-SVG conversion

