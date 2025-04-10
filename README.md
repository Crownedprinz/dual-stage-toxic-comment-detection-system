# dual-stage-toxic-comment-detection-system
The growing prevalence of toxic online comments negatively impacts user experiences and community engagement across social platforms.

## Dataset Preparation

### Running the Dataset Combiner

The [notebooks/jigsaw_datasets_combiner.ipynb](notebooks/jigsaw_datasets_combiner.ipynb) notebook handles downloading and combining multiple toxic comment datasets. Here's what it does:

1. Downloads required datasets from Google Drive:
   - Main training data (`train.csv`)
   - Test datasets:
     - Private test set (`test_private_expanded.csv`) 
     - Public test set (`test_public_expanded.csv`)
     - Standard test set (`test.csv`)

2. Processes the raw data:
   - Cleans text by lowercasing and removing punctuation
   - Tokenizes comments using spaCy for improved performance
   - Handles missing values and removes duplicates

3. Combines datasets:
   - Merges original training data (159,571 samples) with cleaned data (1,804,874 samples)
   - Standardizes column names across sources
   - Adds source tracking
   - Results in combined dataset with 1,964,445 samples

### Requirements

```python
pip install pandas gdown nltk spacy tqdm
python -m spacy download en_core_web_sm
