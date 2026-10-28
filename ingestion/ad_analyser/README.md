# Ad Analyser Library

The `ad_analyser` package is a reusable processor designed to audit ad creatives by utilizing Google's Gemini 2.5 models (Vertex AI). It allows you to process pandas DataFrames or BigQuery SQL streams in a parallelized manner to extract structured visual intelligence (like brand, product details, categories, and languages).

## Architecture

Following clean code guidelines, the monolithic configuration has been decoupled into multiple specialized modules:
- `analyser.py`: Main orchestrator class (`AdAnalyser`).
- `config.py`: Declarative setup (`AdAnalyserConfig`).
- `gemini.py`: Thread-safe, retry-enabled wrapper for Vertex AI interactions.
- `storage.py`: Manage I/O across GCS, BigQuery, and Local scopes.
- `asset_staging.py`: Pure networking logic for fetching and streaming multimedia.
- `prompts.py`: Logic managing the construction of system instructions.
- `models.py`: Pydantic definitions for standardizing the results.
- `exceptions.py`: Custom, domain-specific exception classes.

## Getting Started

### 1. Initialization via Config

The massive multi-argument setup has been cleanly replaced with a dedicated `AdAnalyserConfig` dataclass.

```python
import pandas as pd
from ad_analyser import AdAnalyser, AdAnalyserConfig

# 1. Define configuration settings
config = AdAnalyserConfig(
    df=pd.read_csv("ads.csv"), # Or use input_query="SELECT * FROM dataset.table"
    gcp_project_id="your-gcp-project",
    gcp_region="us-central1",
    gcs_bucket="your-bucket-name",
    gcs_prefix="ad_audits",
    dest_table="your_project.dataset.enriched_ads",
    include_logs=True, # Will append 'status', 'error_message' to output
    chunk_size=500,
    max_workers=20,
    params={
        "master_brand_list": ["Maxis", "CelcomDigi", "U Mobile", "Unifi"],
        "category_l1": ["5G Postpaid", "Home Fibre", "Prepaid", "Device Bundles"],
    },
    labels={
        "app": "creative-theme-clustering",
        "domain": "malaysia-telco",
        "env": "prod"
    }
)

# 2. Instantiate Orchestrator
analyser = AdAnalyser(config)
```

### 2. Project-Specific Configurations (JSON)

To support multiple environments or projects, you can define configurations in a standalone JSON file and initialize the service dynamically. This saves you from hardcoding settings in your script.

**`project_a_config.json`**
```json
{
    "gcp_project_id": "your-gcp-project",
    "gcp_region": "us-central1",
    "gcs_bucket": "your-bucket-name",
    "gcs_prefix": "ad_audits",
    "dest_table": "your_project.dataset.enriched_ads",
    "chunk_size": 500,
    "max_workers": 20,
    "include_logs": true,
    "params": {
        "master_brand_list": ["Nike", "Adidas", "Puma"],
        "category_l1": ["Footwear", "Apparel", "Accessories"]
    }
}
```

**`main.py`**
```python
import pandas as pd
from ad_analyser import AdAnalyser, AdAnalyserConfig

# Load from JSON and apply runtime overrides (like passing the DataFrame)
config = AdAnalyserConfig.from_json(
    "project_a_config.json", 
    df=pd.read_csv("ads.csv") 
)

analyser = AdAnalyser(config)
analyser.run_job()
```

### 3. Project-Specific Configurations (Python Class)

If you prefer to keep logic in Python (e.g., to construct dynamic queries based on attributes), you can load settings directly from any Python class or object using `.from_object()`.

**`proof_of_concept.py`**
```python
class ProofOfConceptConfigs:
    gcp_project_id = "your-gcp-project"
    gcs_bucket = "your-bucket-name"
    chunk_size = 500
    
    @classmethod
    def get_query(cls):
        return f"SELECT * FROM `{cls.gcp_project_id}.dataset.table`"

    # This will be picked up properly
    input_query = get_query.__func__(None)
```

**`main.py`**
```python
from ad_analyser import AdAnalyser, AdAnalyserConfig
from proof_of_concept import ProofOfConceptConfigs

# Automatically extract matching properties from the class
config = AdAnalyserConfig.from_object(
    ProofOfConceptConfigs, 
    include_logs=True # Override fields via kwargs
)

analyser = AdAnalyser(config)
analyser.run_job()
```

### 4. Running a Job

Once initialized, trigger the main processing loop. The orchestrator automatically handles:
- **Deduplication:** Dropping duplicate creatives to save API costs.
- **Parallel Processing:** Utilizing `ThreadPoolExecutor` and `GeminiAIManager` to asynchronously analyze files.
- **Auto-Reconciliation:** Merging enriched AI metadata back to your initial DataFrame cleanly after duplicate compression.
- **Persistence:** Transparently routing outputs to your Google Cloud Storage or BigQuery instance utilizing `StorageManager`.

```python
# 3. Fire the execution
# Optional: provide a limit (e.g., limit=100) to test a subset of records.
result_df = analyser.run_job(limit=None)

# If no 'output_path' or 'dest_table' is specified, the final 
# enriched DataFrame is returned synchronously.
if result_df is not None:
    print(f"Processed DataFrame Shape: {result_df.shape}")
    print(result_df.head())
```

## Supported Features

- **Automatic Content Fallback**: If an image URL download fails or Vertex AI rejects a direct URI (like `url_rejected` or `robots.txt` failures), `http_utils.py` automatically stages it tightly to GCS and tries again using a native blob transfer line.
- **WebM Conversion**: Automatic detection of `.webm` `403 Forbidden` issues routing back directly to `.mp4` fallbacks.
- **Structured JSON Outpaces**: Extensively enforces single-format outputs mapping explicitly over `response_schema` attributes via Pydantic model (`AdAnalysisResults`).
- **Granular Retries:** Implements an internal back-off schema resolving network timeouts autonomously and mitigating noisy transient errors.

## Error Handling
You can explicitly import exceptions to handle logic downstream robustly:
```python
from ad_analyser import DataValidationError, ContentDownloadError, IntegrationError

try:
    analyser.run_job()
except DataValidationError as e:
    print(f"Schema problem with DataFrame input: {e}")
```
