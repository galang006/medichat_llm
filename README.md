# MediChat LLM: Fine-tuned Medical Language Model

This project aims to develop a specialized Large Language Model (LLM) for medical question-answering. It involves a complete pipeline from preparing diverse medical conversational datasets to fine-tuning a base LLM, specifically Google's Gemma-3-1b-it, to enhance its ability to provide clear and concise answers to medical queries.

## Features
-   **Automated Data Preprocessing:** Efficiently processes and transforms raw medical Q&A datasets (HealthCareMagic-100k-en and MedQuAD) into a unified instruction-response format suitable for LLM fine-tuning.
-   **Dataset Combination and Shuffling:** Combines multiple preprocessed datasets and shuffles them to create a comprehensive and randomized training corpus.
-   **Parameter-Efficient Fine-tuning (PEFT):** Utilizes Low-Rank Adaptation (LoRA) alongside 4-bit quantization (BitsAndBytes) to fine-tune the Gemma model with reduced computational resources and memory footprint.
-   **Supervised Fine-tuning (SFT):** Employs the `trl.SFTTrainer` from the TRL library for a streamlined fine-tuning process.
-   **Hugging Face Hub Integration:** Seamlessly saves and pushes the fine-tuned model and its tokenizer to the Hugging Face Hub, enabling easy sharing and deployment.
-   **Google Colab Optimized:** The fine-tuning notebook is designed to run effectively within Google Colab environments, leveraging GPU acceleration.

## Installation
To set up the project and run the notebooks, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/galang006/medichat_llm.git
    cd medichat_llm
    ```

2.  **Install required Python libraries:**
    The project relies on several libraries for data manipulation, LLM operations, and fine-tuning. These can be installed using pip:
    ```bash
    pip install torch transformers accelerate peft trl datasets bitsandbytes pandas
    ```

3.  **Prepare your environment for Hugging Face (optional but recommended for fine-tuning):**
    If you plan to fine-tune the model and push it to the Hugging Face Hub, you will need to log in:
    ```python
    from huggingface_hub import login
    login() # Follow the prompts to enter your Hugging Face token
    ```
    Alternatively, you can set your token as an environment variable: `HF_TOKEN=hf_YOUR_TOKEN_HERE`.

4.  **Google Drive Setup (for Colab users):**
    The `fine_tune_gemma.ipynb` notebook expects the preprocessed dataset to be available on Google Drive. If running in Google Colab, mount your Google Drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    Ensure your dataset path in the notebook (`dataset_path = "/content/drive/MyDrive/Dataset/merged_medical_dataset_clean.jsonl"`) matches the location of your processed data on Google Drive.

## Usage

### 1. Data Preprocessing
The `data_preprocessing.ipynb` notebook is responsible for preparing the raw medical datasets.

**Input Data:**
Place the following raw datasets in a `dataset/` directory within your project root:
-   `HealthCareMagic-100k-en.jsonl` (JSONL format)
-   `medquad.csv` (CSV format)

**Steps:**
1.  Open and run the `data_preprocessing.ipynb` notebook.
2.  The notebook will:
    *   Load `HealthCareMagic-100k-en.jsonl` and convert it into an instruction-input-output format.
    *   Load `medquad.csv` and convert it into the same instruction-input-output format.
    *   Save these individual formatted datasets as `hc_instruction_format.jsonl` and `mq_instruction_format.jsonl` respectively in the `dataset/` directory.
    *   Combine and shuffle both formatted datasets.
    *   Save the final combined and shuffled dataset as `medical_qa_combined_shuffled.jsonl` in the `dataset/` directory. This file will be used for fine-tuning.

### 2. Fine-tuning the Gemma Model
The `fine_tune_gemma.ipynb` notebook handles the fine-tuning of the Gemma-3-1b-it model.

**Steps:**
1.  Ensure you have run the data preprocessing step and the `medical_qa_combined_shuffled.jsonl` file is available (preferably on Google Drive if using Colab, or locally).
2.  Open and run the `fine_tune_gemma.ipynb` notebook.
3.  The notebook will perform the following actions:
    *   Install necessary libraries (if not already installed).
    *   Log in to Hugging Face (if required).
    *   Mount Google Drive (if in Colab).
    *   Load the `medical_qa_combined_shuffled.jsonl` dataset. Note that the notebook currently takes only the first 1000 examples for demonstration purposes. You might want to adjust `df = df.head(1000)` to use the full dataset.
    *   Initialize the `gemma-3-1b-it` tokenizer and model, applying 4-bit quantization.
    *   Configure and apply LoRA adapters to the model.
    *   Set up the `SFTTrainer` with specified training arguments (e.g., 1 epoch, batch size 1, gradient accumulation 4).
    *   Initiate the fine-tuning process.
    *   Save the fine-tuned LoRA adapter weights.
    *   Merge the LoRA adapters with the base model to create a fully fine-tuned model.
    *   Push the merged model and its tokenizer to your specified Hugging Face repository (e.g., `galang006/gemma-medical-sft`).

### 3. Inference (Example)
After fine-tuning, you can load your merged model from Hugging Face Hub (or locally) and use it for medical question-answering.

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Replace with your Hugging Face model ID or local path
model_id = "galang006/gemma-medical-sft" # Or "./merged_gemma_medical_model"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

def generate_medical_response(instruction, user_input):
    prompt = f"{instruction}\n{user_input}"
    input_ids = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Post-process to extract only the bot's answer if the prompt includes the instruction and input
    # This assumes the model generates the full conversation including the prompt.
    # You might need to adjust this based on your model's output format.
    if prompt in response:
        return response.split(prompt)[-1].strip()
    return response.strip()

instruction = "Answer the following medical question in a clear and concise way."
user_question = "What are the symptoms of a common cold?"
response = generate_medical_response(instruction, user_question)
print(f"Bot: {response}")

user_question = "How to treat a minor burn?"
response = generate_medical_response(instruction, user_question)
print(f"Bot: {response}")
```

## Code Structure

The project is organized as follows:

```
medichat_llm/
├── .gitignore               # Specifies intentionally untracked files to ignore
├── data_preprocessing.ipynb # Jupyter notebook for preparing and combining datasets
├── fine_tune_gemma.ipynb    # Jupyter notebook for fine-tuning the Gemma model
└── dataset/                 # Directory to store raw and processed datasets (created during preprocessing)
    ├── HealthCareMagic-100k-en.jsonl # Raw medical Q&A dataset 1
    ├── medquad.csv          # Raw medical Q&A dataset 2
    ├── hc_instruction_format.jsonl   # Preprocessed HealthCareMagic dataset
    ├── mq_instruction_format.jsonl   # Preprocessed MedQuAD dataset
    └── medical_qa_combined_shuffled.jsonl # Combined and shuffled dataset for fine-tuning
```

-   **`.gitignore`**: Standard file to exclude generated files (like `/dataset` content) and other temporary files from version control.
-   **`data_preprocessing.ipynb`**: This notebook contains all the logic for loading the raw datasets, parsing their specific formats, converting them into a consistent instruction-input-output JSONL format, combining them, and saving the final dataset ready for model training.
-   **`fine_tune_gemma.ipynb`**: This notebook orchestrates the LLM fine-tuning process. It includes steps for installing dependencies, authenticating with Hugging Face, loading the preprocessed data, initializing the base Gemma model with quantization, applying LoRA, training the model, and finally saving and pushing the merged, fine-tuned model to the Hugging Face Hub.
-   **`dataset/`**: This directory serves as the central location for both the initial raw datasets and the intermediate/final processed datasets.