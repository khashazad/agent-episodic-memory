"""
Local HuggingFace model setup for the MineRL agent.
"""

import os
import sys

# Enable HuggingFace progress bars and ensure output is unbuffered
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"  # Faster downloads
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Avoid tokenizer warnings

try:
    from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoConfig, pipeline
    import torch
    HF_AVAILABLE = True
except ImportError as e:
    HF_AVAILABLE = False
    _import_error = e


def get_local_config():
    """Get local model configuration from environment variables."""
    return {
        "model_name": os.environ.get("LOCAL_MODEL_NAME", "microsoft/Phi-3-mini-4k-instruct"),
        "device": os.environ.get("LOCAL_MODEL_DEVICE", "auto"),  # 'auto', 'cuda', 'cpu', 'mps'
        "max_new_tokens": int(os.environ.get("LOCAL_MODEL_MAX_NEW_TOKENS", "256")),
        "temperature": float(os.environ.get("LOCAL_MODEL_TEMPERATURE", "0.2")),
        "dtype": os.environ.get("LOCAL_MODEL_DTYPE", "float16"),  # 'float16', 'float32', 'bfloat16'
    }


def setup_local_agent(config: dict | None = None):
    """
    Setup local Hugging Face model-based agent.

    Args:
        config: Optional configuration dictionary. If None, reads from environment variables.
               Expected keys: model_name, device, max_new_tokens, temperature, dtype

    Returns:
        ChatHuggingFace: The configured chat model wrapped in LangChain interface.

    Raises:
        ImportError: If required HuggingFace dependencies are not installed.
    """
    if not HF_AVAILABLE:
        raise ImportError(
            "This agent requires langchain-huggingface and transformers. "
            "Install with: pip install langchain-huggingface transformers torch"
        ) from _import_error

    # Use provided config or get from environment
    if config is None:
        config = get_local_config()

    model_name = config["model_name"]
    configured_device = config["device"]
    max_new_tokens = config["max_new_tokens"]
    temperature = config["temperature"]
    dtype_str = config["dtype"]

    print(f"\nLoading local model: {model_name}")
    print(f"Configured device: {configured_device}")
    sys.stdout.flush()

    # Determine the device
    if configured_device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = configured_device

    print(f"Using device: {device}")

    # Determine dtype
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype_str, torch.float16)
    if device == 'cpu':
        torch_dtype = torch.float32  # CPU typically needs float32

    print(f"Using dtype: {torch_dtype}")
    sys.stdout.flush()

    # Load tokenizer with progress indication
    print("\n[1/5] Loading tokenizer...")
    sys.stdout.flush()
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
        )
        print("✓ Tokenizer loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        sys.stdout.flush()
        raise

    # Set device_map for automatic device placement
    device_map = "auto" if device in ['cuda', 'mps'] else None

    # Check if this is a vision-language model by inspecting the config
    print("\n[2/5] Loading model configuration...")
    sys.stdout.flush()
    try:
        model_config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=True,
            resume_download=True,
        )
        config_class_name = model_config.__class__.__name__
        print(f"✓ Config loaded: {config_class_name}")
        sys.stdout.flush()

        # Vision-language models typically have "VL" in their config name
        # or are not supported by AutoModelForCausalLM
        is_vision_language = (
            'VL' in config_class_name or
            'Vision' in config_class_name or
            'Multimodal' in config_class_name
        )

        print("\n[3/5] Loading model weights (this may take several minutes)...")
        print("      If this is the first time, the model will be downloaded from HuggingFace.")
        print("      Large models can take 5-15 minutes to download depending on network speed.")
        print("      Progress bars should appear below if downloading...")
        sys.stdout.flush()

        if is_vision_language:
            print(f"      Detected vision-language model ({config_class_name}), using AutoModel")
            sys.stdout.flush()
            model = AutoModel.from_pretrained(
                model_name,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
        print("✓ Model weights loaded successfully")
        sys.stdout.flush()
    except Exception as e:
        # Fallback: try AutoModelForCausalLM first, then AutoModel if it fails
        print(f"✗ Config check failed ({e}), trying fallback loading methods...")
        print("\n[3/5] Attempting to load with AutoModelForCausalLM...")
        sys.stdout.flush()
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded with AutoModelForCausalLM")
            sys.stdout.flush()
        except ValueError as ve:
            print(f"✗ AutoModelForCausalLM failed: {ve}")
            print("   Trying AutoModel...")
            sys.stdout.flush()
            model = AutoModel.from_pretrained(
                model_name,
                dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True,
                resume_download=True,
                low_cpu_mem_usage=True,
            )
            print("✓ Model loaded with AutoModel")
            sys.stdout.flush()
        except Exception as e2:
            print(f"✗ Failed to load model: {e2}")
            sys.stdout.flush()
            raise

    # Move to device if not using device_map
    if device_map is None:
        print(f"\n[4/5] Moving model to {device}...")
        sys.stdout.flush()
        model = model.to(device)
        print(f"✓ Model moved to {device}")
        sys.stdout.flush()

    # Create the text generation pipeline
    print("\n[5/5] Creating text generation pipeline...")
    sys.stdout.flush()
    try:
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.9,
            return_full_text=False,
        )
        print("✓ Pipeline created successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to create pipeline: {e}")
        sys.stdout.flush()
        raise

    # Wrap in LangChain's HuggingFacePipeline
    print("\nWrapping in LangChain interface...")
    sys.stdout.flush()
    try:
        hf_llm = HuggingFacePipeline(pipeline=pipe)
        chat_model = ChatHuggingFace(llm=hf_llm)
        print("✓ LangChain wrapper created successfully")
        sys.stdout.flush()
    except Exception as e:
        print(f"✗ Failed to create LangChain wrapper: {e}")
        sys.stdout.flush()
        raise

    print(f"\n{'='*50}")
    print(f"Local model {model_name} loaded successfully!")
    print(f"{'='*50}\n")
    sys.stdout.flush()
    return chat_model
