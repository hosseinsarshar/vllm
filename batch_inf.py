import vllm
import time

# --- Configuration ---

# Model Identifier on Hugging Face Hub
# Use "meta-llama/Meta-Llama-3-8B-Instruct" for instruction-following/chat
# Use "meta-llama/Meta-Llama-3-8B" for the base model (less common for direct use)
MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Number of GPUs to use. vLLM automatically detects and uses available GPUs if set to 1.
# Increase this number if you have multiple GPUs and want tensor parallelism.
TENSOR_PARALLEL_SIZE = 1

print(f"Initializing vLLM engine for model: {MODEL_ID}")
print(f"Using {TENSOR_PARALLEL_SIZE} GPU(s) for tensor parallelism.")

# --- vLLM Initialization ---

try:
    # Initialize the LLM engine using the LLM class
    # trust_remote_code=True is necessary for many newer models like Llama 3
    # dtype="auto" lets vLLM choose the best precision (like bfloat16 if supported)
    # The first time you run this, it will download the model weights, which may take time.
    start_init_time = time.time()
    llm = vllm.LLM(
        model=MODEL_ID,
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        trust_remote_code=True, # Essential for Llama 3 models
        dtype="auto",           # Use 'bfloat16' or 'float16' if 'auto' causes issues
        gpu_memory_utilization=0.7, # Optional: Adjust GPU memory usage (0.0 to 1.0)
        # max_model_len=4096,      # Optional: Set maximum sequence length if needed
    )
    end_init_time = time.time()
    print(f"LLM engine initialized successfully in {end_init_time - start_init_time:.2f} seconds.")

except Exception as e:
    print(f"Error initializing LLM engine: {e}")
    print("Troubleshooting tips:")
    print("- Ensure you have accepted the model's terms on Hugging Face Hub.")
    print("- Try running 'huggingface-cli login' in your terminal.")
    print("- Check if you have sufficient GPU memory.")
    print("- Verify CUDA and driver compatibility with vLLM.")
    exit()

# --- Sampling Parameters ---

# Configure sampling parameters for generation
# See vLLM documentation for more options: https://docs.vllm.ai/en/latest/dev/sampling_params.html
sampling_params = vllm.SamplingParams(
    temperature=0.6,        # Controls randomness (0.0-1.0+). Lower = more deterministic.
    top_p=0.9,              # Nucleus sampling (0.0-1.0). Considers tokens cumulative probability >= top_p.
    # top_k=50,             # Alternative/additional: Considers top K probable tokens (-1 to disable).
    max_tokens=512,         # Max number of tokens to GENERATE (doesn't include prompt).
    # stop=["<|eot_id|>"],  # Specify stop tokens relevant to the model. Llama 3 uses <|eot_id|>.
                            # Adding this prevents the model from generating past its natural end-of-turn.
    frequency_penalty=0.05, # Small penalty for repeating tokens frequently.
    presence_penalty=0.0,   # Penalty for tokens already present (0 ensures no penalty).
)
print(f"Using sampling parameters: {sampling_params}")


# --- Prompts ---

# List of prompts to feed to the model.
# For Instruct models, it's best practice to format prompts according to the model's chat template.
# Llama 3 Instruct format:
# <|begin_of_text|><|start_header_id|>system<|end_header_id|>
#
# {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
#
# {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
#
# {assistant_response}<|eot_id|>  <- Model generates from here onwards

prompts = [
    # Simple completion prompt (less ideal for instruct models)
    "The capital of Canada is",

    # Formatted prompt using Llama 3 Instruct Template
    (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        "Explain the main benefits of using the vLLM library for LLM inference in three bullet points.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ),

    # Another formatted prompt
    (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        "Write a short Python function that demonstrates how to calculate the Fibonacci sequence up to the nth number recursively.<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
]
print(f"\n--- Running Inference on {len(prompts)} prompts ---")

# --- Generate Text ---

try:
    # Run inference using the llm.generate() method
    # This handles batching automatically if multiple prompts are provided.
    start_gen_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_gen_time = time.time()
    print(f"Inference generation completed in {end_gen_time - start_gen_time:.2f} seconds.")

except Exception as e:
    print(f"Error during text generation: {e}")
    exit()

# --- Print Outputs ---

print("\n--- Inference Results ---")
total_prompt_tokens = 0
total_generated_tokens = 0

for i, output in enumerate(outputs):
    prompt = output.prompt
    # The generated text is in output.outputs[0].text
    # There might be multiple outputs per prompt if n > 1 in SamplingParams
    generated_text = output.outputs[0].text
    prompt_tokens = len(output.prompt_token_ids)
    generated_tokens = len(output.outputs[0].token_ids)
    total_prompt_tokens += prompt_tokens
    total_generated_tokens += generated_tokens

    print(f"\n--- Output {i+1} ---")
    print(f"Prompt (Tokens: {prompt_tokens}):")
    print(f"{prompt!r}") # Use !r to clearly show start/end and special chars like \n
    print(f"\nGenerated Text (Tokens: {generated_tokens}):")
    print(f"{generated_text!r}")
    print("-" * 20)

# Calculate and print performance metrics
total_time = end_gen_time - start_gen_time
total_tokens = total_prompt_tokens + total_generated_tokens
overall_tokens_per_sec = total_generated_tokens / total_time if total_time > 0 else 0
avg_latency = total_time / len(prompts) if len(prompts) > 0 else 0

print("\n--- Performance Summary ---")
print(f"Total time taken for generation: {total_time:.2f} seconds")
print(f"Number of prompts processed: {len(prompts)}")
print(f"Total prompt tokens: {total_prompt_tokens}")
print(f"Total generated tokens: {total_generated_tokens}")
print(f"Average latency per prompt: {avg_latency:.3f} seconds")
print(f"Throughput (generated tokens/sec): {overall_tokens_per_sec:.2f} tokens/sec")
print("\nScript finished.")