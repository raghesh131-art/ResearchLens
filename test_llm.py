from llama_cpp import Llama

model_path = "models/qwen2.5-0.5b-instruct-q4_k_m.gguf"

llm = Llama(
    model_path=model_path,
    n_ctx=2048,
    verbose=False
)

prompt = """<|im_start|>system
You are a helpful assistant. Answer briefly.
<|im_end|>
<|im_start|>user
What is the Internet of Things?
<|im_end|>
<|im_start|>assistant
"""

output = llm(
    prompt,
    max_tokens=100,
    temperature=0.2
)

answer = output["choices"][0]["text"]

print("\nANSWER:")
print(answer)