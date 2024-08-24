from transformers import pipeline

# the downloaded cached model is available at ~./cache/huggingface/hub
# Use the CLI to manage cache using the TUI at https://huggingface.co/docs/huggingface_hub/main/en/guides/cli
# Within the terminal in this activated environment
# https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache

# Mode 1 using the pipeline interface

# Text classfication
text = "I would go to this restaurant even if you woke me from my sleep."

# specify only task, default model selected
# pipe1 = pipeline("text-classification") 

# specify which model to use, which is then downloaded and cached
# sentiment in terms of many different emotions, sadness, anger, inspiration, approval, etc
pipe1 = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions", top_k=5)  # top 5 emotions
print(pipe1(text))

