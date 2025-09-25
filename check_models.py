import google.generativeai as genai

# Replace with your actual Gemini API key
genai.configure(api_key="AIzaSyA6ZytX3n1ISvbpi3rNbeqI56DJZRvLY68")  # <-- Insert your key here

# List available models
for model in genai.list_models():
    print(f"Model name: {model.name}")
    print(f"Supports generation: {model.supports('generateContent')}")
    print("-" * 30)
