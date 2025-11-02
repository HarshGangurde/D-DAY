#pip install transformers
from transformers import pipeline
gen = pipeline("text-generation", model="gpt2")
prompts = ["The doctor said", "The nurse said"]
outputs = [gen(p, max_new_tokens=10)[0]['generated_text'] for p in prompts]
print("Generated outputs:", outputs)
bias_words = ["he", "she", "his", "her"]
filtered = []
for text in outputs:
    if not any(b in text.lower().split() for b in bias_words):
        filtered.append(text)
print("\nFiltered (Bias-Mitigation) Outputs:", filtered)
print(f"\nBias reduction: {len(outputs)-len(filtered)} biased → {len(filtered)} unbiased")

import random
class Agent:
    def __init__(self, name):
        self.name = name
        self.energy = 5
    def act(self, other):
        if self.energy <= 0:
            return f"{self.name} is too tired to act."
        action = random.choice(["help", "attack", "rest"])
        if action == "help":
            self.energy -= 1
            other.energy += 2
            return f"{self.name} helps {other.name}."
        elif action == "attack":
            self.energy -= 2
            other.energy -= 2
            return f"{self.name} attacks {other.name}!"
        else:
            self.energy += 1
            return f"{self.name} rests."
a1 = Agent("Agent A")
a2 = Agent("Agent B")
for step in range(5):
    print(f"\nStep {step + 1}:")
    print(a1.act(a2))
    print(a2.act(a1))
    print(f"Status → {a1.name}: {a1.energy}, {a2.name}: {a2.energy}")
