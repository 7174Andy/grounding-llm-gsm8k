import re
def normalize(text):
    return text.replace("’", "'").replace("‘", "'").strip().lower()

# Keywords for classifying thoughts
REFLECTION_PREFIXES = ["wait"]
REFLECTION_PHRASES = [
    "verify", "make sure", "hold on", "think again",
    "'s correct", "'s incorrect", "let me check", "seems right"
]

TRANSITION_PREFIXES = ["alternatively"]
TRANSITION_PHRASES = [
    "think differently", "another way", "another approach",
    "another method", "another solution", "another strategy",
    "another technique"
]


def contains_phrase(text, phrases):
    return any(re.search(r'\b' + re.escape(p) + r'\b', text) for p in phrases)


def classify_thought(thought: str) -> str:
    t = normalize(thought)
    
    for prefix in REFLECTION_PREFIXES:
        if t.startswith(prefix):
            return "reflection"
    for prefix in TRANSITION_PREFIXES:
        if t.startswith(prefix):
            return "transition"
    
    if contains_phrase(t, REFLECTION_PHRASES):
        return "reflection"
    if contains_phrase(t, TRANSITION_PHRASES):
        return "transition"

    return "execution"

def extract_and_classify_traces(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    all_classified = []

    for line in lines:
        stripped = line.strip()
        if stripped == "":
            continue

        lowered = stripped.lower()
        if lowered.startswith("answer:") or lowered.startswith("the correct answer"):
            continue

        label = classify_thought(stripped)
        all_classified.append(f"[{label}] {stripped}")

    with open(output_file, "w", encoding="utf-8") as f:
        for line in all_classified:
            f.write(line + "\n")

    print(f"✅ Parsed and classified {len(all_classified)} individual thoughts into {output_file}")


if __name__ == "__main__":
    extract_and_classify_traces(
        "./SEAL_extracted_thoughts_llama.txt",
        "./SEAL_classified_thoughts_llama.txt"
    )