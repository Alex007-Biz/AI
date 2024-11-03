from transformers import pipeline

classifier = pipeline("zero-shot-classification")
result = classifier(
    "This is a course about climate change",
    candidate_labels=["education", "politics", "ecology"],
)
print(result)