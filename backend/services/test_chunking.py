
from chunking import chunk_text

sample_text = """
Artificial Intelligence is an important technology.

Machine Learning is a subset of Artificial Intelligence.

Deep Learning uses neural networks to process complex data.
""" * 30

chunks = chunk_text(sample_text)

print("Total number of chunks:", len(chunks))

for i, chunk in enumerate(chunks, start=1):
    print(f"\nChunk {i}:")
    print(chunk[:150])
