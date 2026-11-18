
def chunk_text(text, chunk_size=1000, overlap=200):
    """
    Divide extracted document text into smaller chunks.

    chunk_size: Maximum approximate characters per chunk.
    overlap: Number of characters shared between chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("Overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(start + chunk_size, text_length)

        # Prefer ending at a word boundary
        if end < text_length:
            boundary = text.rfind(" ", start, end)

            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(end - overlap, start + 1)

    return chunks
