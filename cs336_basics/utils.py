import random


def sample_documents(
    filepath: str,
    num_samples: int = 10,
    num_load: int | None = 30,
    separator: str | None = "<|endoftext|>",
    seed: int | None = None,
) -> list[str]:
    """Load up to n documents from file, then sample m documents from those."""
    if seed is not None:
        random.seed(seed)

    documents = []
    current_doc = ""

    with open(filepath, encoding="utf-8") as file:
        buffer = ""

        while True:
            # Stop if we've loaded enough documents
            if num_load is not None and len(documents) >= num_load:
                break

            # Read chunk
            chunk = file.read(8192)  # 8KB chunks
            if not chunk:
                # Handle the last document if file ends without <|endoftext|>
                if current_doc.strip():
                    documents.append(current_doc.strip())
                break

            buffer += chunk

            # Process complete documents in buffer
            while separator in buffer:
                doc_content, buffer = buffer.split(separator, 1)
                current_doc += doc_content

                if current_doc.strip():  # Only add non-empty documents
                    documents.append(current_doc.strip())

                current_doc = ""

                # Stop if we've loaded enough documents
                if num_load is not None and len(documents) >= num_load:
                    break

            # Keep partial document in current_doc
            current_doc += buffer
            buffer = ""

    # Sample m documents from the loaded documents
    if num_samples > len(documents):
        print(f"Warning: Requested {num_samples} samples but only {len(documents)} documents loaded. Returning all.")
        return documents

    sampled_docs = random.sample(documents, num_samples)
    return sampled_docs
