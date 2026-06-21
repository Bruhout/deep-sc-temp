from datasets import load_dataset

ds = load_dataset("lucadiliello/bookcorpusopen", split="train")

target_bytes = 500 * 1024 * 1024
written = 0

with open("bookcorpusopen_500mb.txt", "w", encoding="utf-8") as out:
    for row in ds:
        text = row["text"].strip()
        if not text:
            continue

        chunk = text + "\n\n"
        encoded = chunk.encode("utf-8")

        if written + len(encoded) > target_bytes:
            break

        out.write(chunk)
        written += len(encoded)

print(f"Wrote {written / (1024 * 1024):.2f} MB")
