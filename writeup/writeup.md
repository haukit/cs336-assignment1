# BPE Tokenizer

## 2.1 The Unicode Standard

### `unicode1`

- `chr(0)` returns `\x00`, which is the null character. 
- The string representation (`chr(0)`) is the escape sequence `\x00`, where the `\x` part is the escape sequence for representing a character by its hex value, and `00` is the hex value itself. The printed representation (`print(chr(0))`) shows no visible output. 
- If the null character appears in text, 
    - The string representation shows its escape sequence, e.g., `'this is a test\x00string'`.
    - The printed representation does not produce any visible output, e.g., `this is a teststring`.

### `unicode2`

- UTF-8 is preferred because it represents most common characters in fewer bytes than UTF-16 or UTF-32. This makes the byte sequences shorter, so the tokenizer produces fewer tokens overall, which reduces sequence length, speeds up training, and simplifies the vocabulary.
- UTF-8 does not map each byte directly to a Unicode code point. Instead, code points may be represented by sequences of 1 to 4 bytes. If you try to decode the input one byte at a time, you will often be interpreting only part of a multi-byte sequence. This produces incorrect characters or even invalid output, because you’re breaking up the intended encoding units.
- The two-byte sequence `\0x80\0x80` does not decode to any Unicode characters, since the first byte `\x80` is a "continuation byte" and thus cannot be used to start a sequence.