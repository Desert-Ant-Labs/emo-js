# Model assets

This example bundles the model into the app, so place the three model files here
before running:

```bash
hf download desert-ant-labs/emo \
  emo.safetensors emo_tokenizer.bin emo_meta.json \
  --local-dir .
```

(Requires the Hugging Face CLI: `pip install -U "huggingface_hub[cli]"`.)

The expected files are:

- `emo.safetensors`
- `emo_tokenizer.bin`
- `emo_meta.json`

They are git-ignored here (see `.gitignore`) so the binaries aren't committed.
