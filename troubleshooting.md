# Errors

1. SHA256 checksum does not not match

```shell
  File "whisperx/vad.py", line 47, in load_vad_model
    raise RuntimeError(
RuntimeError: Model has been downloaded but the SHA256 checksum does not not match. Please retry loading the model.
```

resolution:

```shell
rm ~/.cache/torch/whisperx-vad-segmentation.bin
```
