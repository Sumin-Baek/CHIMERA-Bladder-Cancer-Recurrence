# model/

Place the following files here before building the Docker image
(they are not tracked in git — see the root README, "Model weights"):

```
model/
├── vit_large_patch16_224.dinov2.uni_mass100k/
│   └── pytorch_model.bin          # UNI patch encoder (MahmoodLab/UNI on Hugging Face, gated)
├── bcg_best_model_epoch_9.pth      # submitted AdvancedBCGTransformerNet checkpoint
└── bcg_preprocessor.pkl            # fitted BCGDataPreprocessor (label encoders)
```
