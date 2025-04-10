## **Maintainer: Yifan Liang**

currently **zero-shot CLIP** and **CoOp** can be reproduced, but results are different from the OES paper.

# clip
main differences are in model.py
- [ ] `clip-origin`
- [ ] `clip-dpm`: return global and local image features
- [ ] `clip-locoop`: return global and local image features(with MaskCLIP style)
- [ ] `clip-maple`: with learnable prompts in ViT

# customclip.py
- [x] `TextEncoder`
- [ ] `TextEncoder_Maple`
- [x] `CustomCLIP_ZeroShot`
- [ ] `CLIP_scoring` 
- [x] `CustomCLIP` = `TextEncoder` + `PromptLearner`
- [ ] `CustomCLIP-CALIP` 
- [ ] `CustomCLIP-Maple` = `TextEncoder` + `MultiModalPromptLearner`
- [ ] `CustomCLIP-LoCoOp` = `TextEncoder` + `PromptLearner`
- [ ] `CustomCLIP-NegPrompt` = `TextEncoder` + `NegPromptLearner`(later, cannot be reproduced)
- [ ] `CustomCLIP-DPM` = `TextEncoder` + `MLCPromptLearner` + `DPM_Block` + `DPM_Proj1` + `DPM_Proj2`

# promptlearner.py
- [x] `PromptLearner`
- [ ] `MultiModalPromptLearner`
- [ ] `NegPromptLearner` (later, cannot be reproduced)
- [ ] `MLCPromptLearner` (the same as PromptLearner which only allows cls_token at the end, can be merged to PromptLearner later)

# coop.py
- [x] `CoOp` = `CustomCLIP` + `get_class_names` + `load_clip_to_cpu` + `get_text_features`
- [ ] `LoCoOp` -- `CustomCLIP_LoCoOp`
- [ ] `Maple` -- `CustomCLIP_Maple`
- [ ] `NegPrompt` -- `Custom_NegPrompt`(later, cannot be reproduced)

# clip_utils.py
- [x] `classes and templates`
- [x] `load_clip_to_cpu`
- [ ] `load_clip_to_cpu_maple`
- [x] `get_text_features`