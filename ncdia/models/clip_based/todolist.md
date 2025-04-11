## **Maintainer: Yifan Liang**

currently **zero-shot CLIP** and **CoOp** can be reproduced, but results are different from the OES paper.

# clip
main differences are in model.py
- [x] `clip-origin`
- [x] `clip-dpm`: return global and local image features
- [x] `clip-locoop`: return global and local image features(with MaskCLIP style)
- [x] `clip-maple`: with learnable prompts in ViT

# customclip.py
- [x] `TextEncoder`
- [x] `TextEncoder-LoCoOp`
- [ ] `TextEncoder-Maple`
- [x] `CustomCLIP_ZeroShot`
- [ ] `CLIP_scoring` 
- [x] `CustomCLIP` = `TextEncoder` + `PromptLearner`
- [ ] `CustomCLIP-CALIP` 
- [ ] `CustomCLIP-Maple` = `TextEncoder` + `MultiModalPromptLearner`
- [ ] `CustomCLIP-VPT` 
- [ ] `CustomCLIP-LP`
- [x] `CustomCLIP-LoCoOp` = `TextEncoder` + `PromptLearner`
- [ ] `CustomCLIP-NegPrompt` = `TextEncoder` + `NegPromptLearner`(later, cannot be reproduced)
- [x] `CustomCLIP-DPM` = `TextEncoder` + `MLCPromptLearner` + `DPM_Block` + `DPM_Proj1` + `DPM_Proj2`

# promptlearner.py
- [x] `PromptLearner`
- [ ] `MultiModalPromptLearner`
- [ ] `NegPromptLearner` (later, cannot be reproduced)
- [x] `MLCPromptLearner` 

# coop.py
- [x] `CoOp` = `CustomCLIP` + `get_class_names` + `load_clip_to_cpu` + `get_text_features`
- [x] `LoCoOp` -- `CustomCLIP_LoCoOp`
- [ ] `Maple` -- `CustomCLIP_Maple`
- [ ] `NegPrompt` -- `Custom_NegPrompt`(later, cannot be reproduced)

# clip_utils.py
- [x] `classes and templates`
- [x] `load_clip_to_cpu`
- [ ] `load_clip_to_cpu_maple`
- [x] `load_clip_to_cpu_locoop`
- [x] `get_text_features`
  
# Loss
- [x] `LoCoOpLoss`
- [x] `SCTLoss`
- [x] `DPMLoss`
  
# OOD Scores
- [ ] `MCM`
- [ ] `GLMCM`
- [ ] `DPM`
- [ ] `NegLabel`