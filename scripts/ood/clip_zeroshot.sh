# Benchmark: OES
# Model: CLIP-B/16 & RSCLIP-B/32
# Method: Multiple Detection Methods
# Task: Out-of-Distribution Detection (Zero-shot)

bash scripts/ood/det_oes_clip-b16_mls.sh
bash scripts/ood/det_oes_clip-b16_msp.sh
bash scripts/ood/det_oes_clip-b16_mcm.sh
bash scripts/ood/det_oes_clip-b16_glmcm.sh

bash scripts/ood/det_oes_clip-rsb32_mls.sh
bash scripts/ood/det_oes_clip-rsb32_msp.sh
bash scripts/ood/det_oes_clip-rsb32_mcm.sh
bash scripts/ood/det_oes_clip-rsb32_glmcm.sh
