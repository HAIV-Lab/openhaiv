# Benchmark: OES
# Model: Multiple CLIP-based Models
# Method: Multiple Detection Methods
# Task: Out-of-Distribution Detection (With Training)

bash scripts/ood/det_oes_coop-b16_mcm.sh
bash scripts/ood/det_oes_locoop-b16_glmcm.sh
bash scripts/ood/det_oes_sct-b16_glmcm.sh
bash scripts/ood/det_oes_dpm-b16_dpm.sh

bash scripts/ood/det_oes_coop-rsb32_mcm.sh
bash scripts/ood/det_oes_locoop-rsb32_glmcm.sh
bash scripts/ood/det_oes_sct-rsb32_glmcm.sh
bash scripts/ood/det_oes_dpm-rsb32_dpm.sh
