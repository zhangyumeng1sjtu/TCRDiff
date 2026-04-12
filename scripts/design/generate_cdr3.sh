python scripts/generate/generate_pmhc_binding_tcr.py \
    --config logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/config.yml \
    --num_seqs 1000 \
    --alpha_seq_len 15 \
    --beta_seq_len 13 \
    --peptide EVDPIGHLY \
    --mhc HLA-A\*01:01 \
    --mhca A\*01:01 \
    --mhcb b2m \
    --trav TRAV21\*01 \
    --trbv TRBV5-1\*01 \
    --organism human \
    --temperature 0.1 \
    --sampling_strategy gumbel_argmax \
    --max_iter 10 \
    --gpu_device 1 \
    --saveto design_pipeline/MAGE-A3


python scripts/generate/generate_pmhc_binding_tcr.py \
    --config logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/config.yml \
    --num_seqs 1000 \
    --alpha_seq_len 15 \
    --alpha_seq CAVRPGGAGPFFVVF \
    --beta_seq_len 13 \
    --peptide EVDPIGHLY \
    --mhc HLA-A\*01:01 \
    --mhca A\*01:01 \
    --mhcb b2m \
    --trav TRAV21\*01 \
    --trbv TRBV5-1\*01 \
    --organism human \
    --temperature 0.1 \
    --sampling_strategy gumbel_argmax \
    --max_iter 10 \
    --gpu_device 1 \
    --saveto design_pipeline/MAGE-A3-beta
