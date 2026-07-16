## TCRDiff: a conditional diffusion model for designing antigen-specific TCR CDR3

<!-- This repository contains the source code for the preprint paper []() -->

TCRDiff generates antigen-specific TCRs conditioned on peptide-MHC targets and germline-encoded V-genes via the denoising diffusion process. It was pre-trained on large T-cell repertoires and TCR-pMHC recognition datasets. The generated CDR3 sequences exhibited superior consistency with natural binding TCRs and structural plausibility of the designed TCR-pMHC complexes.

![model](./model.png)

### Installation

1. Clone the repository

   ```Shell
   git clone https://github.com/zhangyumeng1sjtu/TCRDiff.git
   cd TCRDiff/
   ```
2. Create a virtual environment by conda

   ```Shell
   conda create -n tcrdiff_env python=3.12.2
   conda activate tcrdiff_env
   ```
3. Download PyTorch>=2.4.1, which is compatible with your CUDA version and other Python packages

   ```Shell
   conda install pytorch==2.4.1 pytorch-cuda=12.4 -c pytorch -c nvidia
   pip install -r requirements.txt
   ```

### Download data and model checkpoints

The following data and model checkpoints are available at [Zenodo](https://doi.org/10.5281/zenodo.20586708).

- `data/pep`: pre-trained peptide sequences.
- `data/pmhc`: pre-trained peptide-MHC binding data and MHC sequences.
- `data/tcr`: pre-trained TCRs in T-cell repertoires.
- `data/tcrpmhc`: TCR-pMHC recognition dataset.
- `logs/tcr-pmhc-binding-subsample-2/`: checkponits of TCRDiff binding predictor.
- `logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/`: checkponits of TCRDiff generative model.

### Usage

#### Design antigen-specific TCR

Try the demo in `scripts/design/` to generate MAGE-A3-epitope (EVDPIGHLY) specific TCRs (CDR3), predict their binding probabilities, and filter the results.

```Shell
./scripts/design/generate_cdr3.sh
python scripts/design/prepare_binding_inputs.py
./scripts/design/predict_binding.sh
```

Now you can find the generated TCRs and binding prediction results in `design_pipeline/`.

Finally, you can filter the generated TCRs based on the predicted binding rank percentiles for further investigation as in `scripts/design/MAGE_A3_analysis.ipynb`.

| ID  | CDR3A           | CDR3B         | Peptide   | MHC | TRAV      | TRBV       | Organism | Pred     | On target percentile | Off target percentile |
| --- | --------------- | ------------- | --------- | --- | --------- | ---------- | -------- | -------- | -------------------- | --------------------- |
| 128 | CAVRPSNFANEKLTF | CASSLDGRDGYTF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.220928 | 91.68                | 30.08                 |
| 263 | CAVPNSGTGADGLTF | CASSWGLDNEQFF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.238115 | 92.96                | 48.04                 |
| 386 | CAVGEFGGGSEKFTF | CASSLGGGYEQYF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.250346 | 93.51                | 59.24                 |
| 398 | CAVRGSNFGNEKLTF | CASSLAGDYGYTF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.433676 | 98.44                | 52.54                 |
| 465 | CAVPNSGGSNYKLTF | CASSLAGNYGYTF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.216015 | 91.42                | 36.89                 |
| 530 | CAVRPFKAAGNKLTF | CASSWGGDQPQHF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.240003 | 93.08                | 50.05                 |
| 538 | CAVRGYSGANSKLTF | CASSWGNSYEQYF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.217942 | 91.50                | 58.23                 |
| 709 | CAVMNSGGGADGLTF | CASSWGADQPQHF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.260770 | 93.98                | 57.71                 |
| 791 | CAVDLMKTSYGKLIF | CASRPDGGYGYTF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.211343 | 90.99                | 53.54                 |
| 798 | CAVNPSNFGNERLTF | CASSLGGSYEYTF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.322026 | 96.24                | 42.74                 |
| 994 | CAVPNSGGSNYKLTF | CASSLGGGQPQHF | EVDPIGHLY | HLA-A*01:01 | TRAV21*01 | TRBV5-1*01 | human    | 0.319059 | 96.15                | 58.37                 |

In  summary, the design pipeline consists of these steps:

- Condaitional generation of CDR3 sequences

  ```Shell
  python scripts/generate/generate_pmhc_binding_tcr.py \
      --config logs/tcr-pmhc-cond-dplm-cross-attn-finetune-tcr-dplm-all-constant/config.yml \
      --num_seqs <num_generated_seq> \ # e.g., 1000
      --alpha_seq_len <cdr3_alpha_len> \ # e.g., 15
      --beta_seq_len <cdr3_beta_len> \ # e.g., 13
      --peptide <peptide_seq> \ # e.g., EVDPIGHLY
      --mhc <mhc> \ # e.g.m HLA-A\*01:01
      --mhca <mhc_alpha> \ # e.g., A\*01:01
      --mhcb <mhc_beta> \ # e.g., b2m
      --trav <trav_gene> \ # e.g., TRAV21\*01
      --trbv <trbv_gene> \ # e.g., TRBV5-1\*01
      --organism <oragnism> \ # human or mouse
      --temperature <sampling_temperature> \ # e.g., 0.1
      --sampling_strategy gumbel_argmax \
      --max_iter <sampling_itermations> \ # e.g., 10
      --gpu_device <gpu_device_num> \
      --saveto <output_path>
  ```
- Preprare inputs for binding prediction of the generated TCRs

  Refer to `scripts/design/prepare_binding_inputs.py`, use your peptide-MHC (Optional: off-target peptide), V-gene, and generated outputs instead.
- Predict binding probabilites and rank percentiles to filter TCR designs

  Refer to `scripts/design/predict_binding.sh` and `scripts/design/MAGE_A3_analysis.ipynb`, use your customized binding prediction inputs/outputs, off-target peptide, and binding rank threshold instead.

#### Train TCRDiff from scratch

- Pre-train peptide language model and peptide-MHC binding model

  ```Shell
  python scripts/pretrain/pretrain_peptide_lm.py --config configs/config-pretrain-peptide-lm.yml
  python scripts/train/train_pmhc_model.py --config configs/config-train-pmhc-model.yml
  ```
- Pre-train TCR diffusion language model

  ```Shell
  python scripts/pretrain/pretrain_tcr_lm.py --config configs/config-pretrain-tcr-lm.yml
  python scripts/pretrain/pretrain_tcr_dplm.py --config configs/config-pretrain-tcr-dplm.yml
  ```
- Train TCR-pMHC binding predictor

  ```Shell
  python scripts/train/train_tcr_pmhc_model.py --config configs/config-train-tcr-pmhc-model.yml
  ```
- Train conditional TCR diffusion model (TCRDiff)

  ```Shell
  python scripts/train/train_tcr_pmhc_cond_dplm.py --config configs/config-train-tcr-pmhc-dplm.yml
  ```

<!-- ### Citation -->

### Contact

If you have any questions, please contact us at [yumeng.zhang1@monash.edu](mailto:yumeng.zhang1@monash.edu) or [jiangning.song@monash.edu](mailto:jiangning.song@monash.edu).
