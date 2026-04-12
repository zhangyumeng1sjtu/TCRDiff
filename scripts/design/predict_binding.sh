DESIGN_PIPELINE_DIR='design_pipeline/MAGE-A3'
MODEL_CONFIG='logs/tcr-pmhc-binding-subsample-2/config.yml'


python scripts/predict/predict_tcr_pmhc_binding.py \
    --config ${MODEL_CONFIG} \
    --gpu_device 1 \
    --input_data_path ${DESIGN_PIPELINE_DIR}/MAGE-A3_cdr3_generation.csv \
    --saveto ${DESIGN_PIPELINE_DIR}/MAGE-A3_cdr3_generation_binding

python scripts/predict/predict_tcr_pmhc_binding.py \
    --config ${MODEL_CONFIG} \
    --gpu_device 1 \
    --input_data_path ${DESIGN_PIPELINE_DIR}/sampled_bg_df.csv \
    --saveto ${DESIGN_PIPELINE_DIR}/sampled_bg_df_binding


off_target_peptide='ESDPIVAQY'
python scripts/predict/predict_tcr_pmhc_binding.py \
    --config ${MODEL_CONFIG} \
    --gpu_device 1 \
    --input_data_path ${DESIGN_PIPELINE_DIR}/MAGE-A3_cdr3_generation_${off_target_peptide}.csv \
    --saveto ${DESIGN_PIPELINE_DIR}/MAGE-A3_cdr3_generation_${off_target_peptide}_binding

python scripts/predict/predict_tcr_pmhc_binding.py \
    --config ${MODEL_CONFIG} \
    --gpu_device 1 \
    --input_data_path ${DESIGN_PIPELINE_DIR}/sampled_bg_df_${off_target_peptide}.csv \
    --saveto ${DESIGN_PIPELINE_DIR}/sampled_bg_df_${off_target_peptide}_binding
