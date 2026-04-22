CUDA_VISIBLE_DEVICES=1 \
uv run python scripts/main.py fit \
-c cfg-new/data/data_ampeg-optocomp_trainval.yaml \
-c cfg-new/model/gb/gb_comp/model_gb_comp_peq.s+g.d+peq.s+g.s.yaml \
-c cfg-new/trainer/trainer_gb.yaml