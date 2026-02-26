# txt → RP → inference，不保存 PNG 命令

## D-SE-Mambaout 
```bash
python end2end_rp_infer_benchmark.py \
  --bench e2e \
  --model mambaout \
  --dir_wind data_1d/0wind/test \
  --dir_manual data_1d/1manual/test \
  --dir_digger data_1d/2digger/test \
  --N 1000 --downsample 1 --eps 0.05 --steps 3 --block 512 \
  --weights /home/liu-liuliu/Postgraduate/DASMamba/MambaoutRP/D-SE-Mambaout/model_save/DASFinebest.pth \
  --batch_size 10 \
  --max_per_class 200 \
  --transform minimal \
  --amp \
  --out_dir e2e_results_test_compute
```

## SE-Mambaout

```bash
python end2end_rp_infer_benchmark.py \
  --bench e2e \
  --model dse_mambaout \
  --dir_wind data_1d/0wind/test \
  --dir_manual data_1d/1manual/test \
  --dir_digger data_1d/2digger/test \
  --N 1000 --downsample 1 --eps 0.05 --steps 3 --block 512 \
  --weights /home/liu-liuliu/Postgraduate/DASMamba/MambaoutRP/D-SE-Mambaout/model_save/DASFinebest.pth \
  --batch_size 10 \
  --max_per_class 200 \
  --transform minimal \
  --amp \
  --out_dir e2e_results_test_compute
```

## SE-Mambaout

```bash
python end2end_rp_infer_benchmark.py \
  --bench e2e \
  --model se_mambaout \
  --dir_wind data_1d/0wind/test \
  --dir_manual data_1d/1manual/test \
  --dir_digger data_1d/2digger/test \
  --N 500 --downsample 1 --eps 0.05 --steps 3 --block 512 \
  --weights /home/liu-liuliu/Postgraduate/DASMamba/MambaoutRP/SE-Mambaout/models_savebest.pth \
  --batch_size 10 \
  --max_per_class 200 \
  --transform minimal \
  --amp \
  --out_dir e2e_results_test_compute
```

## Mambaout

```bash
python end2end_rp_infer_benchmark.py \
  --bench e2e \
  --model mambaout \
  --dir_wind data_1d/0wind/test \
  --dir_manual data_1d/1manual/test \
  --dir_digger data_1d/2digger/test \
  --N 500 --downsample 1 --eps 0.05 --steps 3 --block 512 \
  --weights /home/liu-liuliu/Postgraduate/DASMamba/MambaoutRP/Mambaout/model_save/best.pth \
  --batch_size 10 \
  --max_per_class 200 \
  --transform minimal \
  --amp \
  --out_dir e2e_results_test_compute
```
