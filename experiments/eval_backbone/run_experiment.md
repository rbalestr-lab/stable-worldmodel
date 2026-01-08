# run

```bash
python experiments/eval_backbone/run.py policy=pyro_test_epoch_20 world.max_episode_steps=200
```

## Backbone eval
```bash
python experiments/eval_backbone/run.py policy=epoch_20/resnet50_epoch_20,epoch_20/vit_mae_base_epoch_20,epoch_20/vit_base_epoch_20,epoch_20/siglip2_base16_224_epoch_20,epoch_20/dinov2_small_epoch_20,epoch_20/dino_vits16_epoch_20,epoch_20/dinov3_vits16_epoch_20,epoch_20/ijepa_vith14_22k_epoch_20 seed=42,3072,9261 launcher=lucas_eval
```

## Encoder scaling
```bash
python experiments/eval_backbone/run.py -m policy=epoch_20/dinov2_small_epoch_20,epoch_20/pusht_dinov2_base_psmall_epoch_20,epoch_20/pusht_dinov2_large_psmall_epoch_20,epoch_20/pusht_dinov2_giant_psmall_epoch seed=42,3072,9261 launcher=lucas_eval
```

## Predictor scaling
```bash
python experiments/eval_backbone/run.py -m policy=epoch_20/pusht_dinov2_small_ptiny_epoch_20,epoch_20/pusht_dinov2_small_psmall_epoch_20,epoch_20/pusht_dinov2_small_pbase_epoch_20,epoch_20/pusht_dinov2_small_plarge_epoch_20 seed=42,3072,9261 launcher=lucas_eval
```

# TwoRoom (Normal eval)

## All Backbone

```bash
python experiments/eval_backbone/run.py policy=17296730_9/tworoom_resnet50_psmall_epoch_20,17296730_10/tworoom_siglip2_base16_224_psmall_epoch_20,17296730_8/tworoom_vit_mae_base_psmall_epoch_20,17296730_11/tworoom_vit_base_psmall_epoch_20,17296730_7/tworoom_ijepa_vith14_22k_psmall_epoch_20,17296730_12/tworoom_vjepa2_large_psmall_epoch_20,17296730_0/tworoom_dino_vits16_psmall_epoch_20,17296730_2/tworoom_dinov2_giant_psmall_epoch_20,17296730_3/tworoom_dinov2_large_psmall_epoch_20,17296730_1/tworoom_dinov2_base_psmall_epoch_20,17296730_5/tworoom_dinov2_small_psmall_epoch_20 seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/all_backbones.txt"
```

## Encoder scaling
```bash
python experiments/eval_backbone/run.py -m policy=17296766_0/tworoom_dinov2_small_psmall_epoch_20, 17296766_1/tworoom_dinov2_base_psmall_epoch_20,17296766_2/tworoom_dinov2_large_psmall_epoch_20,17296766_3/tworoom_dinov2_giant_psmall_epoch_20 seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/encoder_scaling.txt"
```

## Predictor scaling
```bash
python experiments/eval_backbone/run.py -m policy=17296777_0/tworoom_dinov2_small_ptiny_epoch_20,17296777_1/tworoom_dinov2_small_psmall_epoch_20,17296777_2/tworoom_dinov2_small_pbase_epoch_20,17296777_3/tworoom_dinov2_small_plarge_epoch_20 seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/predictor_scaling.txt"
```

## Data all variations:
```bash
python experiments/eval_backbone/run.py -m policy=17296789/tworoom_dinov2_small_psmall_epoch_20 seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/data_all_variations.txt"
```

## Data Scaling

```bash
python experiments/eval_backbone/run.py -m policy=17296800_0/tworoom_dinov2_small_psmall_epoch_20,17296800_1/tworoom_dinov2_small_psmall_epoch_20,17296800_2/tworoom_dinov2_small_psmall_epoch_20,17296800_3/tworoom_dinov2_small_psmall_epoch_20, seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/data_scaling.txt"
```

## Interaction Quality Interpolation

#### weak:
```bash
python experiments/eval_backbone/run.py -m policy=17296803_0 tworoom_dinov2_small_psmall_epoch_20 seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/interaction_data_quality_interp.txt"
```

####

## Quality Interpolation

```bash
python experiments/eval_backbone/run.py -m policy=17296807_0/tworoom_dinov2_small_psmall_epoch_20,17296807_1/tworoom_dinov2_small_psmall_epoch_20,17296807_2/tworoom_dinov2_small_psmall_epoch_20,17296807_3/tworoom_dinov2_small_psmall_epoch_20,17296807_4/tworoom_dinov2_small_psmall_epoch_20 seed=42,3072,9261 launcher=lucas_eval output.filename="res/two_room/data_quality_interp.txt"
```


## Variations Interpolation (loss blow up explode!!!)

17296810_0/tworoom_dinov2_small_psmall_epoch_20,
17296810_1/tworoom_dinov2_small_psmall_epoch_20,
17296810_2/tworoom_dinov2_small_psmall_epoch_20,
