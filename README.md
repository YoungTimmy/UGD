# Implementation for UGD in CIKM-short 2024
You Can't Ignore Either: Unifying Structure and Feature Denoising for Robust Graph Learning


## Requirements

To install requirements:

```
pip3 install -r requirements.txt
```

## Generate_noisedata
Feature noise generation:
```
python generate_noisedata/attack_injection_noise.py
```

Structure noise generation:

For Cora and Citeseer dataset,
```
python generate_noisedata/attack_Mettack.py 
```
For other datasets,
```
python generate_noisedata/attack_PRBCD.py 
```
## Denoise
```
python denoise/denoise.py 
```
