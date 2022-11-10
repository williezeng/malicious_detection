# poshmal-ai
Code to train a deep neural network to detect malicious PowerShell scripts.

## Getting Started
*NOTE*: If you are already in a conda environment, `deactivate` your current conda environment. We will be using the environment specifically in poshmal-ai/environments to avoid packaging issues.

1. Install conda. You can check if you already have conda by typing `which conda` in the terminal. 
If you already have conda, then skip to step 2. 

2. Make sure you are in the source directory of the repository. The source directory will be the poshmal-ai/ directory.

### Windows
3. Create the same conda environment that we all use: (this will take a while)

`conda env create -f environments/baseline.yaml`

4. Activate the conda environment. The environment's name is "deep" (it is the first line in environments/baseline.yaml) 

`conda activate deep` 

5. Check that you installed the new environment correctly:

`conda env list`

### Linux / MacOS 

3. Create the bare bones conda environment. This should at least get you started on running the existing scripts

`conda env create -f environments/bare_bones.yml`

4. Activate the conda environment. The environment's name is "deep" (it is the first line in environments/baseline.yaml) 

`conda activate deep` 

5. Check that you installed the new environment correctly:

`conda env list`


6. You're environment is all set, so you're now ready to build, train, and evaluate models.

NOTE: You can exit the Conda environment at anytime by typing `deactivate` or `source deactivate`

7. Clone the poshmal dataset using this URL: https://github.gatech.edu/mlester30/poshmal.git

8. Run train.py. There are several options summarized below:

| Argument | Default | Description |
| -------- | ------- | ----------- |
| --datadir | ./../poshmal/ | A path to the directory where the poshmal dataset was cloned. |
| --config | ./configs/fcnn.yaml | A path to the configuration file to load. This file describes the model to train and the hyperparameters to use. |

```bash
python train.py
```

You should get the following output. The trainer will automatically save the best model after each epoch.

```
[*] Get a listing a files in ./../poshmal/clean.
[========================================================================] 100%
[*] Get a listing a files in ./../poshmal/malicious-unmodified.
[!] Required 5000 samples but only found 1253 samples. Padding subset with duplicates.
[========================================================================] 100%
[*] Get a listing a files in ./../poshmal/malicious-insert.
[========================================================================] 100%
[*] Get a listing a files in ./../poshmal/malicious-bypass-techniques-obfuscated.
[!] Required 5000 samples but only found 1675 samples. Padding subset with duplicates.
[========================================================================] 100%
[*] Get a listing a files in ./../poshmal/malicious-obfuscated.
[========================================================================] 100%
FCNN(
  (nn_layers): ModuleList(
    (0): Conv1d(1, 32, kernel_size=(3,), stride=(1,))
    (1): ReLU()
    (2): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Conv1d(32, 64, kernel_size=(3,), stride=(1,))
    (4): ReLU()
    (5): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (6): Conv1d(64, 128, kernel_size=(3,), stride=(1,))
    (7): ReLU()
    (8): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (9): Conv1d(128, 256, kernel_size=(3,), stride=(1,))
    (10): ReLU()
    (11): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (12): Conv1d(256, 384, kernel_size=(3,), stride=(1,))
    (13): ReLU()
    (14): MaxPool1d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (15): AdaptiveMaxPool1d(output_size=10)
    (16): Flatten(start_dim=1, end_dim=-1)
    (17): Dropout(p=0.5, inplace=False)
    (18): Linear(in_features=3840, out_features=4096, bias=True)
    (19): ReLU()
    (20): Linear(in_features=4096, out_features=2, bias=True)
  )
)
[*] Hardware: Cuda
[*] OOMS: 0     Allocated: 61.6694MiB   Reserved: 82.0000MiB
[*] Epoch: [0][1/125]   Time: 2.469 (2.469)     Loss: 0.0019 (0.0019)   Acc: 0.4844     Prec: 0.5041)   Rec: 0.7750     FPR: 0.7721
...similar entries ommitted
[*] Epoch: [0][125/125] Time: 1.321 (1.320)     Loss: 0.0029 (0.0026)   Acc: 0.6719     Prec: 0.6724)   Rec: 0.6512     FPR: 0.3071
[*] Epoch: [0][0/16]    Time: 0.632 (0.632)     Acccuracy: 0.6992       Precision: 0.7044       Recall: 0.5820  FPR: 0.1940
... similar entries ommitted
[*] Epoch: [0][14/16]   Time: 0.615 (0.629)     Acccuracy: 0.7031       Precision: 0.7082       Recall: 0.6412  FPR: 0.2320
[*] Accuracy of Class 0: 0.7407
[*] Accuracy of Class 1: 0.5739
[*] Prec @1: 0.6595
[*] Exporting best model to ./checkpoints/fcnn.pth
```

9. To validate your work, you can call scan.py to see if a single sample is malicious.

```bash
python scan.py -m "fcnn-384-02" -p "./../poshmal/malicious-unmodified/1B7DE918C7E5777F4E15DB4728A6E4EAFAE1CA59FB52571AB3EC8551D3ADD913"
```

You should get the following output.

```
[*] File:      ./../poshmal/malicious-unmodified/1B7DE918C7E5777F4E15DB4728A6E4EAFAE1CA59FB52571AB3EC8551D3ADD913
[*] Malicious: True
[*] Scores:    tensor([-4.1720,  3.7333])
```

10. You can run the sigfind code to see what the model identified as malicious. This code uses the delta debug algorithm to find the 1-minimal subset of the original script that still flags as malicious by the model.

```bash
python scan.py -s -p "./../poshmal/malicious-unmodified/1B7DE918C7E5777F4E15DB4728A6E4EAFAE1CA59FB52571AB3EC8551D3ADD913"
```

You should get the output below. In this case, the model identified the hex representation of an empty LANMAN hash, which is a reasonable signature.

```
[*] n: 2 length: 19335
[*] n: 2 length: 9668
[*] n: 2 length: 4834
[*] n: 2 length: 2417
[*] n: 2 length: 1208
[*] n: 2 length: 604
[*] n: 2 length: 302
[*] n: 2 length: 151
[*] n: 2 length: 76
[*] n: 2 length: 38
[*] n: 4 length: 38
[*] n: 8 length: 38
[*] n: 16 length: 38
[*] n: 32 length: 38
[*] n: 38 length: 38
[*] Analysis complete!
[*] Signaturized string:
[*] 3,0xb4,0x35,0xb5,0x14,0x04,0xee);
```