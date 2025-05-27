## Directory Structure
```  
T3Set/  
├─ models/            # A simple model to validate the usage of dataset 
│  ├─ src/        # src code
│  ├─ weights/     # Pre-trained weights 
│  ├─ requirements.txt     # required packages  
│  └─ README.md              # usage instructions
├─ scripts/           # Data processing & evaluation scripts  
│  ├─ data_scripts/  # scripts for building dataset (stroke detection, data alignment, text preprocessing)  
│  └─ eval_scripts/  # Benchmark testing script 
├─ README.md          # Project overview
└─ LICENSE            # License information

```  
### data_scripts
This directory contains the scripts we used for dataset construction and data cleaning processes. It includes implementations for stroke detection and aligning video and sensor data.

More details can be found in the `data_scripts/README.md`.
### eval_scripts
This directory contains the evaluation scripts used in our paper. These scripts perform evaluations for Stage 2 and the baseline, corresponding to Table 2 and Table 3 in the paper.

More details can be found in the `eval_scripts/README.md`.

## Additional Information

For a comprehensive view of the data construction and processing workflow, please refer to Chapter 3 of our paper.

## License
CC BY-NC-SA 4.0