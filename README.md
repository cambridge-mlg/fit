# FiT: Parameter Efficient Few-shot Transfer Learning for Personalized and Federated Image Classification

This repository contains the code to reproduce the experiments carried out in:
[FiT: Parameter Efficient Few-shot Transfer Learning for Personalized and Federated Image Classification](https://arxiv.org/pdf/2206.08671.pdf)

## Dependencies
This code requires the following:
* Python 3.8 or greater
* PyTorch 1.11 or greater (most of the code is written in PyTorch)
* TensorFlow 2.8 or greater (for reading VTAB datasets)
* TensorFlow Datasets 4.5.2 or greater (for reading VTAB datasets)
* [gsutil](https://cloud.google.com/storage/docs/gsutil_install) (for downloading the [The Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset) dataset)

## GPU Requirements
* The majority of the experiments in the paper are executed on a single NVIDIA A100 GPU with 80 GB of memory. By reducing the batch size, it is possible to run on a GPU with less memory, but classification results may be different.

## Installation
The following steps will take a considerable length of time and disk space.
1. Clone or download this repository.
2. The VTAB-v2 benchmark uses [TensorFlow Datasets](https://www.tensorflow.org/datasets). The majority of these are
   downloaded and pre-processed upon first use. However, the
   [Diabetic Retinopathy](https://www.tensorflow.org/datasets/catalog/diabetic_retinopathy_detection)
   and [Resisc45](https://www.tensorflow.org/datasets/catalog/resisc45) datasets need to be
   downloaded manually. Click on the links for details.
3. Switch to the ```src``` directory in this repo and download the BiT pretrained model:

   ```wget https://storage.googleapis.com/bit_models/BiT-M-R50x1.npz```
5. For the federated learning experiments on [The Quick, Draw!](https://github.com/googlecreativelab/quickdraw-dataset)
   dataset, download the dataset as follows: 
   ```
   mkdir quickdraw-npy 
   gsutil -m cp gs://quickdraw_dataset/full/numpy_bitmap/*.npy quickdraw-npy
   ```

## Usage
Switch to the ```src``` directory in this repo and execute any of the commands below.

### Few-shot
**<ins>1-shot<ins>**:

```python run_fit.py --classifier <qda, lda, or protonets> --examples_per_class 1 -i 0 --mode few_shot -c <path to checkpoint directory> --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded>```

**<ins> > 1-shot<ins>**:

```python run_fit.py --classifier <qda, lda, or protonets> --examples_per_class <2-10, or -1 for all> --mode few_shot -c <path to checkpoint directory> --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded>```

### VTAB-1k
```python run_fit.py --classifier <qda, lda, or protonets> --mode vtab_1000 --do_not_split -c <path to checkpoint directory> --download_path_for_tensorflow_datasets <path to where you want the TensorFlow Datasets downloaded>```

### Federated Learning
```
python run_fed_avg.py --data_path <path to dataset> --checkpoint_dir <path to checkpoint directory> \
--num_local_epochs <number of local updates> --iterations <number communication rounds> \
--num_clients <number of classes> --num_classes <number of classes per client> \
--shots_per_client <shots per client> --dataset <quickdraw, cifar100> --use_npy_data
```

Alternatively, for CIFAR100 the bash script can be used:

```bash fed_avg_cifar100.sh $num_clients $num_shots_per_client $data_path $checkpoint_dir```

and for QuickDraw:

```bash fed_avg_quickdraw.sh $num_clients $num_shots_per_client $data_path $checkpoint_dir```

Other hyperparameters in these scripts are set to the values used for the federated learning experiments in the paper.

## Contact
To ask questions or report issues, please open an issue on the issues tracker.

## Citation
If you use this code, please cite our [paper](https://arxiv.org/pdf/???.pdf).
```
@inproceedings{shysheya2022fit,
  title={FiT: Parameter Efficient Few-shot Transfer Learning for Personalized and Federated Image Classification},
  author={Shysheya, Aliaksandra and Bronskill, John and Patacchiola, Massimiliano and Nowozin, Sebastian and Turner, Richard E.},
  journal={arXiv preprint arXiv:2206.08671},
  year={2022}
}
```
