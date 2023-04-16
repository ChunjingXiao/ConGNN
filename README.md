ConDA
============


A denoising diffusion probabilistic model-based Data Augmentation method for anomaly detection. 

ConDA can generate augmented data which can keep the category information of the source sample while inclining to the reference sample for enhancing detection performance.


## Requirements
python==3.7.3
pytorch>=1.4
dgl-cuda11.6
sklearn>=0.20.1
numpy>=1.16
networkx>=2.1

## Usage
### Download all files.
git clone https://github.com/CodeToPublic/Anomaly.git

### How to train 


>$For\ the\ cora\ dataset$

**Run the main file**

```
python main.py --dataset cora --module GAE --nu 0.1 --lr 0.001 --batch-size 64  --n-hidden 64 --n-layers 2 --weight-decay 0.0005 --n-epochs 1000 --early-stop
```

**Run the generate file**

```
python ddpm/feature_train.py --config ddpm/config/cora_train.json

python ddpm/feature_test.py --config ddpm/config/cora_test.json
```

**Run the main file**

```
python main.py --dataset cora --module GraphSAGE --nu 0.1 --lr 0.001 --batch-size 128 --n-hidden 64 --n-layers 2 --weight-decay 0.0005 --n-epochs 1000 --early-stop
```

>$For\ the\ citeseer\ dataset$

**Run the main file**

```
python main.py --dataset citeseer --module GAE --nu 0.1 --lr 0.001 --batch-size 64 --n-hidden 64 --n-layers 2 --weight-decay 0.0005 --n-epochs 1000 --early-stop
```

**Run the generate file**

```
python ddpm/feature_train.py --config ddpm/config/citeseer_train.json

python ddpm/feature_test.py --config ddpm/config/citeseer_test.json
```

**Run the main file**

```
python main.py --dataset citeseer --module GraphSAGE --nu 0.1 --lr 0.001 --batch-size 128 --n-hidden 64 --n-layers 2 --weight-decay 0.0005 --n-epochs 1000 --early-stop
```

>$For\ the\ pubmed\ dataset$

**Run the main file**

```
python main.py --dataset pubmed --module GAE --nu 0.1 --lr 0.001 --batch-size 64 --n-hidden 64 --n-layers 2 --weight-decay 0.0005 --n-epochs 1000 --early-stop
```

**Run the generate file**

```
python ddpm/feature_train.py --config ddpm/config/pubmed_train.json

python ddpm/feature_test.py --config ddpm/config/pubmed_test.json
```

**Run the main file**

```
python main.py --dataset pubmed --module GCN --nu 0.1 --lr 0.001 --batch-size 128 --n-hidden 64 --n-layers 2 --weight-decay 0.0005 --n-epochs 200 --early-stop
```
## Folder descriptions

*dateset:* This is used to process datasets and divide training, verification and test datasets.

*networks:* This is used to store the graph neural network model used in training.

*train:* This is used for training and loss calculation.

*utils:* This is  some tool classes.

*ddpm:* Relevant documents of DDPM are stored here


