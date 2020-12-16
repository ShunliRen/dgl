# GAT(norm.adj.) + label reuse + self KD for ogbn-arxiv

This repository contains the code to reproduce the performance of "GAT(norm.adj.) + label reuse + self KD" on ogbn-arxiv dataset. Codes of this repo is modified based on [Espylapiza's GAT implement](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv).

All experiments were runned with a GeForce RTX 1080Ti with 11GB memory.

## Learning with Self-KD

Self-KD means Knowledge Distillation where the student model is the same as teacher model.

We firstly train a pretrained teacher model(GAT(norm.adj.) + label reuse) with [previous implement](https://github.com/Espylapiza/dgl/tree/master/examples/pytorch/ogb/ogbn-arxiv), and then use logits of the teacher model and ground truth labels to train a student model with KD loss and CE loss. The architecture of the student model is the same as teacher model.


## Usage

Firstly, train the teacher model and you should see the result of the teacher;

```
python mkteacher.py --gpu 0 --no-attn-dst --use-norm --edge-drop 0.3 --input-drop 0.25 --use-labels --n-label-iters 1  
```
Then, train student model with Self-KD and you should see the final result.

```
python train_kd.py --gpu 0 --no-attn-dst --use-norm --edge-drop 0.3 --input-drop 0.25 --use-labels --n-label-iters 1 --alpha 0.95 --temp 0.7 
```

### Option

```
usage: GAT on OGBN-Arxiv [-h] [--cpu] [--gpu GPU] [--n-runs N_RUNS] [--n-epochs N_EPOCHS] [--use-labels] [--n-label-iters N_LABEL_ITERS] [--no-attn-dst]
                         [--use-norm] [--lr LR] [--n-layers N_LAYERS] [--n-heads N_HEADS] [--n-hidden N_HIDDEN] [--dropout DROPOUT] [--input-drop INPUT_DROP]
                         [--attn-drop ATTN_DROP] [--edge-drop EDGE_DROP] [--wd WD] [--log-every LOG_EVERY] [--plot-curves]

optional arguments:
  -h, --help            show this help message and exit
  --cpu                 CPU mode. This option overrides --gpu. (default: False)
  --gpu GPU             GPU device ID. (default: 0)
  --n-runs N_RUNS       running times (default: 10)
  --n-epochs N_EPOCHS   number of epochs (default: 2000)
  --use-labels          Use labels in the training set as input features. (default: False)
  --n-label-iters N_LABEL_ITERS
                        number of label iterations (default: 0)
  --no-attn-dst         Don't use attn_dst. (default: False)
  --use-norm            Use symmetrically normalized adjacency matrix. (default: False)
  --lr LR               learning rate (default: 0.002)
  --n-layers N_LAYERS   number of layers (default: 3)
  --n-heads N_HEADS     number of heads (default: 3)
  --n-hidden N_HIDDEN   number of hidden units (default: 250)
  --dropout DROPOUT     dropout rate (default: 0.75)
  --input-drop INPUT_DROP
                        input drop rate (default: 0.1)
  --attn-drop ATTN_DROP
                        attention dropout rate (default: 0.0)
  --edge-drop EDGE_DROP
                        edge drop rate (default: 0.0)
  --wd WD               weight decay (default: 0)
  --log-every LOG_EVERY
                        log every LOG_EVERY epochs (default: 20)
  --plot-curves         plot learning curves (default: False)
  --alpha               ratio of kd loss
  --temp                temperature of kd
```

## Results

We follow the instruction of OGB rules and set the random seed from 0~9 and get the following results:

```
Val Accs: [0.7512332628611699, 0.7517030772844726, 0.7514010537266351, 0.7517366354575656, 0.7510319138226115, 0.7510990301687976, 0.7508976811302392, 0.7522735662270545, 0.7514010537266351, 0.7512668210342629]
Test Accs: [0.7426496306812337, 0.7415180132913606, 0.7428553792975742, 0.7416414624611649, 0.7422587083101866, 0.7410653663354114, 0.7419706602473098, 0.7414974384297266, 0.7405304199329259, 0.7403863959014876]
Average val accuracy: 0.7514044095439444 ± 0.0003862690040867831
Average test accuracy: 0.741637347488838 ± 0.0007846444080142637
Number of params: 1441580
```
