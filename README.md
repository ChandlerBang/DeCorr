# DeCorr
[KDD 2022] Implementation of ["Feature Overcorrelation in Deep Graph Neural Networks: A New Perspective"](https://arxiv.org/abs/2206.07743)

<!--
Code will be updated soon. Stay Tuned :) **If you cannot wait, feel free to directly email me (jinwei2@msu.edu).**
-->


Abstract
----
Oversmoothing has been identified as one of the key issues which limit the performance of deep GNNs. In this work, we propose a new perspective to look at the performance degradation of deep GNNs, i.e., feature overcorrelation. Through empirical and theoretical study on this matter, we demonstrate the existence of feature overcorrelation in deeper GNNs and reveal potential reasons leading to this issue. To reduce the feature correlation, we propose a general framework DeCorr which can encourage GNNs to encode less redundant information. 


## Requirements
The experiments are performed with `Python=3.7.5`. The required Python packages can be founded in `requirements.txt`.
```
numpy==1.19.1
ogb==1.2.3
pandas==1.1.2
scikit_learn==1.1.2
scipy==1.5.2
torch==1.2.0
torch_cluster==1.4.5
torch_geometric==1.3.2
torch_scatter==1.4.0
tqdm==4.49.0
```

Note that this repository is based on DGN (https://github.com/Kaixiong-Zhou/DGN). You may also refer to their page to see the package dependencies. 


## Run our code 
Clone our code
```
git clone git@github.com:ChandlerBang/DeCorr.git
cd DeCorr
```
Then install the required Python packages.

To run our code, we can simply use the following command:
```
python main.py --dataset=Cora --type_model=GCN --alpha=1 --beta=10 --dropout=0 --lr=0.01 --epoch=1000 --cuda_num=1
```

## Reproduce the performance
For table 1, we can run the experiments with the following command:
```
bash scripts/table1.sh
```
which includes the hyper-parameter setup for our method.


## Acknowledgement
This repository is based on the code from DGN (https://github.com/Kaixiong-Zhou/DGN). We sincerely thank the authors for their contributions.

## Cite
For more information, you can take a look at the [paper](https://arxiv.org/abs/2206.07743).

If you find this repo to be useful, please cite our paper. Thank you.
```
@inproceedings{jin2022feature,
 author = {Wei Jin and Xiaorui Liu and Yao Ma and Charu Aggarwal and  Jiliang Tang},
 booktitle = {Proceedings of the 28th {ACM} {SIGKDD}  Conference on
 Knowledge Discovery and Data Mining},
 title = {Feature Overcorrelation in Deep Graph Neural Networks: A New Perspective},
 year = {2022}
}
```
