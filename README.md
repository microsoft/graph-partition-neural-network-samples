# GPNNs
This is the Tensorflow implementation of Graph Partition Neural Networks as described in the following paper:

```
@article{liao2018graph,
  title={Graph Partition Neural Networks for Semi-Supervised Classification},
  author={Renjie Liao and Marc Brockschmidt and Daniel Tarlow and Alexander~L. Gaunt and Raquel Urtasun and Richard Zemel},
  journal={arXiv preprint arXiv:1803.06272},
  year={2018},
}
```


## Setup
To set up experiments of NELL/DIEL, please run the following scripts:
```
./setup_nell.sh
./setup_diel.sh
``` 
Note that since these two datasets are large, it may take a while to finish.
You may also need to switch to CPU mode before running demos by:
```
export CUDA_VISIBLE_DEVICES=
```


## Dependencies
tensorflow(>= 1.0), numpy, scipy, sklearn


## Run Demos
* To run experiments ```X``` where ```X``` is one of {citeseer, cora, pubmed, nell, diel}:

   ```python run_exp.py -c config/gpnn_X.json```

* For experiments on NELL, you can specify the label rate by changing ```label_rate```	of ```config/gpnn_nell.json```
* For experiments on DIEL, you can specify the split id by changing ```split_id```	of ```config/gpnn_diel.json```
	

## Hyper-parameters
We list some notable hyper-parameters here and you can refer to configuration files under the ```config``` folder for more details.
* ```num_pass```: # Propagation pass
* ```num_cluster```: # Clusters
* ```prop_step_intra```: # Propagation step within cluster
* ```prop_step_inter```: # Propagation step between clusters
* ```decomp_method```: one of {"spectral_cluster", "multi_seed_flood_fill"}
* ```hidden_dim```: dimension of state vector
* ```aggregate```: aggregation method, one of {"avg", "sum", "min", "max"}
* ```msg_type```: message function type, one of {"msg_embedding", "msg_mlp"}
* ```update_type```: update function type, one of {"GRU", "MLP"}
* ```update_MLP_dim```: hidden dimension of update MLP
* ```update_MLP_act```: activation function of update MLP, one of {"relu", "tanh", "sigmoid", null}
* ```output_MLP_dim```: similar to ```update_MLP_dim```
* ```output_MLP_act```: similar to ```update_MLP_act```

**Notes**:
* An example of specifying a 2 hidden layer output MLP with "tanh" as activation function:  
   ```output_MLP_dim = [128, 128]```, ```output_MLP_act = ["tanh", "tanh"]```.
* We also provide an easy-to-use implementation of LSTM in ```nn_cells.py``` which could potentially be used as update function. We will support this feature soon.
* By setting ```num_cluster = 1``` and ```update_type = "GRU"```, the resultant model is roughly the same as [Gated Graph Neural Network](https://arxiv.org/abs/1511.05493) except the input and output models are slightly different from the ones described in the original paper.


## Customized Usage
To use our code for your customized problem, you need to prepare the following (pickle) files:
* ```your_dataset.graph```: graph, python dictionary, key = node id, value = list of neighbor ids, id ranges from 0 to N-1.
* ```your_dataset.feature```: feature, N by D numpy array (D is the number of features per node).
* ```your_dataset.label```: label, N numpy array (integer label within range(C), C is the number of classes).
* ```your_dataset.split```: mask of split, N numpy array (0, 1, 2 stand for train/validation/test respectively).
* Append the customized dataset information in ```config/dataset_info.json``` by setting ```label_size, feat_dim, num_nodes, num_valid, data_folder```.  

Please refer to ```gpnn/reader/gpnn_reader_custom.py``` for more information.


## Cite
Please cite our paper if you use this code in your research work:
```
Bibtex goes here!
```


## Questions/Bugs
Please submit a Github issue or contact rjliao@cs.toronto.edu if you have any questions or find any bugs.


## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.
