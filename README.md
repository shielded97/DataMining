# COMP 6130 - Graduate Group 01 - Final Project <br />Aidan Lambrecht, Alex Norris, Kyle Lesinger
For this project we analyzed/implemented work done here https://arxiv.org/pdf/1906.04214.pdf. Along with the authors' source code here https://github.com/KaidiXu/GCN_ADV_Train, we extended the application of the GCN by writing scripts to handle new datasets (create_dataset.py & add_cats.py) and modified the attack.py script. See the following instructions below on how to prepare your workspace and train the models yourself.


## Setting Up Your Environment
1. Run within WSL or native linux env
2. Install miniconda (python3.6/3.7)
    * `wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.8.3-Linux-x86_64.sh -O ~/miniconda.sh`
    * `sh miniconda.sh`
3. Start conda env (restart terminal)
    * `conda activate`
4. Install dependencies (`conda install *PckgName*`)
    * numpy - `conda install numpy`
    * scipy - `conda install scipy`
    * tensorflow (Make sure it is TensorFlow1.14, 2 will not work) - `conda install -c conda-forge tensorflow` 
    * If AMD CPU - `conda install nomkl`
    * May require further Python modules like scipy or matplotlib. Install with pip or packg manager of your choice

## Build Canvas Dataset
1. Run the add_cats.py script to add categories to each node in the canvas dataset found in data/CA-GrQc.txt
2. Run the create_dataset.py script to make the Cavnas dataset
3. Ensure all the Canvas files are placed in the 'data' folder

## Training and Attack Steps
1. Start by training a natural model for your preferred dataset
    * `python train.py`
    * On line 22 of 'train.py' you may change the flag for one of the 3 available datasets (cora default)
    * natural cora model is saved in 'nat_cora' dir
	* cmd line code: python train.py --dataset=cora'
2. Next, train a robust model from the above natural model
    * `python adv_train_pgd.py` make sure to change the flag for your desired dataset
    * robust cora model is saved in 'rob_cora'
	* cmd line code: python adv_train_pgd.py --dataset=cora'
3. Finally, run your prefferred attack on the whatever model you like
    * you may run one of two attacks (PGD and CW), at a time on one selected trained model (natural or robust)
    * `python attack.py --model_dir=nat_cora --method=PGD` for performing a PGD attack on the natural cora model
    * `python attack.py --model_dir=rob_cora --method=CW --hidden1=64` for performing a CW attack on the robust cora model
	
## Original code has been added to the attack.py script in the codebase. It consists of information printed to the console as well as an original evaluation function called 'hamming()' that not only evaluates Hamming Loss but also F1 Micro and Macro score. 