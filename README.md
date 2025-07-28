# VISTA-Map
ArXiv: <https://arxiv.org/abs/2507.01125>.

Repository for VISTA-Map, the voxel grid representation and geometric information gain presented in the paper. For all code associated with the original paper, please see <insert url here>. 

## Setup & Dependencies
If you have already followed the instructions in the main VISTA repo, there is no need to install the following dependencies, and you can skip to the "Testing the Info Gain" section. The following instructions are for setting up this repository as a stand-alone code. In this repository, we provide an example obj file of the [Stanford Bunny](https://graphics.stanford.edu/data/3Dscanrep/) from the Stanford University Computer Graphics Laboratory so that the information gain code can be run without ROS2 and without the SemanticSplatBridge repository <insert SemanticSplatBridge repo here>.

First, clone this repository in a directory of your choice: 
```bash
git clone git@github.com:StanfordMSL/VISTA-Map.git
```
Set up a conda environment for this project: 
```bash
conda create --name vistamap -y python=3.10

conda activate vistamap
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate

# Activate conda environment, and upgrade pip
conda activate vistamap
python -m pip install --upgrade pip

# PyTorch, Torchvision dependency
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

# CUDA dependency (by far the easiest way to manage cuda versions)
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

# Numpy 2.0 bricks the current install...
pip install numpy==1.26.3

# Fix ROS2 CMake version dep
conda install -c conda-forge gcc=12.1.0

# Navigate to the VISTA-Map repo and pip install
cd VISTA-Map
pip install -e .
```

## Testing the Info Gain
To generate a gif that visualizes the geometric information gain, run `test_gain.py`:
```
cd VISTA-Map
python test_gain.py
```
In the directory `VISTA-Map/vista_map/images` a gif of the coverage metric will be generated.

## Citation
In case anyone does uses VISTA-Map as a starting point for any research please cite this repository.

```
# --------------------------- VISTA ---------------------
@article{nagami2025vista,
    title={VISTA: Open-Vocabulary, Task-Relevant Robot Exploration with Online Semantic Gaussian Splatting}, 
    author={Keiko Nagami and Timothy Chen and Javier Yu and Ola Shorinwa and Maximilian Adang and Carlyn Dougherty and Eric Cristofalo and Mac Schwager},
    journal={arXiv preprint arXiv:2507.01125}
    year={2025},
}
```
