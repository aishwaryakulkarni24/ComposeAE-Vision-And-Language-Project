# ComposeAE - Vision-And-Language-Course-project
An autoencoder based model, ComposeAE, to learn the composition of image and text query for retrieving images.

# Compositional Learning of Image-Text Query for Image Retrieval : WACV 2021
The paper can be accessed at: https://arxiv.org/pdf/2006.11149.pdf. This is the code accompanying the WACV 2021 paper: Compositional Learning of Image-Text Query for Image Retrieval.

## Requirements and Installation
* Python 3.6
* [PyTorch](http://pytorch.org/) 1.2.0
* [NumPy](http://www.numpy.org/) (1.16.4)
* [TensorBoard](https://github.com/TeamHG-Memex/tensorboard_logger)
* Other packages can be found in [requirements.txt](https://github.com/ecom-research/ComposeAE/blob/master/requirements.txt)


## Description of the Code 
- `main.py`: driver script to run training/testing
- `datasets.py`: Dataset classes for loading images & generate training retrieval queries
- `text_model.py`: LSTM model to extract text features
- `img_text_composition_models.py`: various image text compostion models 
- `torch_function.py`: contains soft triplet loss function and feature normalization function
- `test_retrieval.py`: functions to perform retrieval test and compute recall performance

- ## Running the experiments 

### Download the datasets
#### MITStates dataset

Download the dataset via this [link](http://web.mit.edu/phillipi/Public/states_and_transformations/index.html) and save it in the ``data`` folder. Kindly take care that the dataset should have these files:

```data/mitstates/images/<adj noun>/*.jpg```

## Running the Code

For training and testing new models, pass the appropriate arguments. 

For instance, for training the model on MITStates dataset run the following command:

```
python -W ignore  main.py --dataset=mitstates --dataset_path=../data/mitstates/  --model=composeAE --loss=soft_triplet --learning_rate_decay_frequency=50000 --num_iters=160000 --weight_decay=5e-5 --comment=mitstates_tirg_original --log_dir ../logs/mitstates/
```

## Notes:
### Running the BERT model
ComposeAE uses pretrained BERT model for encoding the text query. 
Concretely, the project employs BERT-as-service and use Uncased BERT-Base which outputs a 768-dimensional feature vector for a text query. 
Detailed instructions on how to use it, can be found [here](https://github.com/hanxiao/bert-as-service).
It is important to note that before running the training of the models, BERT-as-service should already be running in the background.


### Citation
```
@InProceedings{Anwaar_2021_WACV,
    author    = {Anwaar, Muhammad Umer and Labintcev, Egor and Kleinsteuber, Martin},
    title     = {Compositional Learning of Image-Text Query for Image Retrieval},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1140-1149}
}
```
