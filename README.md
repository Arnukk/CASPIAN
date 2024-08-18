# Deep Vision-Based Framework for Coastal Flood Prediction Under Climate Change Impacts and Shoreline Adaptations
 

This repository contains the complete source code and data for reproducing the results reported in the paper. The proposed framework and models were implemented in `tensorflow.keras` (v 2.1). The weights of all the trained DL models are included.

  
The implementations of the SWIN-Unet and Attention U-net were adapted from the [keras-unet-collection](https://github.com/yingkaisha/keras-unet-collection) repository of [Yingkai (Kyle) Sha](https://github.com/yingkaisha).


## Repository Structure

  

- `data` includes the raw data, as well as the datasets (in  `tf.data.Dataset` format) created from it, based on which the coastal flood prediction models were trained, validated and tested.

- `models` contains the implementation of the models (in `tensorflow.keras` v 2.1 ) along with the weights of the trained models (in `h5` format).

- `model_training.ipynb` provides a sample code for training Deep Vision-based coastal flood prediction models with the proposed  approach.

- `performance_evaluation.ipynb` includes a sample code for assessing the performance of the developed models and visualizing predictions (see also `Illustrations.ipynb`).

  

## Training From Scratch

  

To re-train the aforementioned three models (SWIN-Unet, Attention U-net, CASPIAN):

  

1: Open `model_training.ipynb`, select the model and define your desired hyperparameters:

```python

grid_size = 1024

AUTOTUNE = tf.data.AUTOTUNE

batch_size = 2

split = 1

output_1d = False

EPOCHS = 200

  

MODEL_NAME = "SWIN-Unet"

LR = 0.0008

MIN_LR = LR/10

WARMUP_EPOCHS = 20

```

2: Load the dataset:

```python

ds = {

'train': tf.data.Dataset.load("./data/train_ds_aug_split_%d" % split).map(lambda f,x,y,yf: tf.py_function(clear_ds,

inp=[f,x,y,yf, output_1d],

Tout=[tf.float32, tf.float32])),

'val': tf.data.Dataset.load("./data/val_ds_aug_split_%d" % split).map(lambda f,x,y,yf: tf.py_function(clear_ds,

inp=[f,x,y,yf, output_1d],

Tout=[tf.float32, tf.float32]))

}

```

In the current implementation, the training and validation datasets are assumed to be pre-augmented. To recreate these datasets run the `data/Dataset_construction.ipynb` notebook. For a more memory-efficient implementation the augmentation can be performed on the fly during the training by passing a data generator to the `model.fit()` function.

  

3: Select the remaining hyperparameters, callbacks and initiate the training:

```python

model.summary()

  

history_warmup = model.fit(ds['train'],

epochs=WARMUP_EPOCHS,

validation_data=ds['val'],

callbacks=[checkpoint, tensorboard_callback, warm_up_lr]) #PrintLearningRate()#reduce_lr#early_stop

  

model.load_weights("./models/trained_models/%s/initial/" % MODEL_NAME)

history = model.fit(ds['train'],

epochs=EPOCHS,

validation_data=ds['val'],

callbacks=[checkpoint, tensorboard_callback, early_stop, reduce_lr]) #PrintLearningRate()#reduce_lr#early_stop

  

model.load_weights("./models/trained_models/%s/initial/" % MODEL_NAME)

model.save("./models/trained_models/"+MODEL_NAME+"_split_{}".format(str(split)), save_format='h5')

```

  
## Applying the Trained Models:

  

To ensure the robustness of the results, the models were trained on three (randomly generated) data splits. To produce a flood inundation map with the trained models for a given *shoreline protection scenario*, select a split, load the corresponding weights of the chosen trained model, and provide the input *hypothetical flood susceptibility map*:

```python

model = tf.keras.models.load_model("./models/trained_models/"+MODEL_NAME+"_split_{}".format(str(split)), compile=False)

for sample in ds_eval['test'].as_numpy_iterator():

scenario, input_grid, label, label_flat = sample

pred = model.predict(input_grid)[0, :, :, 0]

```