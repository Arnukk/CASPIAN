import tensorflow as tf
import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA as sk_PCA
from smt.surrogate_models import KRG
from tensorflow.keras import backend as K


def set_seed(seed: int = 42) -> None:
  #random.seed(seed)
  np.random.seed(seed)
  tf.random.set_seed(seed)
  tf.experimental.numpy.random.seed(seed)
  #tf.set_random_seed(seed)
  # When running on the CuDNN backend, two further options must be set
  os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
  os.environ['TF_DETERMINISTIC_OPS'] = '1'
  # Set a fixed value for the hash seed
  os.environ["PYTHONHASHSEED"] = str(seed)
  print(f"Random seed set to {seed}")


class PrintLearningRate(tf.keras.callbacks.Callback):
    def __init__(self, **kwargs):
        super(PrintLearningRate, self).__init__(**kwargs)

    def on_epoch_begin(self, epoch, logs=None):
        lr = self.model.optimizer.lr(self.model.optimizer.iterations)
        print(f"Epoch {epoch+1}: Learning rate is {lr:.6f}.")
  
class LinearDecayPerEpoch(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, steps_per_epoch, total_epochs, end_learning_rate=0.0001):
        super(LinearDecayPerEpoch, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.steps_per_epoch = steps_per_epoch
        self.total_epochs = total_epochs
        self.end_learning_rate = end_learning_rate
        self.total_steps = steps_per_epoch * total_epochs
        self.decay_rate = (initial_learning_rate - end_learning_rate) / (total_epochs)

    def __call__(self, step):
        epoch = tf.floor(step / self.steps_per_epoch)
        new_learning_rate = self.initial_learning_rate - epoch * self.decay_rate
        new_learning_rate = tf.maximum(new_learning_rate, self.end_learning_rate)
        return new_learning_rate

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "steps_per_epoch": self.steps_per_epoch,
            "total_epochs": self.total_epochs,
            "end_learning_rate": self.end_learning_rate
        }

      
class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """Warmup learning rate scheduler
    """
    # https://www.dlology.com/blog/bag-of-tricks-for-image-classification-with-convolutional-neural-networks-in-keras/  
    
    def __init__(self, warmup_batches, init_lr, verbose=0):
        """Constructor for warmup learning rate scheduler
        Arguments:
            warmup_batches {int} -- Number of batch for warmup.
            init_lr {float} -- Learning rate after warmup.
        Keyword Arguments:
            verbose {int} -- 0: quiet, 1: update messages. (default: {0})
        """

        super(WarmUpLearningRateScheduler, self).__init__()
        self.warmup_batches = warmup_batches
        self.init_lr = init_lr
        self.verbose = verbose
        self.batch_count = 0
        self.learning_rates = []

    def on_batch_end(self, batch, logs=None):
        self.batch_count = self.batch_count + 1
        lr = K.get_value(self.model.optimizer.lr)
        self.learning_rates.append(lr)

    def on_batch_begin(self, batch, logs=None):
        if self.batch_count <= self.warmup_batches:
            lr = self.batch_count*self.init_lr/self.warmup_batches
            K.set_value(self.model.optimizer.lr, lr)
            if self.verbose > 0:
                print('\nBatch %05d: WarmUpLearningRateScheduler setting learning '
                      'rate to %s.' % (self.batch_count + 1, lr))
      
def clear_ds(f,x,y,yf, one_d=False):
    
    nx = tf.expand_dims(x, -1)
    if not one_d:
        ny = tf.expand_dims(y, -1)
    else:
        ny = yf
    return nx,ny

  
def rmv_aug(f,x,y,yf):
    return not tf.strings.regex_full_match(f, ".*rnd.*")

def find_scenario(scenario):
    def find(f,x,y,yf):
      return tf.equal(f, scenario)
    return find

def custom_rmse(mask):
    def rmse_score(y_true, y_pred):
        # applying the mask
        true_result = tf.boolean_mask(y_true, mask)
        pred_result = tf.boolean_mask(y_pred, mask)

        # Use the mask to compute the RMSE only where y_true is not equal to 1
        squared_error = tf.square(true_result - pred_result)
        mean_squared_error = tf.reduce_mean(squared_error)
        root_mean_squared_error = tf.sqrt(mean_squared_error)
        return root_mean_squared_error

    return rmse_score


def custom_mae(mask):
    def mae_score(y_true, y_pred):
        # applying the mask
        true_result = tf.boolean_mask(y_true, mask)
        pred_result = tf.boolean_mask(y_pred, mask)

        # Use the mask to compute the MAE only where y_true is not equal to 1
        absolute_error = tf.abs(true_result - pred_result)
        mean_absolute_error = tf.reduce_mean(absolute_error)
        return mean_absolute_error
    return mae_score


def custom_loss(mask):
    def loss(y_true, y_pred):
        # applying the mask
        true_result = tf.boolean_mask(y_true, mask)
        pred_result = tf.boolean_mask(y_pred, mask)
        # Use the mask to compute the loss only where y_true is not equal to 1
        Loss = tf.keras.losses.Huber(delta=0.5)(true_result, pred_result)
        return Loss
    return loss

def get_the_mask():
    mask = np.load("./ad_full_grid_1024.npy", allow_pickle=True).item()
    print("Length of the masked points: %d" % len(mask))
    the_mask = np.zeros((1024, 1024), dtype=bool)
    idx = np.array(list(mask.values()))
    the_mask [idx[:, 0], idx[:, 1]] = True
    the_mask = the_mask.T
    return the_mask


def delta_score(y_true, y_pred, val):
  a = np.abs(y_true - y_pred)
  delta = (a > val).sum(axis=1)/a.shape[1]*100
  return delta


def artae_score(y_true, y_pred):
  a = np.sum(np.abs(y_true - y_pred), axis=1)
  b = np.sum(np.abs(y_true), axis=1)
  return 100.*a/b


def acc0(y_true, y_pred):
    zero_mask = y_true == 0

    correct_zero_predictions = (y_pred == 0) & zero_mask

    # Calculate the percentage
    percentage = 100 * np.sum(correct_zero_predictions, axis=1) / np.sum(zero_mask, axis=1)

    return percentage

  
def rearrange_dimensions(x,y):
    # The new order of dimensions will be [batch_size, channels, height, width]
    return tf.transpose(x, perm=[0, 3, 1, 2]), tf.transpose(y, perm=[0, 3, 1, 2])
  
def expand_input_channels(x,y):
    #create masks for each condition
    x = tf.squeeze(x)
    
    # Create a mask for each color channel
    red_mask = tf.cast(tf.equal(x, 1), tf.float32)
    green_mask = tf.cast(tf.equal(x, 0.5), tf.float32)
    black_mask = tf.zeros_like(x) 

    # Stack the masks along the last axis to create an RGB image
    color_image = tf.stack([red_mask, green_mask, black_mask], axis=-1)
    
    return color_image, y
  
def expand_input_channels_full(f,x,y, yf):
    #create masks for each condition
    x = tf.squeeze(x)
    
    # Create a mask for each color channel
    red_mask = tf.cast(tf.equal(x, 1), tf.float32)
    green_mask = tf.cast(tf.equal(x, 0.5), tf.float32)
    black_mask = tf.zeros_like(x) 

    # Stack the masks along the last axis to create an RGB image
    color_image = tf.stack([red_mask, green_mask, black_mask], axis=-1)
    
    return f, color_image, y, yf
  
def rearrange_dimensions_full(f,x,y, yf):
    # The new order of dimensions will be [batch_size, channels, height, width]
    return f, tf.transpose(x, perm=[2, 0, 1]), tf.transpose(y, perm=[2, 0, 1]), yf

def PCA(Y, retain=0.999):
  # Create a PCA object and fit the data
  pca = sk_PCA(n_components=retain)
  pca.fit(Y)
  reduced = pca.transform(Y)
  print ("PCA reduced the dimension from %s to %s" % (Y.shape, reduced.shape))
  return reduced, pca.components_, pca.mean_

def Kriging(X, Z, corr='squar_exp', poly='linear'):
  sm = KRG(theta0=[1e-2], poly=poly, corr=corr, print_global=False, print_problem=False)
  sm.set_training_values(X, Z)
  sm.train()
  return sm

def tfDS_to_np (split=1, baseline=False):
  dss = {
    'train': tf.data.Dataset.load("./data/train_ds_split_%d" % split),
    'test': tf.data.Dataset.load("./data/test_ds_split_%d" % split),
    'val': tf.data.Dataset.load("./data/val_ds_split_%d" % split),
    'holdout': tf.data.Dataset.load("./data/holdout_dataset"),
  }
  trainds = dss['train'].concatenate(dss['val'])
  xdata = []
  ydata = []
  for element in list(trainds.as_numpy_iterator()): 
      scenario, input_map, output, output_flat = element
      if baseline:
        scenario = input_map
      else:
        scenario = [int(i) for i in str(scenario.decode('UTF-8'))[:17]]
      xdata.append(scenario)
      ydata.append(output_flat)
      
  xtest = []
  ytest = []
  for element in list(dss["test"].as_numpy_iterator()): 
      scenario, input_map, output, output_flat = element
      if baseline:
        scenario = input_map
      else:
        scenario = [int(i) for i in str(scenario.decode('UTF-8'))[:17]]
      xtest.append(scenario)
      ytest.append(output_flat)
      
  xhold = []
  yhold = []
  for element in list(dss["holdout"].as_numpy_iterator()): 
      scenario, input_map, output, output_flat = element
      if baseline:
        scenario = input_map
      else:
        scenario = [int(i) for i in str(scenario.decode('UTF-8'))[:17]]
      xhold.append(scenario)
      yhold.append(output_flat)
  return np.array(xdata), np.array(ydata), np.array(xtest), np.array(ytest), np.array(xhold), np.array(yhold)

def dataset_average_pwl(ds):
    return np.mean(ds[ds > 0])

def simple_predictor(scenarios, avg):
    # Initialize the list to hold all prediction vectors
    all_predictions = []
    the_mask = get_the_mask()
    # Iterate over each scenario in the batch
    for scenario in scenarios:
        scenario = scenario[the_mask]
        output_vector = []
        for i in scenario:
            if i == 0.5:
                output_vector.append(0)
            else:
                output_vector.append(avg)
        # Add the output vector to the list of all predictions
        assert len(output_vector) == np.sum(the_mask)
        all_predictions.append(output_vector)

    return np.array(all_predictions)


  

