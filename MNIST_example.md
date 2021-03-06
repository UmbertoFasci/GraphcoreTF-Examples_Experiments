# Graphcore MNIST example work-through for TensorFlow

## File architecture

When implementing a run utilizing the GraphcorexSpell integration, the creation of three separate files has been noted:
1. `main.py`: Containing the main code to run.
2. `model.py`: Containing the actual keras architecture for model generation.
3. `utils.py`: Containing the functions which can load, preprocess data, and parsers to generate IPU specific arguments to the run.

## Notes while running the completed example:

The model run with the `completed_example` was streamlined and optimized given the additional IPU specific arguments which will be personally experimented with later on.
Additionally, I experienced a rapid increase in model accuracy per epoch in comparison to CPU/GPU assisted runs. This can also be seen within the example when you compare the runs between the `completed_model` and the CPU leveraged model `mnist_cpu.py`.

## About the IPU specific arguments

These arguments defined in the parser function in `utils.py` such as the `--use-ipu` and `--pipelining` flags allow the user to run the Keras model on the cloud IPU and adopt the pipelining feature respectively. On an interesting note, the gradient accumulation count can be adjusted with the `--gradient-accumulation-count` flag.

## Running the CPU example on the IPU

### Import the TensorFlow IPU module
The first action to take when applying Graphcore's cloud IPU to this example is to include a new import:
```python
from tensorflow.python import ipu
```
**NOTE:** The `ipu` module must be imported directly rather than accessing it through the top-level Tensorflow module to function properly.

## Preparing the dataset
The next action to take is to prepare the dataset in a fashion where the IPU can recieve appropriately sized tensors, as the IPU framework which leverages the Poplar software stack does not support using tensors with shapes that are not known when the model is compiled. To successfully complete this preperation, the sizes of the datasets must be divisible by the batch size. The following code allows for this:
```python
def make_divisible(number, divisor):
    return number - number % divisor
    
train_data_len = x.train.shape[0]
train_data_len = make_divisible(train_data_len, batch_size)
x_train, y_train = x_train[:train_data_len], y_train[:train_data_len]

test_data_len = x_test.shape[0]
test_data_len = make_divisible(test_data_len, batch_size)
x_test, y_test = x_test[:test_data_len], y_test[:test_data_len]
```
After this is implemented, we lose 32 training examples and 48 evaluation examples to make these datasets usable in this IPU framework.

Future experiments will implement different approaches to preparing the data for training a Keras maodel on the IPU. These approaches include creating a `tf.data.Dataset` object which utilizes the data and calling the `.repeat()` method to create a looped version of the dataset. Also, padding the datasets with tensors of zeros (_will experiment with several padding configurations_), and setting the `sample_weight` to be a vector of 1's and 0's according to which values are real.

## Adding IPU configuration

As stated in the Graphcore example. to use the IPU, you must create an IPU session configuration. The following code accomplishes this:
```python
ipu_config = ipu.config.IPUConfig()
ipu_config.auto_select_ipus = 1
ipu_config.configure_ipu_system()
```
## Specify IPU strategy
```python
strategy = ipu.ipu_strategy.IPUStrategy
```
This strategy is a subclass of the `tf.disribute.strategy` which targets a system with one or more IPUs attached.

## Wrap the model within the IPU strategy scope

Wrapping the model within the scope of the `IPUStrategy` will ensure that they are placed on the IPU. The following code allows this:
```python
with strategy.scope():
    # Model.__init__ takes two required arguments, inputs and outputs.
    model = keras.Model(*model_fn())

    # Compile our model with Stochastic Gradient Descent as an optimizer
    # and Categorical Cross Entropy as a loss.
    model.compile('sgd', 'categorical_crossentropy', metrics=["accuracy"])

    model.summary()
    print('\nTraining')
    model.fit(x_train, y_train, epochs=3, batch_size=64)
    print('\nEvaluation')
    model.evaluate(x_test, y_test)
```
Above, the function `model_fn()` is already defined and readily reused within the IPU strategy wrapper.

## Exaluating the model's performance compared to its original CPU configuration.
### Hardware Metrics
**CPU Only**
![CPU only](https://github.com/UmbertoFasci/GraphcoreTF-Examples_Experiments/blob/main/GraphcoreMNISTExmpAssets/HardwareMetricsCPU.png)
**Complete Example Leveraging IPU**
![Complete Example IPU Leveraged](https://github.com/UmbertoFasci/GraphcoreTF-Examples_Experiments/blob/main/GraphcoreMNISTExmpAssets/HardwareMetricsComplete.png)
**CPU Example with IPU Wrapper**
![CPU example with IPU Wrapper](https://github.com/UmbertoFasci/GraphcoreTF-Examples_Experiments/blob/main/GraphcoreMNISTExmpAssets/HardwareMetricsIPU_no_parser_functions.png)
#
