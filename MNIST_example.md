# Graphcore MNIST example for TensorFlow

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

## Running the cpu example on the IPU
