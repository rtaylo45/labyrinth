# Labyrinth
Synthetic training data generation pipeline for object detection tasks. Inspired by [SynthText](https://github.com/ankush-me/SynthText), this project aims to assists ML training tasks which are burdened by limited data avaliability. Labyrinth provides tools to statically generate datasets for repeated use, or to dynamically generate training samples, reducing the training sets memory foot print.

The data generate pipeline is broken down into 3 simple steps:
* Background generation
* Sprite sampeling
* Sprite placement

Augmentations are also a key feature of Labyrinth, allowing users to independently augment backgrounds, sprites, and samples. Labyrinth also has the capability to dependently augment backgrounds based on inputes from the sprites. This allows for more domain specific applications.

# Examples
TODO: Point to examples folder and talk about them.

# Install
In the main directory run `pip install -Ue .`

Install pre-commit hooks:
* Install the `pre-commit` package in your local environment and run `pip install pre-commit`
* In main directory run `pre-commit install`
