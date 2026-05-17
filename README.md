# Labyrinth
Synthetic training data generation pipeline for object detection tasks. Inspired by [SynthText](https://github.com/ankush-me/SynthText), this project aims to assist ML training tasks which are burdened by limited data availability. Labyrinth provides tools to statically generate datasets for repeated use, or to dynamically generate training samples, reducing the training sets memory footprint.

The data generate pipeline is broken down into 3 simple steps:
* Background generation
* Sprite sampling
* Sprite placement

Augmentations are also a key feature of Labyrinth, allowing users to independently augment backgrounds, sprites, and samples. Labyrinth also has the capability to dependently augment backgrounds based on inputs from the sprites. This allows for more domain specific applications.

## Definitions
| Term          | Definition |
| --------      | -------    |
| Target        | Object of interest you are trying to detect. Common examples would include, Dog, Cat, Car, and Human. |
| Background    | Image in which the object will be placed. |
| Sprite        | An RGBA representation of the target, where the target has no transparency and the original target's background is fully transparent. |
| Augmentation  | Typical image or pixel level augmentation, but this augmentation is applied independent of other steps in the generation pipeline. |
| Modifier      | Pixel/image level augmentation which is dependent on another step in the generation pipeline. A common example of this would be augmenting the background to match the color space of the sprite. |


# Examples
TODO: Point to examples folder and talk about them.

# Install
In the main directory run `uv sync` to create a virtual environment and install all dependencies.

To include PyTorch support, run `uv sync --extra torch`.

Install pre-commit hooks:
* Install the `pre-commit` package in your local environment and run `pip install pre-commit`
* In main directory run `pre-commit install`
