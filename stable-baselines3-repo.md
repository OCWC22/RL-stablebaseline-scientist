Directory structure:
└── dlr-rm-stable-baselines3/
    ├── README.md
    ├── CITATION.bib
    ├── CODE_OF_CONDUCT.md
    ├── CONTRIBUTING.md
    ├── Dockerfile
    ├── LICENSE
    ├── Makefile
    ├── NOTICE
    ├── pyproject.toml
    ├── setup.py
    ├── .readthedocs.yml
    ├── docs/
    │   ├── README.md
    │   ├── conda_env.yml
    │   ├── conf.py
    │   ├── index.rst
    │   ├── make.bat
    │   ├── Makefile
    │   ├── spelling_wordlist.txt
    │   ├── _static/
    │   │   ├── css/
    │   │   │   └── baselines_theme.css
    │   │   └── img/
    │   ├── common/
    │   │   ├── atari_wrappers.rst
    │   │   ├── distributions.rst
    │   │   ├── env_checker.rst
    │   │   ├── env_util.rst
    │   │   ├── envs.rst
    │   │   ├── evaluation.rst
    │   │   ├── logger.rst
    │   │   ├── monitor.rst
    │   │   ├── noise.rst
    │   │   └── utils.rst
    │   ├── guide/
    │   │   ├── algos.rst
    │   │   ├── callbacks.rst
    │   │   ├── checking_nan.rst
    │   │   ├── custom_env.rst
    │   │   ├── custom_policy.rst
    │   │   ├── developer.rst
    │   │   ├── examples.rst
    │   │   ├── export.rst
    │   │   ├── imitation.rst
    │   │   ├── install.rst
    │   │   ├── integrations.rst
    │   │   ├── migration.rst
    │   │   ├── quickstart.rst
    │   │   ├── rl.rst
    │   │   ├── rl_tips.rst
    │   │   ├── rl_zoo.rst
    │   │   ├── save_format.rst
    │   │   ├── sb3_contrib.rst
    │   │   ├── sbx.rst
    │   │   ├── tensorboard.rst
    │   │   └── vec_envs.rst
    │   ├── misc/
    │   │   ├── changelog.rst
    │   │   └── projects.rst
    │   └── modules/
    │       ├── a2c.rst
    │       ├── base.rst
    │       ├── ddpg.rst
    │       ├── dqn.rst
    │       ├── her.rst
    │       ├── ppo.rst
    │       ├── sac.rst
    │       └── td3.rst
    ├── scripts/
    │   ├── build_docker.sh
    │   ├── run_docker_cpu.sh
    │   ├── run_docker_gpu.sh
    │   └── run_tests.sh
    ├── stable_baselines3/
    │   ├── __init__.py
    │   ├── py.typed
    │   ├── version.txt
    │   ├── a2c/
    │   │   ├── __init__.py
    │   │   ├── a2c.py
    │   │   └── policies.py
    │   ├── common/
    │   │   ├── __init__.py
    │   │   ├── atari_wrappers.py
    │   │   ├── base_class.py
    │   │   ├── buffers.py
    │   │   ├── callbacks.py
    │   │   ├── distributions.py
    │   │   ├── env_checker.py
    │   │   ├── env_util.py
    │   │   ├── evaluation.py
    │   │   ├── logger.py
    │   │   ├── monitor.py
    │   │   ├── noise.py
    │   │   ├── off_policy_algorithm.py
    │   │   ├── on_policy_algorithm.py
    │   │   ├── policies.py
    │   │   ├── preprocessing.py
    │   │   ├── results_plotter.py
    │   │   ├── running_mean_std.py
    │   │   ├── save_util.py
    │   │   ├── torch_layers.py
    │   │   ├── type_aliases.py
    │   │   ├── utils.py
    │   │   ├── envs/
    │   │   │   ├── __init__.py
    │   │   │   ├── bit_flipping_env.py
    │   │   │   ├── identity_env.py
    │   │   │   └── multi_input_envs.py
    │   │   ├── sb2_compat/
    │   │   │   ├── __init__.py
    │   │   │   └── rmsprop_tf_like.py
    │   │   └── vec_env/
    │   │       ├── __init__.py
    │   │       ├── base_vec_env.py
    │   │       ├── dummy_vec_env.py
    │   │       ├── patch_gym.py
    │   │       ├── stacked_observations.py
    │   │       ├── subproc_vec_env.py
    │   │       ├── util.py
    │   │       ├── vec_check_nan.py
    │   │       ├── vec_extract_dict_obs.py
    │   │       ├── vec_frame_stack.py
    │   │       ├── vec_monitor.py
    │   │       ├── vec_normalize.py
    │   │       ├── vec_transpose.py
    │   │       └── vec_video_recorder.py
    │   ├── ddpg/
    │   │   ├── __init__.py
    │   │   ├── ddpg.py
    │   │   └── policies.py
    │   ├── dqn/
    │   │   ├── __init__.py
    │   │   ├── dqn.py
    │   │   └── policies.py
    │   ├── her/
    │   │   ├── __init__.py
    │   │   ├── goal_selection_strategy.py
    │   │   └── her_replay_buffer.py
    │   ├── ppo/
    │   │   ├── __init__.py
    │   │   ├── policies.py
    │   │   └── ppo.py
    │   ├── sac/
    │   │   ├── __init__.py
    │   │   ├── policies.py
    │   │   └── sac.py
    │   └── td3/
    │       ├── __init__.py
    │       ├── policies.py
    │       └── td3.py
    ├── tests/
    │   ├── __init__.py
    │   ├── test_buffers.py
    │   ├── test_callbacks.py
    │   ├── test_cnn.py
    │   ├── test_custom_policy.py
    │   ├── test_deterministic.py
    │   ├── test_dict_env.py
    │   ├── test_distributions.py
    │   ├── test_env_checker.py
    │   ├── test_envs.py
    │   ├── test_gae.py
    │   ├── test_her.py
    │   ├── test_identity.py
    │   ├── test_logger.py
    │   ├── test_monitor.py
    │   ├── test_predict.py
    │   ├── test_preprocessing.py
    │   ├── test_run.py
    │   ├── test_save_load.py
    │   ├── test_sde.py
    │   ├── test_spaces.py
    │   ├── test_tensorboard.py
    │   ├── test_train_eval_mode.py
    │   ├── test_utils.py
    │   ├── test_vec_check_nan.py
    │   ├── test_vec_envs.py
    │   ├── test_vec_extract_dict_obs.py
    │   ├── test_vec_monitor.py
    │   ├── test_vec_normalize.py
    │   └── test_vec_stacked_obs.py
    ├── .dockerignore -> .gitignore
    └── .github/
        ├── PULL_REQUEST_TEMPLATE.md
        ├── ISSUE_TEMPLATE/
        │   ├── bug_report.yml
        │   ├── custom_env.yml
        │   ├── documentation.yml
        │   ├── feature_request.yml
        │   └── question.yml
        └── workflows/
            └── ci.yml


Files Content:

(Files content cropped to 300k characters, download full ingest to see more)
================================================
FILE: README.md
================================================
<!-- [![pipeline status](https://gitlab.com/araffin/stable-baselines3/badges/master/pipeline.svg)](https://gitlab.com/araffin/stable-baselines3/-/commits/master) -->
[![CI](https://github.com/DLR-RM/stable-baselines3/workflows/CI/badge.svg)](https://github.com/DLR-RM/stable-baselines3/actions/workflows/ci.yml)
[![Documentation Status](https://readthedocs.org/projects/stable-baselines/badge/?version=master)](https://stable-baselines3.readthedocs.io/en/master/?badge=master) [![coverage report](https://gitlab.com/araffin/stable-baselines3/badges/master/coverage.svg)](https://github.com/DLR-RM/stable-baselines3/actions/workflows/ci.yml)
[![codestyle](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)


# Stable Baselines3

<img src="docs/\_static/img/logo.png" align="right" width="40%"/>

Stable Baselines3 (SB3) is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).

You can read a detailed presentation of Stable Baselines3 in the [v1.0 blog post](https://araffin.github.io/post/sb3/) or our [JMLR paper](https://jmlr.org/papers/volume22/20-1364/20-1364.pdf).


These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.

**Note: Despite its simplicity of use, Stable Baselines3 (SB3) assumes you have some knowledge about Reinforcement Learning (RL).** You should not utilize this library without some practice. To that extent, we provide good resources in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/rl.html) to get started with RL.

## Main Features

**The performance of each algorithm was tested** (see *Results* section in their respective page),
you can take a look at the issues [#48](https://github.com/DLR-RM/stable-baselines3/issues/48) and [#49](https://github.com/DLR-RM/stable-baselines3/issues/49) for more details.

We also provide detailed logs and reports on the [OpenRL Benchmark](https://wandb.ai/openrlbenchmark/sb3) platform.


| **Features**                | **Stable-Baselines3** |
| --------------------------- | ----------------------|
| State of the art RL methods | :heavy_check_mark: |
| Documentation               | :heavy_check_mark: |
| Custom environments         | :heavy_check_mark: |
| Custom policies             | :heavy_check_mark: |
| Common interface            | :heavy_check_mark: |
| `Dict` observation space support  | :heavy_check_mark: |
| Ipython / Notebook friendly | :heavy_check_mark: |
| Tensorboard support         | :heavy_check_mark: |
| PEP8 code style             | :heavy_check_mark: |
| Custom callback             | :heavy_check_mark: |
| High code coverage          | :heavy_check_mark: |
| Type hints                  | :heavy_check_mark: |


### Planned features

Since most of the features from the [original roadmap](https://github.com/DLR-RM/stable-baselines3/issues/1) have been implemented, there are no major changes planned for SB3, it is now *stable*.
If you want to contribute, you can search in the issues for the ones where [help is welcomed](https://github.com/DLR-RM/stable-baselines3/labels/help%20wanted) and the other [proposed enhancements](https://github.com/DLR-RM/stable-baselines3/labels/enhancement).

While SB3 development is now focused on bug fixes and maintenance (doc update, user experience, ...), there is more active development going on in the associated repositories:
- newer algorithms are regularly added to the [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) repository
- faster variants are developed in the [SBX (SB3 + Jax)](https://github.com/araffin/sbx) repository
- the training framework for SB3, the RL Zoo, has an active [roadmap](https://github.com/DLR-RM/rl-baselines3-zoo/issues/299)

## Migration guide: from Stable-Baselines (SB2) to Stable-Baselines3 (SB3)

A migration guide from SB2 to SB3 can be found in the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/migration.html).

## Documentation

Documentation is available online: [https://stable-baselines3.readthedocs.io/](https://stable-baselines3.readthedocs.io/)

## Integrations

Stable-Baselines3 has some integration with other libraries/services like Weights & Biases for experiment tracking or Hugging Face for storing/sharing trained models. You can find out more in the [dedicated section](https://stable-baselines3.readthedocs.io/en/master/guide/integrations.html) of the documentation.


## RL Baselines3 Zoo: A Training Framework for Stable Baselines3 Reinforcement Learning Agents

[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo) is a training framework for Reinforcement Learning (RL).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

Github repo: https://github.com/DLR-RM/rl-baselines3-zoo

Documentation: https://rl-baselines3-zoo.readthedocs.io/en/master/

## SB3-Contrib: Experimental RL Features

We implement experimental features in a separate contrib repository: [SB3-Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib)

This allows SB3 to maintain a stable and compact core, while still providing the latest features, like Recurrent PPO (PPO LSTM), CrossQ, Truncated Quantile Critics (TQC), Quantile Regression DQN (QR-DQN) or PPO with invalid action masking (Maskable PPO).

Documentation is available online: [https://sb3-contrib.readthedocs.io/](https://sb3-contrib.readthedocs.io/)

## Stable-Baselines Jax (SBX)

[Stable Baselines Jax (SBX)](https://github.com/araffin/sbx) is a proof of concept version of Stable-Baselines3 in Jax, with recent algorithms like DroQ or CrossQ.

It provides a minimal number of features compared to SB3 but can be much faster (up to 20x times!): https://twitter.com/araffin2/status/1590714558628253698


## Installation

**Note:** Stable-Baselines3 supports PyTorch >= 2.3

### Prerequisites
Stable Baselines3 requires Python 3.9+.

#### Windows

To install stable-baselines on Windows, please look at the [documentation](https://stable-baselines3.readthedocs.io/en/master/guide/install.html#prerequisites).


### Install using pip
Install the Stable Baselines3 package:
```sh
pip install 'stable-baselines3[extra]'
```

This includes an optional dependencies like Tensorboard, OpenCV or `ale-py` to train on atari games. If you do not need those, you can use:
```sh
pip install stable-baselines3
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more details and alternatives (from source, using docker).


## Example

Most of the code in the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run PPO on a cartpole environment:
```python
import gymnasium as gym

from stable_baselines3 import PPO

env = gym.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = env.reset()

env.close()
```

Or just train a model with a one liner if [the environment is registered in Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)
```

Please read the [documentation](https://stable-baselines3.readthedocs.io/) for more examples.


## Try it online with Colab Notebooks !

All the following examples can be executed online using Google Colab notebooks:

- [Full Tutorial](https://github.com/araffin/rl-tutorial-jnrr19)
- [All Notebooks](https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3)
- [Getting Started](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb)
- [Training, Saving, Loading](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb)
- [Multiprocessing](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb)
- [Monitor Training and Plotting](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb)
- [Atari Games](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb)
- [RL Baselines Zoo](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb)
- [PyBullet](https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb)


## Implemented Algorithms

| **Name**         | **Recurrent**      | `Box`          | `Discrete`     | `MultiDiscrete` | `MultiBinary`  | **Multi Processing**              |
| ------------------- | ------------------ | ------------------ | ------------------ | ------------------- | ------------------ | --------------------------------- |
| ARS<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: |
| A2C   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: |
| CrossQ<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DDPG  | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| DQN   | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| HER   | :x: | :heavy_check_mark: | :heavy_check_mark: | :x: | :x: | :heavy_check_mark: |
| PPO   | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| QR-DQN<sup>[1](#f1)</sup>  | :x: | :x: | :heavy_check_mark: | :x:                 | :x:                | :heavy_check_mark: |
| RecurrentPPO<sup>[1](#f1)</sup>   | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| SAC   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TD3   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x:                | :heavy_check_mark: |
| TQC<sup>[1](#f1)</sup>   | :x: | :heavy_check_mark: | :x:                | :x:                 | :x: | :heavy_check_mark: |
| TRPO<sup>[1](#f1)</sup>  | :x: | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark: |
| Maskable PPO<sup>[1](#f1)</sup>   | :x: | :x: | :heavy_check_mark: | :heavy_check_mark:  | :heavy_check_mark: | :heavy_check_mark:  |

<b id="f1">1</b>: Implemented in [SB3 Contrib](https://github.com/Stable-Baselines-Team/stable-baselines3-contrib) GitHub repository.

Actions `gymnasium.spaces`:
 * `Box`: A N-dimensional box that contains every point in the action space.
 * `Discrete`: A list of possible actions, where each timestep only one of the actions can be used.
 * `MultiDiscrete`: A list of possible actions, where each timestep only one action of each discrete set can be used.
 * `MultiBinary`: A list of possible actions, where each timestep any of the actions can be used in any combination.



## Testing the installation
### Install dependencies
```sh
pip install -e .[docs,tests,extra]
```
### Run tests
All unit tests in stable baselines3 can be run using `pytest` runner:
```sh
make pytest
```
To run a single test file:
```sh
python3 -m pytest -v tests/test_env_checker.py
```
To run a single test:
```sh
python3 -m pytest -v -k 'test_check_env_dict_action'
```

You can also do a static type check using `mypy`:
```sh
pip install mypy
make type
```

Codestyle check with `ruff`:
```sh
pip install ruff
make lint
```

## Projects Using Stable-Baselines3

We try to maintain a list of projects using stable-baselines3 in the [documentation](https://stable-baselines3.readthedocs.io/en/master/misc/projects.html),
please tell us if you want your project to appear on this page ;)

## Citing the Project

To cite this repository in publications:

```bibtex
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}
```

Note: If you need to refer to a specific version of SB3, you can also use the [Zenodo DOI](https://doi.org/10.5281/zenodo.8123988).

## Maintainers

Stable-Baselines3 is currently maintained by [Ashley Hill](https://github.com/hill-a) (aka @hill-a), [Antonin Raffin](https://araffin.github.io/) (aka [@araffin](https://github.com/araffin)), [Maximilian Ernestus](https://github.com/ernestum) (aka @ernestum), [Adam Gleave](https://github.com/adamgleave) (@AdamGleave), [Anssi Kanervisto](https://github.com/Miffyli) (@Miffyli) and [Quentin Gallouédec](https://gallouedec.com/) (@qgallouedec).

**Important Note: We do not provide technical support, or consulting** and do not answer personal questions via email.
Please post your question on the [RL Discord](https://discord.com/invite/xhfNqQv), [Reddit](https://www.reddit.com/r/reinforcementlearning/), or [Stack Overflow](https://stackoverflow.com/) in that case.


## How To Contribute

To any interested in making the baselines better, there is still some documentation that needs to be done.
If you want to contribute, please read [**CONTRIBUTING.md**](./CONTRIBUTING.md) guide first.

## Acknowledgments

The initial work to develop Stable Baselines3 was partially funded by the project *Reduced Complexity Models* from the *Helmholtz-Gemeinschaft Deutscher Forschungszentren*, and by the EU's Horizon 2020 Research and Innovation Programme under grant number 951992 ([VeriDream](https://www.veridream.eu/)).

The original version, Stable Baselines, was created in the [robotics lab U2IS](http://u2is.ensta-paristech.fr/index.php?lang=en) ([INRIA Flowers](https://flowers.inria.fr/) team) at [ENSTA ParisTech](http://www.ensta-paristech.fr/en).


Logo credits: [L.M. Tenkes](https://www.instagram.com/lucillehue/)



================================================
FILE: CITATION.bib
================================================
@article{stable-baselines3,
  author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
  title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
  journal = {Journal of Machine Learning Research},
  year    = {2021},
  volume  = {22},
  number  = {268},
  pages   = {1-8},
  url     = {http://jmlr.org/papers/v22/20-1364.html}
}



================================================
FILE: CODE_OF_CONDUCT.md
================================================
# Contributor Covenant Code of Conduct

## Our Pledge

We as members, contributors, and leaders pledge to make participation in our
community a harassment-free experience for everyone, regardless of age, body
size, visible or invisible disability, ethnicity, sex characteristics, gender
identity and expression, level of experience, education, socioeconomic status,
nationality, personal appearance, race, religion, or sexual identity
and orientation.

We pledge to act and interact in ways that contribute to an open, welcoming,
diverse, inclusive, and healthy community.

## Our Standards

Examples of behavior that contributes to a positive environment for our
community include:

* Demonstrating empathy and kindness toward other people
* Being respectful of differing opinions, viewpoints, and experiences
* Giving and gracefully accepting constructive feedback
* Accepting responsibility and apologizing to those affected by our mistakes,
  and learning from the experience
* Focusing on what is best not just for us as individuals, but for the
  overall community

Examples of unacceptable behavior include:

* The use of sexualized language or imagery, and sexual attention or
  advances of any kind
* Trolling, insulting or derogatory comments, and personal or political attacks
* Public or private harassment
* Publishing others' private information, such as a physical or email
  address, without their explicit permission
* Other conduct which could reasonably be considered inappropriate in a
  professional setting

## Enforcement Responsibilities

Community leaders are responsible for clarifying and enforcing our standards of
acceptable behavior and will take appropriate and fair corrective action in
response to any behavior that they deem inappropriate, threatening, offensive,
or harmful.

Community leaders have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions that are
not aligned to this Code of Conduct, and will communicate reasons for moderation
decisions when appropriate.

## Scope

This Code of Conduct applies within all community spaces, and also applies when
an individual is officially representing the community in public spaces.
Examples of representing our community include using an official e-mail address,
posting via an official social media account, or acting as an appointed
representative at an online or offline event.

## Enforcement

Instances of abusive, harassing, or otherwise unacceptable behavior may be
reported to the community leaders responsible for enforcement at
antonin [dot] raffin [at] dlr [dot] de.
All complaints will be reviewed and investigated promptly and fairly.

All community leaders are obligated to respect the privacy and security of the
reporter of any incident.

## Enforcement Guidelines

Community leaders will follow these Community Impact Guidelines in determining
the consequences for any action they deem in violation of this Code of Conduct:

### 1. Correction

**Community Impact**: Use of inappropriate language or other behavior deemed
unprofessional or unwelcome in the community.

**Consequence**: A private, written warning from community leaders, providing
clarity around the nature of the violation and an explanation of why the
behavior was inappropriate. A public apology may be requested.

### 2. Warning

**Community Impact**: A violation through a single incident or series
of actions.

**Consequence**: A warning with consequences for continued behavior. No
interaction with the people involved, including unsolicited interaction with
those enforcing the Code of Conduct, for a specified period of time. This
includes avoiding interactions in community spaces as well as external channels
like social media. Violating these terms may lead to a temporary or
permanent ban.

### 3. Temporary Ban

**Community Impact**: A serious violation of community standards, including
sustained inappropriate behavior.

**Consequence**: A temporary ban from any sort of interaction or public
communication with the community for a specified period of time. No public or
private interaction with the people involved, including unsolicited interaction
with those enforcing the Code of Conduct, is allowed during this period.
Violating these terms may lead to a permanent ban.

### 4. Permanent Ban

**Community Impact**: Demonstrating a pattern of violation of community
standards, including sustained inappropriate behavior,  harassment of an
individual, or aggression toward or disparagement of classes of individuals.

**Consequence**: A permanent ban from any sort of public interaction within
the community.

## Attribution

This Code of Conduct is adapted from the [Contributor Covenant][homepage],
version 2.0, available at
https://www.contributor-covenant.org/version/2/0/code_of_conduct.html.

Community Impact Guidelines were inspired by [Mozilla's code of conduct
enforcement ladder](https://github.com/mozilla/diversity).

[homepage]: https://www.contributor-covenant.org

For answers to common questions about this code of conduct, see the FAQ at
https://www.contributor-covenant.org/faq. Translations are available at
https://www.contributor-covenant.org/translations.



================================================
FILE: CONTRIBUTING.md
================================================
## Contributing to Stable-Baselines3

If you are interested in contributing to Stable-Baselines, your contributions will fall
into two categories:
1. You want to propose a new Feature and implement it
    - Create an issue about your intended feature, and we shall discuss the design and
    implementation. Once we agree that the plan looks good, go ahead and implement it.
2. You want to implement a feature or bug-fix for an outstanding issue
    - Look at the outstanding issues here: https://github.com/DLR-RM/stable-baselines3/labels/help%20wanted
    - Pick an issue or feature and comment on the task that you want to work on this feature.
    - If you need more context on a particular issue, please ask, and we shall provide.

Once you finish implementing a feature or bug-fix, please send a Pull Request to
https://github.com/DLR-RM/stable-baselines3


If you are not familiar with creating a Pull Request, here are some guides:
- http://stackoverflow.com/questions/14680711/how-to-do-a-github-pull-request
- https://help.github.com/articles/creating-a-pull-request/


## Developing Stable-Baselines3

To develop Stable-Baselines3 on your machine, here are some tips:

1. Clone a copy of Stable-Baselines3 from source:

```bash
git clone https://github.com/DLR-RM/stable-baselines3
cd stable-baselines3/
```

2. Install Stable-Baselines3 in develop mode, with support for building the docs and running tests:

```bash
pip install -e .[docs,tests,extra]
```

## Codestyle

We use [black codestyle](https://github.com/psf/black) (max line length of 127 characters) together with [ruff](https://github.com/astral-sh/ruff) (isort rules) to sort the imports.
For the documentation, we use the default line length of 88 characters per line.

**Please run `make format`** to reformat your code. You can check the codestyle using `make check-codestyle` and `make lint`.

Please document each function/method and [type](https://google.github.io/pytype/user_guide.html) them using the following template:

```python

def my_function(arg1: type1, arg2: type2) -> returntype:
    """
    Short description of the function.

    :param arg1: describe what is arg1
    :param arg2: describe what is arg2
    :return: describe what is returned
    """
    ...
    return my_variable
```

## Pull Request (PR)

Before proposing a PR, please open an issue, where the feature will be discussed. This prevents from duplicated PR to be proposed and also ease the code review process.

Each PR need to be reviewed and accepted by at least one of the maintainers (@hill-a, @araffin, @ernestum, @AdamGleave, @Miffyli or @qgallouedec).
A PR must pass the Continuous Integration tests to be merged with the master branch.


## Tests

All new features must add tests in the `tests/` folder ensuring that everything works fine.
We use [pytest](https://pytest.org/).
Also, when a bug fix is proposed, tests should be added to avoid regression.

To run tests with `pytest`:

```
make pytest
```

Type checking with `mypy`:

```
make type
```

Codestyle check with `black`, and `ruff` (`isort` rules):

```
make check-codestyle
make lint
```

To run `type`, `format` and `lint` in one command:
```
make commit-checks
```

Build the documentation:

```
make doc
```

Check documentation spelling (you need to install `sphinxcontrib.spelling` package for that):

```
make spelling
```


## Changelog and Documentation

Please do not forget to update the changelog (`docs/misc/changelog.rst`) and add documentation if needed.
You should add your username next to each changelog entry that you added. If this is your first contribution, please add your username at the bottom too.
A README is present in the `docs/` folder for instructions on how to build the documentation.


Credits: this contributing guide is based on the [PyTorch](https://github.com/pytorch/pytorch/) one.



================================================
FILE: Dockerfile
================================================
ARG PARENT_IMAGE=mambaorg/micromamba:2.0-ubuntu24.04
FROM $PARENT_IMAGE
ARG PYTORCH_DEPS=https://download.pytorch.org/whl/cpu
ARG PYTHON_VERSION=3.12
ARG MAMBA_DOCKERFILE_ACTIVATE=1  # (otherwise python will not be found)

# Install micromamba env and dependencies
RUN micromamba install -n base -y python=$PYTHON_VERSION && \
    micromamba clean --all --yes

ENV CODE_DIR=/home/$MAMBA_USER

# Copy setup file only to install dependencies
COPY --chown=$MAMBA_USER:$MAMBA_USER ./setup.py ${CODE_DIR}/stable-baselines3/setup.py
COPY --chown=$MAMBA_USER:$MAMBA_USER ./stable_baselines3/version.txt ${CODE_DIR}/stable-baselines3/stable_baselines3/version.txt

RUN cd ${CODE_DIR}/stable-baselines3 && \
    pip install uv && \
    uv pip install --system torch --default-index ${PYTORCH_DEPS} && \
    uv pip install --system -e .[extra,tests,docs] && \
    # Use headless version for docker
    uv pip uninstall opencv-python && \
    uv pip install --system opencv-python-headless && \
    pip cache purge && \
    uv cache clean

CMD /bin/bash



================================================
FILE: LICENSE
================================================
The MIT License

Copyright (c) 2019 Antonin Raffin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.



================================================
FILE: Makefile
================================================
SHELL=/bin/bash
LINT_PATHS=stable_baselines3/ tests/ docs/conf.py setup.py

pytest:
	./scripts/run_tests.sh

mypy:
	mypy ${LINT_PATHS}

missing-annotations:
	mypy --disallow-untyped-calls --disallow-untyped-defs --ignore-missing-imports stable_baselines3

# missing docstrings
# pylint -d R,C,W,E -e C0116 stable_baselines3 -j 4

type: mypy

lint:
	# stop the build if there are Python syntax errors or undefined names
	# see https://www.flake8rules.com/
	ruff check ${LINT_PATHS} --select=E9,F63,F7,F82 --output-format=full
	# exit-zero treats all errors as warnings.
	ruff check ${LINT_PATHS} --exit-zero --output-format=concise

format:
	# Sort imports
	ruff check --select I ${LINT_PATHS} --fix
	# Reformat using black
	black ${LINT_PATHS}

check-codestyle:
	# Sort imports
	ruff check --select I ${LINT_PATHS}
	# Reformat using black
	black --check ${LINT_PATHS}

commit-checks: format type lint

doc:
	cd docs && make html

spelling:
	cd docs && make spelling

clean:
	cd docs && make clean

# Build docker images
# If you do export RELEASE=True, it will also push them
docker: docker-cpu docker-gpu

docker-cpu:
	./scripts/build_docker.sh

docker-gpu:
	USE_GPU=True ./scripts/build_docker.sh

# PyPi package release
release:
	python -m build
	twine upload dist/*

# Test PyPi package release
test-release:
	python -m build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: clean spelling doc lint format check-codestyle commit-checks



================================================
FILE: NOTICE
================================================
Large portion of the code of Stable-Baselines3 (in `common/`) were ported from Stable-Baselines, a fork of OpenAI Baselines,
both licensed under the MIT License:

before the fork (June 2018):
Copyright (c) 2017 OpenAI (http://openai.com)

after the fork (June 2018):
Copyright (c) 2018-2019 Stable-Baselines Team


Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.



================================================
FILE: pyproject.toml
================================================
[tool.ruff]
# Same as Black.
line-length = 127
# Assume Python 3.9
target-version = "py39"

[tool.ruff.lint]
# See https://beta.ruff.rs/docs/rules/
select = ["E", "F", "B", "UP", "C90", "RUF"]
# B028: Ignore explicit stacklevel`
# RUF013: Too many false positives (implicit optional)
ignore = ["B028", "RUF013"]

[tool.ruff.lint.per-file-ignores]
# Default implementation in abstract methods
"./stable_baselines3/common/callbacks.py" = ["B027"]
"./stable_baselines3/common/noise.py" = ["B027"]
# ClassVar, implicit optional check not needed for tests
"./tests/*.py" = ["RUF012", "RUF013"]

[tool.ruff.lint.mccabe]
# Unlike Flake8, default to a complexity level of 10.
max-complexity = 15

[tool.black]
line-length = 127

[tool.mypy]
ignore_missing_imports = true
follow_imports = "silent"
show_error_codes = true
exclude = """(?x)(
    tests/test_logger.py$
    | tests/test_train_eval_mode.py$
  )"""

[tool.pytest.ini_options]
# Deterministic ordering for tests; useful for pytest-xdist.
env = ["PYTHONHASHSEED=0"]

filterwarnings = [
    # A2C/PPO on GPU
    "ignore:You are trying to run (PPO|A2C) on the GPU",
    # Tensorboard warnings
    "ignore::DeprecationWarning:tensorboard",
    # Gymnasium warnings
    "ignore::UserWarning:gymnasium",
    # tqdm warning about rich being experimental
    "ignore:rich is experimental",
]
markers = [
    "expensive: marks tests as expensive (deselect with '-m \"not expensive\"')",
]

[tool.coverage.run]
disable_warnings = ["couldnt-parse"]
branch = false
omit = [
    "tests/*",
    "setup.py",
    # Require graphical interface
    "stable_baselines3/common/results_plotter.py",
    # Require ffmpeg
    "stable_baselines3/common/vec_env/vec_video_recorder.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "raise NotImplementedError()",
    "if typing.TYPE_CHECKING:",
]



================================================
FILE: setup.py
================================================
import os

from setuptools import find_packages, setup

with open(os.path.join("stable_baselines3", "version.txt")) as file_handler:
    __version__ = file_handler.read().strip()


long_description = """

# Stable Baselines3

Stable Baselines3 is a set of reliable implementations of reinforcement learning algorithms in PyTorch. It is the next major version of [Stable Baselines](https://github.com/hill-a/stable-baselines).

These algorithms will make it easier for the research community and industry to replicate, refine, and identify new ideas, and will create good baselines to build projects on top of. We expect these tools will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. We also hope that the simplicity of these tools will allow beginners to experiment with a more advanced toolset, without being buried in implementation details.


## Links

Repository:
https://github.com/DLR-RM/stable-baselines3

Blog post:
https://araffin.github.io/post/sb3/

Documentation:
https://stable-baselines3.readthedocs.io/en/master/

RL Baselines3 Zoo:
https://github.com/DLR-RM/rl-baselines3-zoo

SB3 Contrib:
https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

## Quick example

Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms using Gym.

Here is a quick example of how to train and run PPO on a cartpole environment:

```python
import gymnasium

from stable_baselines3 import PPO

env = gymnasium.make("CartPole-v1", render_mode="human")

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()

```

Or just train a model with a one liner if [the environment is registered in Gymnasium](https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/) and if [the policy is registered](https://stable-baselines3.readthedocs.io/en/master/guide/custom_policy.html):

```python
from stable_baselines3 import PPO

model = PPO("MlpPolicy", "CartPole-v1").learn(10_000)
```

"""  # noqa:E501


setup(
    name="stable_baselines3",
    packages=[package for package in find_packages() if package.startswith("stable_baselines3")],
    package_data={"stable_baselines3": ["py.typed", "version.txt"]},
    install_requires=[
        "gymnasium>=0.29.1,<1.2.0",
        "numpy>=1.20,<3.0",
        "torch>=2.3,<3.0",
        # For saving models
        "cloudpickle",
        # For reading logs
        "pandas",
        # Plotting learning curves
        "matplotlib",
    ],
    extras_require={
        "tests": [
            # Run tests and coverage
            "pytest",
            "pytest-cov",
            "pytest-env",
            "pytest-xdist",
            # Type check
            "mypy",
            # Lint code and sort imports (flake8 and isort replacement)
            "ruff>=0.3.1",
            # Reformat
            "black>=25.1.0,<26",
        ],
        "docs": [
            "sphinx>=5,<9",
            "sphinx-autobuild",
            "sphinx-rtd-theme>=1.3.0",
            # For spelling
            "sphinxcontrib.spelling",
            # Copy button for code snippets
            "sphinx_copybutton",
        ],
        "extra": [
            # For render
            "opencv-python",
            "pygame",
            # Tensorboard support
            "tensorboard>=2.9.1",
            # Checking memory taken by replay buffer
            "psutil",
            # For progress bar callback
            "tqdm",
            "rich",
            # For atari games,
            "ale-py>=0.9.0",
            "pillow",
        ],
    },
    description="Pytorch version of Stable Baselines, implementations of reinforcement learning algorithms.",
    author="Antonin Raffin",
    url="https://github.com/DLR-RM/stable-baselines3",
    author_email="antonin.raffin@dlr.de",
    keywords="reinforcement-learning-algorithms reinforcement-learning machine-learning "
    "gymnasium gym openai stable baselines toolbox python data-science",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=__version__,
    python_requires=">=3.9",
    # PyPI package information.
    project_urls={
        "Code": "https://github.com/DLR-RM/stable-baselines3",
        "Documentation": "https://stable-baselines3.readthedocs.io/",
        "Changelog": "https://stable-baselines3.readthedocs.io/en/master/misc/changelog.html",
        "SB3-Contrib": "https://github.com/Stable-Baselines-Team/stable-baselines3-contrib",
        "RL-Zoo": "https://github.com/DLR-RM/rl-baselines3-zoo",
        "SBX": "https://github.com/araffin/sbx",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)

# python setup.py sdist
# python setup.py bdist_wheel
# twine upload --repository-url https://test.pypi.org/legacy/ dist/*
# twine upload dist/*



================================================
FILE: .readthedocs.yml
================================================
# Read the Docs configuration file
# See https://docs.readthedocs.io/en/stable/config-file/v2.html for details

# Required
version: 2

# Build documentation in the docs/ directory with Sphinx
sphinx:
  configuration: docs/conf.py

# Optionally build your docs in additional formats such as PDF and ePub
formats: all

# Set requirements using conda env
conda:
  environment: docs/conda_env.yml

build:
  os: ubuntu-24.04
  tools:
    python: "mambaforge-23.11"



================================================
FILE: docs/README.md
================================================
## Stable Baselines3 Documentation

This folder contains documentation for the RL baselines.


### Build the Documentation

#### Install Sphinx and Theme
Execute this command in the project root:
```
pip install -e ".[docs]"
```

#### Building the Docs

In the `docs/` folder:
```
make html
```

if you want to building each time a file is changed:

```
sphinx-autobuild . _build/html
```



================================================
FILE: docs/conda_env.yml
================================================
name: root
channels:
  - pytorch
  - conda-forge
dependencies:
  - cpuonly=1.0=0
  - pip=24.2
  - python=3.11
  - pytorch=2.5.0=py3.11_cpu_0
  - pip:
    - gymnasium>=0.29.1,<1.1.0
    - cloudpickle
    - opencv-python-headless
    - pandas
    - numpy>=1.20,<3.0
    - matplotlib
    - sphinx>=5,<9
    - sphinx_rtd_theme>=1.3.0
    - sphinx_copybutton



================================================
FILE: docs/conf.py
================================================
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import datetime
import os
import sys

# We CANNOT enable 'sphinxcontrib.spelling' because ReadTheDocs.org does not support
# PyEnchant.
try:
    import sphinxcontrib.spelling  # noqa: F401

    enable_spell_check = True
except ImportError:
    enable_spell_check = False

# Try to enable copy button
try:
    import sphinx_copybutton  # noqa: F401

    enable_copy_button = True
except ImportError:
    enable_copy_button = False

# source code directory, relative to this file, for sphinx-autobuild
sys.path.insert(0, os.path.abspath(".."))

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "../stable_baselines3", "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()

# -- Project information -----------------------------------------------------

project = "Stable Baselines3"
copyright = f"2021-{datetime.date.today().year}, Stable Baselines3"
author = "Stable Baselines3 Contributors"

# The short X.Y version
version = "master (" + __version__ + " )"
# The full version, including alpha/beta/rc tags
release = __version__


# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.mathjax",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    # 'sphinx.ext.intersphinx',
    # 'sphinx.ext.doctest'
]

autodoc_typehints = "description"

if enable_spell_check:
    extensions.append("sphinxcontrib.spelling")

if enable_copy_button:
    extensions.append("sphinx_copybutton")

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.

html_theme = "sphinx_rtd_theme"
html_logo = "_static/img/logo.png"


def setup(app):
    app.add_css_file("css/baselines_theme.css")


# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "StableBaselines3doc"


# -- Options for LaTeX output ------------------------------------------------

latex_elements: dict[str, str] = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "StableBaselines3.tex", "Stable Baselines3 Documentation", "Stable Baselines3 Contributors", "manual"),
]


# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "stablebaselines3", "Stable Baselines3 Documentation", [author], 1)]


# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "StableBaselines3",
        "Stable Baselines3 Documentation",
        author,
        "StableBaselines3",
        "One line description of project.",
        "Miscellaneous",
    ),
]


# -- Extension configuration -------------------------------------------------

# Example configuration for intersphinx: refer to the Python standard library.
# intersphinx_mapping = {
#     'python': ('https://docs.python.org/3/', None),
#     'numpy': ('http://docs.scipy.org/doc/numpy/', None),
#     'torch': ('http://pytorch.org/docs/master/', None),
# }



================================================
FILE: docs/index.rst
================================================
.. Stable Baselines3 documentation master file, created by
   sphinx-quickstart on Thu Sep 26 11:06:54 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Stable-Baselines3 Docs - Reliable Reinforcement Learning Implementations
========================================================================

`Stable Baselines3 (SB3) <https://github.com/DLR-RM/stable-baselines3>`_ is a set of reliable implementations of reinforcement learning algorithms in PyTorch.
It is the next major version of `Stable Baselines <https://github.com/hill-a/stable-baselines>`_.


Github repository: https://github.com/DLR-RM/stable-baselines3

Paper: https://jmlr.org/papers/volume22/20-1364/20-1364.pdf

RL Baselines3 Zoo (training framework for SB3): https://github.com/DLR-RM/rl-baselines3-zoo

RL Baselines3 Zoo provides a collection of pre-trained agents, scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

SB3 Contrib (experimental RL code, latest algorithms): https://github.com/Stable-Baselines-Team/stable-baselines3-contrib

SBX (SB3 + Jax): https://github.com/araffin/sbx


Main Features
--------------

- Unified structure for all algorithms
- PEP8 compliant (unified code style)
- Documented functions and classes
- Tests, high code coverage and type hints
- Clean code
- Tensorboard support
- **The performance of each algorithm was tested** (see *Results* section in their respective page)


.. toctree::
   :maxdepth: 2
   :caption: User Guide

   guide/install
   guide/quickstart
   guide/rl_tips
   guide/rl
   guide/algos
   guide/examples
   guide/vec_envs
   guide/custom_policy
   guide/custom_env
   guide/callbacks
   guide/tensorboard
   guide/integrations
   guide/rl_zoo
   guide/sb3_contrib
   guide/sbx
   guide/imitation
   guide/migration
   guide/checking_nan
   guide/developer
   guide/save_format
   guide/export


.. toctree::
  :maxdepth: 1
  :caption: RL Algorithms

  modules/base
  modules/a2c
  modules/ddpg
  modules/dqn
  modules/her
  modules/ppo
  modules/sac
  modules/td3

.. toctree::
  :maxdepth: 1
  :caption: Common

  common/atari_wrappers
  common/env_util
  common/envs
  common/distributions
  common/evaluation
  common/env_checker
  common/monitor
  common/logger
  common/noise
  common/utils

.. toctree::
  :maxdepth: 1
  :caption: Misc

  misc/changelog
  misc/projects


Citing Stable Baselines3
------------------------
To cite this project in publications:

.. code-block:: bibtex

  @article{stable-baselines3,
    author  = {Antonin Raffin and Ashley Hill and Adam Gleave and Anssi Kanervisto and Maximilian Ernestus and Noah Dormann},
    title   = {Stable-Baselines3: Reliable Reinforcement Learning Implementations},
    journal = {Journal of Machine Learning Research},
    year    = {2021},
    volume  = {22},
    number  = {268},
    pages   = {1-8},
    url     = {http://jmlr.org/papers/v22/20-1364.html}
  }

Note: If you need to refer to a specific version of SB3, you can also use the `Zenodo DOI <https://doi.org/10.5281/zenodo.8123988>`_.

Contributing
------------

To any interested in making the rl baselines better, there are still some improvements
that need to be done.
You can check issues in the `repository <https://github.com/DLR-RM/stable-baselines3/labels/help%20wanted>`_.

If you want to contribute, please read `CONTRIBUTING.md <https://github.com/DLR-RM/stable-baselines3/blob/master/CONTRIBUTING.md>`_ first.

Indices and tables
-------------------

* :ref:`genindex`
* :ref:`search`
* :ref:`modindex`



================================================
FILE: docs/make.bat
================================================
@ECHO OFF

pushd %~dp0

REM Command file for Sphinx documentation

if "%SPHINXBUILD%" == "" (
	set SPHINXBUILD=sphinx-build
)
set SOURCEDIR=.
set BUILDDIR=_build
set SPHINXPROJ=StableBaselines

if "%1" == "" goto help

%SPHINXBUILD% >NUL 2>NUL
if errorlevel 9009 (
	echo.
	echo.The 'sphinx-build' command was not found. Make sure you have Sphinx
	echo.installed, then set the SPHINXBUILD environment variable to point
	echo.to the full path of the 'sphinx-build' executable. Alternatively you
	echo.may add the Sphinx directory to PATH.
	echo.
	echo.If you don't have Sphinx installed, grab it from
	echo.http://sphinx-doc.org/
	exit /b 1
)

%SPHINXBUILD% -M %1 %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%
goto end

:help
%SPHINXBUILD% -M help %SOURCEDIR% %BUILDDIR% %SPHINXOPTS%

:end
popd



================================================
FILE: docs/Makefile
================================================
# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
# For debug: SPHINXOPTS = -nWT --keep-going -vvv
SPHINXOPTS    = -W  # make warnings fatal
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = StableBaselines
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)



================================================
FILE: docs/spelling_wordlist.txt
================================================
py
env
atari
argparse
Argparse
TensorFlow
feedforward
envs
VecEnv
pretrain
petrained
tf
th
nn
np
str
mujoco
cpu
ndarray
ndarrays
timestep
timesteps
stepsize
dataset
adam
fn
normalisation
Kullback
Leibler
boolean
deserialized
pretrained
minibatch
subprocesses
ArgumentParser
Tensorflow
Gaussian
approximator
minibatches
hyperparameters
hyperparameter
vectorized
rl
colab
dataloader
npz
datasets
vf
logits
num
Utils
backpropagate
prepend
NaN
preprocessing
Cloudpickle
async
multiprocess
tensorflow
mlp
cnn
neglogp
tanh
coef
repo
Huber
params
ppo
arxiv
Arxiv
func
DQN
Uhlenbeck
Ornstein
multithread
cancelled
Tensorboard
parallelize
customising
serializable
Multiprocessed
cartpole
toolset
lstm
rescale
ffmpeg
avconv
unnormalized
Github
pre
preprocess
backend
attr
preprocess
Antonin
Raffin
araffin
Homebrew
Numpy
Theano
rollout
kfac
Piecewise
csv
nvidia
visdom
tensorboard
preprocessed
namespace
sklearn
GoalEnv
Torchy
pytorch
dicts
optimizers
Deprecations
forkserver
cuda
Polyak
gSDE
rollouts
Pyro
softmax
stdout
Contrib
Quantile



================================================
FILE: docs/_static/css/baselines_theme.css
================================================
/* Main colors  adapted from pytorch doc */
:root{
  --main-bg-color: #343A40;
  --link-color: #FD7E14;
}

/* Header fonts y */
h1, h2, .rst-content .toctree-wrapper p.caption, h3, h4, h5, h6, legend, p.caption {
    font-family: "Lato","proxima-nova","Helvetica Neue",Arial,sans-serif;
}


/* Docs background */
.wy-side-nav-search{
  background-color: var(--main-bg-color);
}

/* Mobile version */
.wy-nav-top{
  background-color: var(--main-bg-color);
}

/* Change link colors (except for the menu) */
a {
    color: var(--link-color);
}

a:hover {
    color: #4F778F;
}

.wy-menu a {
    color: #b3b3b3;
}

.wy-menu a:hover {
    color: #b3b3b3;
}

a.icon.icon-home {
    color: #b3b3b3;
}

.version{
    color: var(--link-color) !important;
}


/* Make code blocks have a background */
.codeblock,pre.literal-block,.rst-content .literal-block,.rst-content pre.literal-block,div[class^='highlight'] {
        background: #f8f8f8;;
}

/* Change style of types in the docstrings .rst-content .field-list */
.field-list .xref.py.docutils, .field-list code.docutils, .field-list .docutils.literal.notranslate
{
  border: None;
  padding-left: 0;
  padding-right: 0;
  color: #404040;
}




================================================
FILE: docs/common/atari_wrappers.rst
================================================
.. _atari_wrapper:

Atari Wrappers
==============

.. automodule:: stable_baselines3.common.atari_wrappers
  :members:



================================================
FILE: docs/common/distributions.rst
================================================
.. _distributions:

Probability Distributions
=========================

Probability distributions used for the different action spaces:

- ``CategoricalDistribution`` -> Discrete
- ``DiagGaussianDistribution`` -> Box (continuous actions)
- ``StateDependentNoiseDistribution`` -> Box (continuous actions) when ``use_sde=True``

.. - ``MultiCategoricalDistribution`` -> MultiDiscrete
.. - ``BernoulliDistribution`` -> MultiBinary

The policy networks output parameters for the distributions (named ``flat`` in the methods).
Actions are then sampled from those distributions.

For instance, in the case of discrete actions. The policy network outputs probability
of taking each action. The ``CategoricalDistribution`` allows sampling from it,
computes the entropy, the log probability (``log_prob``) and backpropagate the gradient.

In the case of continuous actions, a Gaussian distribution is used. The policy network outputs
mean and (log) std of the distribution (assumed to be a ``DiagGaussianDistribution``).

.. automodule:: stable_baselines3.common.distributions
  :members:



================================================
FILE: docs/common/env_checker.rst
================================================
.. _env_checker:

Gym Environment Checker
========================

.. automodule:: stable_baselines3.common.env_checker
  :members:



================================================
FILE: docs/common/env_util.rst
================================================
.. _env_util:

Environments Utils
=========================

.. automodule:: stable_baselines3.common.env_util
  :members:



================================================
FILE: docs/common/envs.rst
================================================
.. _envs:

.. automodule:: stable_baselines3.common.envs



Custom Environments
===================

Those environments were created for testing purposes.


BitFlippingEnv
--------------

.. autoclass:: BitFlippingEnv
  :members:


SimpleMultiObsEnv
-----------------

.. autoclass:: SimpleMultiObsEnv
  :members:


================================================
FILE: docs/common/evaluation.rst
================================================
.. _eval:

Evaluation Helper
=================

.. automodule:: stable_baselines3.common.evaluation
  :members:



================================================
FILE: docs/common/logger.rst
================================================
.. _logger:

Logger
======

To overwrite the default logger, you can pass one to the algorithm.
Available formats are ``["stdout", "csv", "log", "tensorboard", "json"]``.


.. warning::

  When passing a custom logger object,
  this will overwrite ``tensorboard_log`` and ``verbose`` settings
  passed to the constructor.


.. code-block:: python

  from stable_baselines3 import A2C
  from stable_baselines3.common.logger import configure

  tmp_path = "/tmp/sb3_log/"
  # set up logger
  new_logger = configure(tmp_path, ["stdout", "csv", "tensorboard"])

  model = A2C("MlpPolicy", "CartPole-v1", verbose=1)
  # Set new logger
  model.set_logger(new_logger)
  model.learn(10000)


Explanation of logger output
----------------------------

You can find below short explanations of the values logged in Stable-Baselines3 (SB3).
Depending on the algorithm used and of the wrappers/callbacks applied, SB3 only logs a subset of those keys during training.

Below you can find an example of the logger output when training a PPO agent:

.. code-block:: bash

  -----------------------------------------
  | eval/                   |             |
  |    mean_ep_length       | 200         |
  |    mean_reward          | -157        |
  | rollout/                |             |
  |    ep_len_mean          | 200         |
  |    ep_rew_mean          | -227        |
  | time/                   |             |
  |    fps                  | 972         |
  |    iterations           | 19          |
  |    time_elapsed         | 80          |
  |    total_timesteps      | 77824       |
  | train/                  |             |
  |    approx_kl            | 0.037781604 |
  |    clip_fraction        | 0.243       |
  |    clip_range           | 0.2         |
  |    entropy_loss         | -1.06       |
  |    explained_variance   | 0.999       |
  |    learning_rate        | 0.001       |
  |    loss                 | 0.245       |
  |    n_updates            | 180         |
  |    policy_gradient_loss | -0.00398    |
  |    std                  | 0.205       |
  |    value_loss           | 0.226       |
  -----------------------------------------


eval/
^^^^^
All ``eval/`` values are computed by the ``EvalCallback``.

- ``mean_ep_length``: Mean episode length
- ``mean_reward``: Mean episodic reward (during evaluation)
- ``success_rate``: Mean success rate during evaluation (1.0 means 100% success), the environment info dict must contain an ``is_success`` key to compute that value

rollout/
^^^^^^^^
- ``ep_len_mean``: Mean episode length (averaged over ``stats_window_size`` episodes, 100 by default)
- ``ep_rew_mean``: Mean episodic training reward (averaged over ``stats_window_size`` episodes, 100 by default), a ``Monitor`` wrapper is required to compute that value (automatically added by `make_vec_env`).
- ``exploration_rate``: Current value of the exploration rate when using DQN, it corresponds to the fraction of actions taken randomly (epsilon of the "epsilon-greedy" exploration)
- ``success_rate``: Mean success rate during training (averaged over ``stats_window_size`` episodes, 100 by default), you must pass an extra argument to the ``Monitor`` wrapper to log that value (``info_keywords=("is_success",)``) and provide ``info["is_success"]=True/False`` on the final step of the episode

time/
^^^^^
- ``episodes``: Total number of episodes
- ``fps``: Number of frames per seconds (includes time taken by gradient update)
- ``iterations``: Number of iterations (data collection + policy update for A2C/PPO)
- ``time_elapsed``: Time in seconds since the beginning of training
- ``total_timesteps``: Total number of timesteps (steps in the environments)

train/
^^^^^^
- ``actor_loss``: Current value for the actor loss for off-policy algorithms
- ``approx_kl``: approximate mean KL divergence between old and new policy (for PPO), it is an estimation of how much changes happened in the update
- ``clip_fraction``: mean fraction of surrogate loss that was clipped (above ``clip_range`` threshold) for PPO.
- ``clip_range``: Current value of the clipping factor for the surrogate loss of PPO
- ``critic_loss``: Current value for the critic function loss for off-policy algorithms, usually error between value function output and TD(0), temporal difference estimate
- ``ent_coef``: Current value of the entropy coefficient (when using SAC)
- ``ent_coef_loss``: Current value of the entropy coefficient loss (when using SAC)
- ``entropy_loss``: Mean value of the entropy loss (negative of the average policy entropy)
- ``explained_variance``: Fraction of the return variance explained by the value function, see https://scikit-learn.org/stable/modules/model_evaluation.html#explained-variance-score
  (ev=0 => might as well have predicted zero, ev=1 => perfect prediction, ev<0 => worse than just predicting zero)
- ``learning_rate``: Current learning rate value
- ``loss``: Current total loss value
- ``n_updates``: Number of gradient updates applied so far
- ``policy_gradient_loss``: Current value of the policy gradient loss (its value does not have much meaning)
- ``value_loss``: Current value for the value function loss for on-policy algorithms, usually error between value function output and Monte-Carlo estimate (or TD(lambda) estimate)
- ``std``: Current standard deviation of the noise when using generalized State-Dependent Exploration (gSDE)


.. automodule:: stable_baselines3.common.logger
  :members:



================================================
FILE: docs/common/monitor.rst
================================================
.. _monitor:

Monitor Wrapper
===============

.. automodule:: stable_baselines3.common.monitor
  :members:



================================================
FILE: docs/common/noise.rst
================================================
.. _noise:

Action Noise
=============

.. automodule:: stable_baselines3.common.noise
  :members:



================================================
FILE: docs/common/utils.rst
================================================
.. _utils:

Utils
=====

.. automodule:: stable_baselines3.common.utils
  :members:



================================================
FILE: docs/guide/algos.rst
================================================
RL Algorithms
=============

This table displays the rl algorithms that are implemented in the Stable Baselines3 project,
along with some useful characteristics: support for discrete/continuous actions, multiprocessing.


===================  =========== ============ ================= =============== ================
Name                 ``Box``     ``Discrete`` ``MultiDiscrete`` ``MultiBinary`` Multi Processing
===================  =========== ============ ================= =============== ================
ARS [#f1]_           ✔️           ✔️            ❌                 ❌              ✔️
A2C                  ✔️           ✔️            ✔️                 ✔️               ✔️
CrossQ [#f1]_        ✔️           ❌            ❌                ❌               ✔️
DDPG                 ✔️           ❌            ❌                ❌               ✔️
DQN                  ❌           ✔️            ❌                ❌               ✔️
HER                  ✔️           ✔️            ❌                ❌               ✔️
PPO                  ✔️           ✔️            ✔️                 ✔️               ✔️
QR-DQN [#f1]_        ❌          ️ ✔️            ❌                ❌               ✔️
RecurrentPPO [#f1]_  ✔️           ✔️             ✔️                ✔️               ✔️
SAC                  ✔️           ❌            ❌                ❌               ✔️
TD3                  ✔️           ❌            ❌                ❌               ✔️
TQC [#f1]_           ✔️           ❌            ❌                ❌               ✔️
TRPO  [#f1]_         ✔️           ✔️            ✔️                 ✔️               ✔️
Maskable PPO [#f1]_  ❌           ✔️            ✔️                 ✔️               ✔️
===================  =========== ============ ================= =============== ================


.. [#f1] Implemented in `SB3 Contrib <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>`_

.. note::
  ``Tuple`` observation spaces are not supported by any environment,
  however, single-level ``Dict`` spaces are (cf. :ref:`Examples <examples>`).


Actions ``gym.spaces``:

-  ``Box``: A N-dimensional box that contains every point in the action
   space.
-  ``Discrete``: A list of possible actions, where each timestep only
   one of the actions can be used.
-  ``MultiDiscrete``: A list of possible actions, where each timestep only one action of each discrete set can be used.
- ``MultiBinary``: A list of possible actions, where each timestep any of the actions can be used in any combination.


.. note::

  More algorithms (like QR-DQN or TQC) are implemented in our :ref:`contrib repo <sb3_contrib>`
  and in our :ref:`SBX (SB3 + Jax) repo <sbx>` (DroQ, CrossQ, ...).

.. note::

  Some logging values (like ``ep_rew_mean``, ``ep_len_mean``) are only available when using a ``Monitor`` wrapper
  See `Issue #339 <https://github.com/hill-a/stable-baselines/issues/339>`_ for more info.


.. note::

  When using off-policy algorithms, `Time Limits <https://arxiv.org/abs/1712.00378>`_ (aka timeouts) are handled
  properly (cf. `issue #284 <https://github.com/DLR-RM/stable-baselines3/issues/284>`_).
  You can revert to SB3 < 2.1.0 behavior by passing ``handle_timeout_termination=False``
  via the ``replay_buffer_kwargs`` argument.



Reproducibility
---------------

Completely reproducible results are not guaranteed across PyTorch releases or different platforms.
Furthermore, results need not be reproducible between CPU and GPU executions, even when using identical seeds.

In order to make computations deterministics, on your specific problem on one specific platform,
you need to pass a ``seed`` argument at the creation of a model.
If you pass an environment to the model using ``set_env()``, then you also need to seed the environment first.


Credit: part of the *Reproducibility* section comes from `PyTorch Documentation <https://pytorch.org/docs/stable/notes/randomness.html>`_



================================================
FILE: docs/guide/callbacks.rst
================================================
.. _callbacks:

Callbacks
=========

A callback is a set of functions that will be called at given stages of the training procedure.
You can use callbacks to access internal state of the RL model during training.
It allows one to do monitoring, auto saving, model manipulation, progress bars, ...


Custom Callback
---------------

To build a custom callback, you need to create a class that derives from ``BaseCallback``.
This will give you access to events (``_on_training_start``, ``_on_step``) and useful variables (like `self.model` for the RL model).


You can find two examples of custom callbacks in the documentation: one for saving the best model according to the training reward (see :ref:`Examples <examples>`), and one for logging additional values with Tensorboard (see :ref:`Tensorboard section <tensorboard>`).


.. code-block:: python

    from stable_baselines3.common.callbacks import BaseCallback


    class CustomCallback(BaseCallback):
        """
        A custom callback that derives from ``BaseCallback``.

        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        def __init__(self, verbose: int = 0):
            super().__init__(verbose)
            # Those variables will be accessible in the callback
            # (they are defined in the base class)
            # The RL model
            # self.model = None  # type: BaseAlgorithm
            # An alias for self.model.get_env(), the environment used for training
            # self.training_env # type: VecEnv
            # Number of time the callback was called
            # self.n_calls = 0  # type: int
            # num_timesteps = n_envs * n times env.step() was called
            # self.num_timesteps = 0  # type: int
            # local and global variables
            # self.locals = {}  # type: Dict[str, Any]
            # self.globals = {}  # type: Dict[str, Any]
            # The logger object, used to report things in the terminal
            # self.logger # type: stable_baselines3.common.logger.Logger
            # Sometimes, for event callback, it is useful
            # to have access to the parent object
            # self.parent = None  # type: Optional[BaseCallback]

        def _on_training_start(self) -> None:
            """
            This method is called before the first rollout starts.
            """
            pass

        def _on_rollout_start(self) -> None:
            """
            A rollout is the collection of environment interaction
            using the current policy.
            This event is triggered before collecting new samples.
            """
            pass

        def _on_step(self) -> bool:
            """
            This method will be called by the model after each call to `env.step()`.

            For child callback (of an `EventCallback`), this will be called
            when the event is triggered.

            :return: If the callback returns False, training is aborted early.
            """
            return True

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            pass

        def _on_training_end(self) -> None:
            """
            This event is triggered before exiting the `learn()` method.
            """
            pass


.. note::
  ``self.num_timesteps`` corresponds to the total number of steps taken in the environment, i.e., it is the number of environments multiplied by the number of time ``env.step()`` was called

  For the other algorithms, ``self.num_timesteps`` is incremented by ``n_envs`` (number of environments) after each call to ``env.step()``


.. note::

  For off-policy algorithms like SAC, DDPG, TD3 or DQN, the notion of ``rollout`` corresponds to the steps taken in the environment between two updates.


.. _EventCallback:

Event Callback
--------------

Compared to Keras, Stable Baselines provides a second type of ``BaseCallback``, named ``EventCallback`` that is meant to trigger events. When an event is triggered, then a child callback is called.

As an example, :ref:`EvalCallback` is an ``EventCallback`` that will trigger its child callback when there is a new best model.
A child callback is for instance :ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>` that stops the training if the mean reward achieved by the RL model is above a threshold.

.. note::

	We recommend taking a look at the source code of :ref:`EvalCallback` and :ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>` to have a better overview of what can be achieved with this kind of callbacks.


.. code-block:: python

    class EventCallback(BaseCallback):
        """
        Base class for triggering callback on event.

        :param callback: Callback that will be called when an event is triggered.
        :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
        """
        def __init__(self, callback: BaseCallback, verbose: int = 0):
            super().__init__(verbose=verbose)
            self.callback = callback
            # Give access to the parent
            self.callback.parent = self
        ...

        def _on_event(self) -> bool:
            return self.callback()


Callback Collection
-------------------

Stable Baselines provides you with a set of common callbacks for:

- saving the model periodically (:ref:`CheckpointCallback`)
- evaluating the model periodically and saving the best one (:ref:`EvalCallback`)
- chaining callbacks (:ref:`CallbackList`)
- triggering callback on events (:ref:`EventCallback`, :ref:`EveryNTimesteps`)
- logging data every N timesteps (:ref:`LogEveryNTimesteps`)
- stopping the training early based on a reward threshold (:ref:`StopTrainingOnRewardThreshold <StopTrainingCallback>`)


.. _CheckpointCallback:

CheckpointCallback
^^^^^^^^^^^^^^^^^^

Callback for saving a model every ``save_freq`` calls to ``env.step()``, you must specify a log folder (``save_path``)
and optionally a prefix for the checkpoints (``rl_model`` by default).
If you are using this callback to stop and resume training, you may want to optionally save the replay buffer if the
model has one (``save_replay_buffer``, ``False`` by default).
Additionally, if your environment uses a :ref:`VecNormalize <vec_env>` wrapper, you can save the
corresponding statistics using ``save_vecnormalize`` (``False`` by default).

.. warning::

  When using multiple environments, each call to ``env.step()`` will effectively correspond to ``n_envs`` steps.
  If you want the ``save_freq`` to be similar when using a different number of environments,
  you need to account for it using ``save_freq = max(save_freq // n_envs, 1)``.
  The same goes for the other callbacks.


.. code-block:: python

  from stable_baselines3 import SAC
  from stable_baselines3.common.callbacks import CheckpointCallback

  # Save a checkpoint every 1000 steps
  checkpoint_callback = CheckpointCallback(
    save_freq=1000,
    save_path="./logs/",
    name_prefix="rl_model",
    save_replay_buffer=True,
    save_vecnormalize=True,
  )

  model = SAC("MlpPolicy", "Pendulum-v1")
  model.learn(2000, callback=checkpoint_callback)


.. _EvalCallback:

EvalCallback
^^^^^^^^^^^^

Evaluate periodically the performance of an agent, using a separate test environment.
It will save the best model if ``best_model_save_path`` folder is specified and save the evaluations results in a NumPy archive (``evaluations.npz``) if ``log_path`` folder is specified.


.. note::

	You can pass child callbacks via ``callback_after_eval`` and ``callback_on_new_best`` arguments. ``callback_after_eval`` will be triggered after every evaluation, and ``callback_on_new_best`` will be triggered each time there is a new best model.


.. warning::

  You need to make sure that ``eval_env`` is wrapped the same way as the training environment, for instance using the ``VecTransposeImage`` wrapper if you have a channel-last image as input.
  The ``EvalCallback`` class outputs a warning if it is not the case.


.. code-block:: python

    import gymnasium as gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback

    # Separate evaluation env
    eval_env = gym.make("Pendulum-v1")
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/",
                                 log_path="./logs/", eval_freq=500,
                                 deterministic=True, render=False)

    model = SAC("MlpPolicy", "Pendulum-v1")
    model.learn(5000, callback=eval_callback)

.. _ProgressBarCallback:

ProgressBarCallback
^^^^^^^^^^^^^^^^^^^

Display a progress bar with the current progress, elapsed time and estimated remaining time.
This callback is integrated inside SB3 via the ``progress_bar`` argument of the ``learn()`` method.

.. note::

	``ProgressBarCallback`` callback requires ``tqdm`` and ``rich`` packages to be installed. This is done automatically when using ``pip install stable-baselines3[extra]``


.. code-block:: python

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import ProgressBarCallback

    model = PPO("MlpPolicy", "Pendulum-v1")
    # Display progress bar using the progress bar callback
    # this is equivalent to model.learn(100_000, callback=ProgressBarCallback())
    model.learn(100_000, progress_bar=True)


.. _Callbacklist:

CallbackList
^^^^^^^^^^^^

Class for chaining callbacks, they will be called sequentially.
Alternatively, you can pass directly a list of callbacks to the ``learn()`` method, it will be converted automatically to a ``CallbackList``.


.. code-block:: python

    import gymnasium as gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path="./logs/")
    # Separate evaluation env
    eval_env = gym.make("Pendulum-v1")
    eval_callback = EvalCallback(eval_env, best_model_save_path="./logs/best_model",
                                 log_path="./logs/results", eval_freq=500)
    # Create the callback list
    callback = CallbackList([checkpoint_callback, eval_callback])

    model = SAC("MlpPolicy", "Pendulum-v1")
    # Equivalent to:
    # model.learn(5000, callback=[checkpoint_callback, eval_callback])
    model.learn(5000, callback=callback)


.. _StopTrainingCallback:

StopTrainingOnRewardThreshold
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stop the training once a threshold in episodic reward (mean episode reward over the evaluations) has been reached (i.e., when the model is good enough).
It must be used with the :ref:`EvalCallback` and use the event triggered by a new best model.


.. code-block:: python

    import gymnasium as gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold

    # Separate evaluation env
    eval_env = gym.make("Pendulum-v1")
    # Stop training when the model reaches the reward threshold
    callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=-200, verbose=1)
    eval_callback = EvalCallback(eval_env, callback_on_new_best=callback_on_best, verbose=1)

    model = SAC("MlpPolicy", "Pendulum-v1", verbose=1)
    # Almost infinite number of timesteps, but the training will stop
    # early as soon as the reward threshold is reached
    model.learn(int(1e10), callback=eval_callback)


.. _EveryNTimesteps:

EveryNTimesteps
^^^^^^^^^^^^^^^

An :ref:`EventCallback` that will trigger its child callback every ``n_steps`` timesteps.


.. note::

	Because of the way ``VecEnv`` work, ``n_steps`` is a lower bound between two events when using multiple environments.


.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import PPO
  from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps

  # this is equivalent to defining CheckpointCallback(save_freq=500)
  # checkpoint_callback will be triggered every 500 steps
  checkpoint_on_event = CheckpointCallback(save_freq=1, save_path="./logs/")
  event_callback = EveryNTimesteps(n_steps=500, callback=checkpoint_on_event)

  model = PPO("MlpPolicy", "Pendulum-v1", verbose=1)

  model.learn(20_000, callback=event_callback)

.. _LogEveryNTimesteps:

LogEveryNTimesteps
^^^^^^^^^^^^^^^^^^

A callback derived from :ref:`EveryNTimesteps` that will dump the logged data every ``n_steps`` timesteps.


.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import PPO
  from stable_baselines3.common.callbacks import LogEveryNTimesteps

  event_callback = LogEveryNTimesteps(n_steps=1_000)

  model = PPO("MlpPolicy", "Pendulum-v1", verbose=1)

  # Disable auto-logging by passing `log_interval=None`
  model.learn(10_000, callback=event_callback, log_interval=None)



.. _StopTrainingOnMaxEpisodes:

StopTrainingOnMaxEpisodes
^^^^^^^^^^^^^^^^^^^^^^^^^

Stop the training upon reaching the maximum number of episodes, regardless of the model's ``total_timesteps`` value.
Also, presumes that, for multiple environments, the desired behavior is that the agent trains on each env for ``max_episodes``
and in total for ``max_episodes * n_envs`` episodes.


.. note::
    For multiple environments, the agent will train for a total of ``max_episodes * n_envs`` episodes.
    However, it can't be guaranteed that this training will occur for an exact number of ``max_episodes`` per environment.
    Thus, there is an assumption that, on average, each environment ran for ``max_episodes``.


.. code-block:: python

    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import StopTrainingOnMaxEpisodes

    # Stops training when the model reaches the maximum number of episodes
    callback_max_episodes = StopTrainingOnMaxEpisodes(max_episodes=5, verbose=1)

    model = A2C("MlpPolicy", "Pendulum-v1", verbose=1)
    # Almost infinite number of timesteps, but the training will stop
    # early as soon as the max number of episodes is reached
    model.learn(int(1e10), callback=callback_max_episodes)

.. _StopTrainingOnNoModelImprovement:

StopTrainingOnNoModelImprovement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stop the training if there is no new best model (no new best mean reward) after more than a specific number of consecutive evaluations.
The idea is to save time in experiments when you know that the learning curves are somehow well-behaved and, therefore,
after many evaluations without improvement the learning has probably stabilized.
It must be used with the :ref:`EvalCallback` and use the event triggered after every evaluation.


.. code-block:: python

    import gymnasium as gym

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement

    # Separate evaluation env
    eval_env = gym.make("Pendulum-v1")
    # Stop training if there is no improvement after more than 3 evaluations
    stop_train_callback = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3, min_evals=5, verbose=1)
    eval_callback = EvalCallback(eval_env, eval_freq=1000, callback_after_eval=stop_train_callback, verbose=1)

    model = SAC("MlpPolicy", "Pendulum-v1", learning_rate=1e-3, verbose=1)
    # Almost infinite number of timesteps, but the training will stop early
    # as soon as the the number of consecutive evaluations without model
    # improvement is greater than 3
    model.learn(int(1e10), callback=eval_callback)


.. automodule:: stable_baselines3.common.callbacks
  :members:



================================================
FILE: docs/guide/checking_nan.rst
================================================
Dealing with NaNs and infs
==========================

During the training of a model on a given environment, it is possible that the RL model becomes completely
corrupted when a NaN or an inf is given or returned from the RL model.

How and why?
------------

The issue arises when NaNs or infs do not crash, but simply get propagated through the training,
until all the floating point number converge to NaN or inf. This is in line with the
`IEEE Standard for Floating-Point Arithmetic (IEEE 754) <https://ieeexplore.ieee.org/document/4610935>`_ standard, as it says:

.. note::
    Five possible exceptions can occur:
        - Invalid operation (:math:`\sqrt{-1}`, :math:`\inf \times 1`, :math:`\text{NaN}\ \mathrm{mod}\ 1`, ...) return NaN
        - Division by zero:
            - if the operand is not zero (:math:`1/0`, :math:`-2/0`, ...) returns :math:`\pm\inf`
            - if the operand is zero (:math:`0/0`) returns signaling NaN
        - Overflow (exponent too high to represent) returns :math:`\pm\inf`
        - Underflow (exponent too low to represent) returns :math:`0`
        - Inexact (not representable exactly in base 2, eg: :math:`1/5`) returns the rounded value (ex: :code:`assert (1/5) * 3 == 0.6000000000000001`)

And of these, only ``Division by zero`` will signal an exception, the rest will propagate invalid values quietly.

In python, dividing by zero will indeed raise the exception: ``ZeroDivisionError: float division by zero``,
but ignores the rest.

The default in numpy, will warn: ``RuntimeWarning: invalid value encountered``
but will not halt the code.


Anomaly detection with PyTorch
------------------------------

To enable NaN detection in PyTorch you can do

.. code-block:: python

  import torch as th
  th.autograd.set_detect_anomaly(True)


Numpy parameters
----------------

Numpy has a convenient way of dealing with invalid value: `numpy.seterr <https://docs.scipy.org/doc/numpy/reference/generated/numpy.seterr.html>`_,
which defines for the python process, how it should handle floating point error.

.. code-block:: python

  import numpy as np

  np.seterr(all="raise")  # define before your code.

  print("numpy test:")

  a = np.float64(1.0)
  b = np.float64(0.0)
  val = a / b  # this will now raise an exception instead of a warning.
  print(val)

but this will also avoid overflow issues on floating point numbers:

.. code-block:: python

  import numpy as np

  np.seterr(all="raise")  # define before your code.

  print("numpy overflow test:")

  a = np.float64(10)
  b = np.float64(1000)
  val = a ** b  # this will now raise an exception
  print(val)

but will not avoid the propagation issues:

.. code-block:: python

  import numpy as np

  np.seterr(all="raise")  # define before your code.

  print("numpy propagation test:")

  a = np.float64("NaN")
  b = np.float64(1.0)
  val = a + b  # this will neither warn nor raise anything
  print(val)


VecCheckNan Wrapper
-------------------

In order to find when and from where the invalid value originated from, stable-baselines3 comes with a ``VecCheckNan`` wrapper.

It will monitor the actions, observations, and rewards, indicating what action or observation caused it and from what.

.. code-block:: python

  import gymnasium as gym
  from gymnasium import spaces
  import numpy as np

  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import DummyVecEnv, VecCheckNan

  class NanAndInfEnv(gym.Env):
      """Custom Environment that raised NaNs and Infs"""
      metadata = {"render.modes": ["human"]}

      def __init__(self):
          super(NanAndInfEnv, self).__init__()
          self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)
          self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float64)

      def step(self, _action):
          randf = np.random.rand()
          if randf > 0.99:
              obs = float("NaN")
          elif randf > 0.98:
              obs = float("inf")
          else:
              obs = randf
          return [obs], 0.0, False, {}

      def reset(self):
          return [0.0]

      def render(self, close=False):
          pass

  # Create environment
  env = DummyVecEnv([lambda: NanAndInfEnv()])
  env = VecCheckNan(env, raise_exception=True)

  # Instantiate the agent
  model = PPO("MlpPolicy", env)

  # Train the agent
  model.learn(total_timesteps=int(2e5))  # this will crash explaining that the invalid value originated from the environment.

RL Model hyperparameters
------------------------

Depending on your hyperparameters, NaN can occurs much more often.
A great example of this: https://github.com/hill-a/stable-baselines/issues/340

Be aware, the hyperparameters given by default seem to work in most cases,
however your environment might not play nice with them.
If this is the case, try to read up on the effect each hyperparameters has on the model,
so that you can try and tune them to get a stable model. Alternatively, you can try automatic hyperparameter tuning (included in the rl zoo).

Missing values from datasets
----------------------------

If your environment is generated from an external dataset, do not forget to make sure your dataset does not contain NaNs.
As some datasets will sometimes fill missing values with NaNs as a surrogate value.

Here is some reading material about finding NaNs: https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html

And filling the missing values with something else (imputation): https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4



================================================
FILE: docs/guide/custom_env.rst
================================================
.. _custom_env:

Using Custom Environments
==========================

To use the RL baselines with custom environments, they just need to follow the *gymnasium* `interface <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#sphx-glr-tutorials-gymnasium-basics-environment-creation-py>`_.
That is to say, your environment must implement the following methods (and inherits from Gym Class):


.. note::

  If you are using images as input, the observation must be of type ``np.uint8`` and be within a space ``Box`` bounded by [0, 255] (``Box(low=0, high=255, shape=(<your image shape>)``).
  By default, the observation is normalized by SB3 pre-processing (dividing by 255 to have values in [0, 1], i.e. ``Box(low=0, high=1)``) when using CNN policies.
  Images can be either channel-first or channel-last.

  If you want to use ``CnnPolicy`` or ``MultiInputPolicy`` with image-like observation (3D tensor) that are already normalized, you must pass ``normalize_images=False``
  to the policy (using ``policy_kwargs`` parameter, ``policy_kwargs=dict(normalize_images=False)``)
  and make sure your image is in the **channel-first** format.


.. note::

  Although SB3 supports both channel-last and channel-first images as input, we recommend using the channel-first convention when possible.
  Under the hood, when a channel-last image is passed, SB3 uses a ``VecTransposeImage`` wrapper to re-order the channels.


.. note::

    SB3 doesn't support ``Discrete`` and ``MultiDiscrete`` spaces with ``start!=0``. However, you can update your environment or use a wrapper to make your env compatible with SB3:

    .. code-block:: python

        import gymnasium as gym

        class ShiftWrapper(gym.Wrapper):
        """Allow to use Discrete() action spaces with start!=0"""
        def __init__(self, env: gym.Env) -> None:
            super().__init__(env)
            assert isinstance(env.action_space, gym.spaces.Discrete)
            self.action_space = gym.spaces.Discrete(env.action_space.n, start=0)

        def step(self, action: int):
            return self.env.step(action + self.env.action_space.start)


.. code-block:: python

  import gymnasium as gym
  import numpy as np
  from gymnasium import spaces


  class CustomEnv(gym.Env):
      """Custom Environment that follows gym interface."""

      metadata = {"render_modes": ["human"], "render_fps": 30}

      def __init__(self, arg1, arg2, ...):
          super().__init__()
          # Define action and observation space
          # They must be gym.spaces objects
          # Example when using discrete actions:
          self.action_space = spaces.Discrete(N_DISCRETE_ACTIONS)
          # Example for using image as input (channel-first; channel-last also works):
          self.observation_space = spaces.Box(low=0, high=255,
                                              shape=(N_CHANNELS, HEIGHT, WIDTH), dtype=np.uint8)

      def step(self, action):
          ...
          return observation, reward, terminated, truncated, info

      def reset(self, seed=None, options=None):
          ...
          return observation, info

      def render(self):
          ...

      def close(self):
          ...


Then you can define and train a RL agent with:

.. code-block:: python

  # Instantiate the env
  env = CustomEnv(arg1, ...)
  # Define and Train the agent
  model = A2C("CnnPolicy", env).learn(total_timesteps=1000)


To check that your environment follows the Gym interface that SB3 supports, please use:

.. code-block:: python

	from stable_baselines3.common.env_checker import check_env

	env = CustomEnv(arg1, ...)
	# It will check your custom environment and output additional warnings if needed
	check_env(env)

Gymnasium also have its own `env checker <https://gymnasium.farama.org/api/utils/#gymnasium.utils.env_checker.check_env>`_ but it checks a superset of what SB3 supports (SB3 does not support all Gym features).

We have created a `colab notebook <https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/sb3/5_custom_gym_env.ipynb>`_ for a concrete example on creating a custom environment along with an example of using it with Stable-Baselines3 interface.

Alternatively, you may look at Gymnasium `built-in environments <https://gymnasium.farama.org>`_.

Optionally, you can also register the environment with gym, that will allow you to create the RL agent in one line (and use ``gym.make()`` to instantiate the env):

.. code-block:: python

	from gymnasium.envs.registration import register
	# Example for the CartPole environment
	register(
	    # unique identifier for the env `name-version`
	    id="CartPole-v1",
	    # path to the class for creating the env
	    # Note: entry_point also accept a class as input (and not only a string)
	    entry_point="gym.envs.classic_control:CartPoleEnv",
	    # Max number of steps per episode, using a `TimeLimitWrapper`
	    max_episode_steps=500,
	)



In the project, for testing purposes, we use a custom environment named ``IdentityEnv``
defined `in this file <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/envs/identity_env.py>`_.
An example of how to use it can be found `here <https://github.com/DLR-RM/stable-baselines3/blob/master/tests/test_identity.py>`_.



================================================
FILE: docs/guide/custom_policy.rst
================================================
.. _custom_policy:

Policy Networks
===============

Stable Baselines3 provides policy networks for images (CnnPolicies),
other type of input features (MlpPolicies) and multiple different inputs (MultiInputPolicies).


.. warning::
  For A2C and PPO, continuous actions are clipped during training and testing
  (to avoid out of bound error). SAC, DDPG and TD3 squash the action, using a ``tanh()`` transformation,
  which handles bounds more correctly.


SB3 Policy
^^^^^^^^^^

SB3 networks are separated into two mains parts (see figure below):

- A features extractor (usually shared between actor and critic when applicable, to save computation)
  whose role is to extract features (i.e. convert to a feature vector) from high-dimensional observations, for instance, a CNN that extracts features from images.
  This is the ``features_extractor_class`` parameter. You can change the default parameters of that features extractor
  by passing a ``features_extractor_kwargs`` parameter.

- A (fully-connected) network that maps the features to actions/value. Its architecture is controlled by the ``net_arch`` parameter.


.. note::

    All observations are first pre-processed (e.g. images are normalized, discrete obs are converted to one-hot vectors, ...) before being fed to the features extractor.
    In the case of vector observations, the features extractor is just a ``Flatten`` layer.


.. image:: ../_static/img/net_arch.png


SB3 policies are usually composed of several networks (actor/critic networks + target networks when applicable) together
with the associated optimizers.

Each of these network have a features extractor followed by a fully-connected network.

.. note::

  When we refer to "policy" in Stable-Baselines3, this is usually an abuse of language compared to RL terminology.
  In SB3, "policy" refers to the class that handles all the networks useful for training,
  so not only the network used to predict actions (the "learned controller").



.. image:: ../_static/img/sb3_policy.png


Default Network Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default network architecture used by SB3 depends on the algorithm and the observation space.
You can visualize the architecture by printing ``model.policy`` (see `issue #329 <https://github.com/DLR-RM/stable-baselines3/issues/329>`_).


For 1D observation space, a 2 layers fully connected net is used with:

- 64 units (per layer) for PPO/A2C/DQN
- 256 units for SAC
- [400, 300] units for TD3/DDPG (values are taken from the original TD3 paper)

For image observation spaces, the "Nature CNN" (see code for more details) is used for feature extraction, and SAC/TD3 also keeps the same fully connected network after it.
The other algorithms only have a linear layer after the CNN.
The CNN is shared between actor and critic for A2C/PPO (on-policy algorithms) to reduce computation.
Off-policy algorithms (TD3, DDPG, SAC, ...) have separate feature extractors: one for the actor and one for the critic, since the best performance is obtained with this configuration.

For mixed observations (dictionary observations), the two architectures from above are used, i.e., CNN for images and then two layers fully-connected network
(with a smaller output size for the CNN).



Custom Network Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^

One way of customising the policy network architecture is to pass arguments when creating the model,
using ``policy_kwargs`` parameter:

.. note::
    An extra linear layer will be added on top of the layers specified in ``net_arch``, in order to have the right output dimensions and activation functions (e.g. Softmax for discrete actions).

    In the following example, as CartPole's action space has a dimension of 2, the final dimensions of the ``net_arch``'s layers will be:


    .. code-block:: none

                obs
                <4>
           /            \
         <32>          <32>
          |              |
         <32>          <32>
          |              |
         <2>            <1>
        action         value


.. code-block:: python

  import gymnasium as gym
  import torch as th

  from stable_baselines3 import PPO

  # Custom actor (pi) and value function (vf) networks
  # of two layers of size 32 each with Relu activation function
  # Note: an extra linear layer will be added on top of the pi and the vf nets, respectively
  policy_kwargs = dict(activation_fn=th.nn.ReLU,
                       net_arch=dict(pi=[32, 32], vf=[32, 32]))
  # Create the agent
  model = PPO("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
  # Retrieve the environment
  env = model.get_env()
  # Train the agent
  model.learn(total_timesteps=20_000)
  # Save the agent
  model.save("ppo_cartpole")

  del model
  # the policy_kwargs are automatically loaded
  model = PPO.load("ppo_cartpole", env=env)


Custom Feature Extractor
^^^^^^^^^^^^^^^^^^^^^^^^

If you want to have a custom features extractor (e.g. custom CNN when using images), you can define class
that derives from ``BaseFeaturesExtractor`` and then pass it to the model when training.


.. note::

  For on-policy algorithms, the features extractor is shared by default between the actor and the critic to save computation (when applicable).
  However, this can be changed setting ``share_features_extractor=False`` in the
  ``policy_kwargs`` (both for on-policy and off-policy algorithms).


.. code-block:: python

  import torch as th
  import torch.nn as nn
  from gymnasium import spaces

  from stable_baselines3 import PPO
  from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


  class CustomCNN(BaseFeaturesExtractor):
      """
      :param observation_space: (gym.Space)
      :param features_dim: (int) Number of features extracted.
          This corresponds to the number of unit for the last layer.
      """

      def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
          super().__init__(observation_space, features_dim)
          # We assume CxHxW images (channels first)
          # Re-ordering will be done by pre-preprocessing or wrapper
          n_input_channels = observation_space.shape[0]
          self.cnn = nn.Sequential(
              nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
              nn.ReLU(),
              nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
              nn.ReLU(),
              nn.Flatten(),
          )

          # Compute shape by doing one forward pass
          with th.no_grad():
              n_flatten = self.cnn(
                  th.as_tensor(observation_space.sample()[None]).float()
              ).shape[1]

          self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

      def forward(self, observations: th.Tensor) -> th.Tensor:
          return self.linear(self.cnn(observations))

  policy_kwargs = dict(
      features_extractor_class=CustomCNN,
      features_extractor_kwargs=dict(features_dim=128),
  )
  model = PPO("CnnPolicy", "BreakoutNoFrameskip-v4", policy_kwargs=policy_kwargs, verbose=1)
  model.learn(1000)


Multiple Inputs and Dictionary Observations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Stable Baselines3 supports handling of multiple inputs by using ``Dict`` Gym space. This can be done using
``MultiInputPolicy``, which by default uses the ``CombinedExtractor`` features extractor to turn multiple
inputs into a single vector, handled by the ``net_arch`` network.

By default, ``CombinedExtractor`` processes multiple inputs as follows:

1. If input is an image (automatically detected, see ``common.preprocessing.is_image_space``), process image with Nature Atari CNN network and
   output a latent vector of size ``256``.
2. If input is not an image, flatten it (no layers).
3. Concatenate all previous vectors into one long vector and pass it to policy.

Much like above, you can define custom features extractors. The following example assumes the environment has two keys in the
observation space dictionary: "image" is a (1,H,W) image (channel first), and "vector" is a (D,) dimensional vector. We process "image" with a simple
downsampling and "vector" with a single linear layer.

.. code-block:: python

  import gymnasium as gym
  import torch as th
  from torch import nn

  from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

  class CustomCombinedExtractor(BaseFeaturesExtractor):
      def __init__(self, observation_space: gym.spaces.Dict):
          # We do not know features-dim here before going over all the items,
          # so put something dummy for now. PyTorch requires calling
          # nn.Module.__init__ before adding modules
          super().__init__(observation_space, features_dim=1)

          extractors = {}

          total_concat_size = 0
          # We need to know size of the output of this extractor,
          # so go over all the spaces and compute output feature sizes
          for key, subspace in observation_space.spaces.items():
              if key == "image":
                  # We will just downsample one channel of the image by 4x4 and flatten.
                  # Assume the image is single-channel (subspace.shape[0] == 0)
                  extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                  total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
              elif key == "vector":
                  # Run through a simple MLP
                  extractors[key] = nn.Linear(subspace.shape[0], 16)
                  total_concat_size += 16

          self.extractors = nn.ModuleDict(extractors)

          # Update the features dim manually
          self._features_dim = total_concat_size

      def forward(self, observations) -> th.Tensor:
          encoded_tensor_list = []

          # self.extractors contain nn.Modules that do all the processing.
          for key, extractor in self.extractors.items():
              encoded_tensor_list.append(extractor(observations[key]))
          # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
          return th.cat(encoded_tensor_list, dim=1)



On-Policy Algorithms
^^^^^^^^^^^^^^^^^^^^

Custom Networks
---------------

If you need a network architecture that is different for the actor and the critic when using ``PPO``, ``A2C`` or ``TRPO``,
you can pass a dictionary of the following structure: ``dict(pi=[<actor network architecture>], vf=[<critic network architecture>])``.

For example, if you want a different architecture for the actor (aka ``pi``) and the critic (value-function aka ``vf``) networks,
then you can specify ``net_arch=dict(pi=[32, 32], vf=[64, 64])``.

Otherwise, to have actor and critic that share the same network architecture,
you only need to specify ``net_arch=[128, 128]`` (here, two hidden layers of 128 units each, this is equivalent to ``net_arch=dict(pi=[128, 128], vf=[128, 128])``).

If shared layers are needed, you need to implement a custom policy network (see `advanced example below <#advanced-example>`_).

Examples
~~~~~~~~

Same architecture for actor and critic with two layers of size 128: ``net_arch=[128, 128]``

.. code-block:: none

            obs
       /            \
     <128>          <128>
      |              |
     <128>          <128>
      |              |
    action         value

Different architectures for actor and critic: ``net_arch=dict(pi=[32, 32], vf=[64, 64])``

.. code-block:: none

            obs
       /            \
     <32>          <64>
      |              |
     <32>          <64>
      |              |
    action         value


Advanced Example
~~~~~~~~~~~~~~~~

If your task requires even more granular control over the policy/value architecture, you can redefine the policy directly:


.. code-block:: python

  from typing import Callable, Dict, List, Optional, Tuple, Type, Union

  from gymnasium import spaces
  import torch as th
  from torch import nn

  from stable_baselines3 import PPO
  from stable_baselines3.common.policies import ActorCriticPolicy


  class CustomNetwork(nn.Module):
      """
      Custom network for policy and value function.
      It receives as input the features extracted by the features extractor.

      :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
      :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
      :param last_layer_dim_vf: (int) number of units for the last layer of the value network
      """

      def __init__(
          self,
          feature_dim: int,
          last_layer_dim_pi: int = 64,
          last_layer_dim_vf: int = 64,
      ):
          super().__init__()

          # IMPORTANT:
          # Save output dimensions, used to create the distributions
          self.latent_dim_pi = last_layer_dim_pi
          self.latent_dim_vf = last_layer_dim_vf

          # Policy network
          self.policy_net = nn.Sequential(
              nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU()
          )
          # Value network
          self.value_net = nn.Sequential(
              nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU()
          )

      def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
          """
          :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
              If all layers are shared, then ``latent_policy == latent_value``
          """
          return self.forward_actor(features), self.forward_critic(features)

      def forward_actor(self, features: th.Tensor) -> th.Tensor:
          return self.policy_net(features)

      def forward_critic(self, features: th.Tensor) -> th.Tensor:
          return self.value_net(features)


  class CustomActorCriticPolicy(ActorCriticPolicy):
      def __init__(
          self,
          observation_space: spaces.Space,
          action_space: spaces.Space,
          lr_schedule: Callable[[float], float],
          *args,
          **kwargs,
      ):
          # Disable orthogonal initialization
          kwargs["ortho_init"] = False
          super().__init__(
              observation_space,
              action_space,
              lr_schedule,
              # Pass remaining arguments to base class
              *args,
              **kwargs,
          )


      def _build_mlp_extractor(self) -> None:
          self.mlp_extractor = CustomNetwork(self.features_dim)


  model = PPO(CustomActorCriticPolicy, "CartPole-v1", verbose=1)
  model.learn(5000)




Off-Policy Algorithms
^^^^^^^^^^^^^^^^^^^^^

If you need a network architecture that is different for the actor and the critic when using ``SAC``, ``DDPG``, ``TQC`` or ``TD3``,
you can pass a dictionary of the following structure: ``dict(pi=[<actor network architecture>], qf=[<critic network architecture>])``.

For example, if you want a different architecture for the actor (aka ``pi``) and the critic (Q-function aka ``qf``) networks,
then you can specify ``net_arch=dict(pi=[64, 64], qf=[400, 300])``.

Otherwise, to have actor and critic that share the same network architecture,
you only need to specify ``net_arch=[256, 256]`` (here, two hidden layers of 256 units each).


.. note::
    For advanced customization of off-policy algorithms policies, please take a look at the code.
    A good understanding of the algorithm used is required, see discussion in `issue #425 <https://github.com/DLR-RM/stable-baselines3/issues/425>`_


.. code-block:: python

  from stable_baselines3 import SAC

  # Custom actor architecture with two layers of 64 units each
  # Custom critic architecture with two layers of 400 and 300 units
  policy_kwargs = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))
  # Create the agent
  model = SAC("MlpPolicy", "Pendulum-v1", policy_kwargs=policy_kwargs, verbose=1)
  model.learn(5000)



================================================
FILE: docs/guide/developer.rst
================================================
.. _developer:

================
Developer Guide
================

This guide is meant for those who want to understand the internals and the design choices of Stable-Baselines3.


At first, you should read the two issues where the design choices were discussed:

- https://github.com/hill-a/stable-baselines/issues/576
- https://github.com/hill-a/stable-baselines/issues/733


The library is not meant to be modular, although inheritance is used to reduce code duplication.


Algorithms Structure
====================


Each algorithm (on-policy and off-policy ones) follows a common structure.
Policy contains code for acting in the environment, and algorithm updates this policy.
There is one folder per algorithm, and in that folder there is the algorithm and the policy definition (``policies.py``).

Each algorithm has two main methods:

- ``.collect_rollouts()`` which defines how new samples are collected, usually inherited from the base class. Those samples are then stored in a ``RolloutBuffer`` (discarded after the gradient update) or ``ReplayBuffer``

- ``.train()`` which updates the parameters using samples from the buffer


.. image:: ../_static/img/sb3_loop.png


Where to start?
===============

The first thing you need to read and understand are the base classes in the ``common/`` folder:

- ``BaseAlgorithm`` in ``base_class.py`` which defines how an RL class should look like.
  It contains also all the "glue code" for saving/loading and the common operations (wrapping environments)

- ``BasePolicy`` in ``policies.py`` which defines how a policy class should look like.
  It contains also all the magic for the ``.predict()`` method, to handle as many spaces/cases as possible

- ``OffPolicyAlgorithm`` in ``off_policy_algorithm.py`` that contains the implementation of ``collect_rollouts()`` for the off-policy algorithms,
  and similarly ``OnPolicyAlgorithm`` in ``on_policy_algorithm.py``.


All the environments handled internally are assumed to be ``VecEnv`` (``gym.Env`` are automatically wrapped).


Pre-Processing
==============

To handle different observation spaces, some pre-processing needs to be done (e.g. one-hot encoding for discrete observation).
Most of the code for pre-processing is in ``common/preprocessing.py`` and ``common/policies.py``.

For images, environment is automatically wrapped with ``VecTransposeImage`` if observations are detected to be images with
channel-last convention to transform it to PyTorch's channel-first convention.


Policy Structure
================

When we refer to "policy" in Stable-Baselines3, this is usually an abuse of language compared to RL terminology.
In SB3, "policy" refers to the class that handles all the networks useful for training,
so not only the network used to predict actions (the "learned controller").
For instance, the ``TD3`` policy contains the actor, the critic and the target networks.

To avoid the hassle of importing specific policy classes for specific algorithm (e.g. both A2C and PPO use ``ActorCriticPolicy``),
SB3 uses names like "MlpPolicy" and "CnnPolicy" to refer policies using small feed-forward networks or convolutional networks,
respectively. Importing ``[algorithm]/policies.py`` registers an appropriate policy for that algorithm under those names.

Probability distributions
=========================

When needed, the policies handle the different probability distributions.
All distributions are located in ``common/distributions.py`` and follow the same interface.
Each distribution corresponds to a type of action space (e.g. ``Categorical`` is the one used for discrete actions.
For continuous actions, we can use multiple distributions ("DiagGaussian", "SquashedGaussian" or "StateDependentDistribution")

State-Dependent Exploration
===========================

State-Dependent Exploration (SDE) is a type of exploration that allows to use RL directly on real robots,
that was the starting point for the Stable-Baselines3 library.
I (@araffin) published a paper about a generalized version of SDE (the one implemented in SB3): https://arxiv.org/abs/2005.05719

Misc
====

The rest of the ``common/`` is composed of helpers (e.g. evaluation helpers) or basic components (like the callbacks).
The ``type_aliases.py`` file contains common type hint aliases like ``GymStepReturn``.

Et voilà?

After reading this guide and the mentioned files, you should be now able to understand the design logic behind the library ;)



================================================
FILE: docs/guide/examples.rst
================================================
.. _examples:

Examples
========

.. note::

  These examples are only to demonstrate the use of the library and its functions, and the trained agents may not solve the environments. Optimized hyperparameters can be found in the RL Zoo `repository <https://github.com/DLR-RM/rl-baselines3-zoo>`_.


Try it online with Colab Notebooks!
-----------------------------------

All the following examples can be executed online using Google colab |colab|
notebooks:

-  `Full Tutorial <https://github.com/araffin/rl-tutorial-jnrr19/tree/sb3>`_
-  `All Notebooks <https://github.com/Stable-Baselines-Team/rl-colab-notebooks/tree/sb3>`_
-  `Getting Started`_
-  `Training, Saving, Loading`_
-  `Multiprocessing`_
-  `Monitor Training and Plotting`_
-  `Atari Games`_
-  `RL Baselines zoo`_
-  `PyBullet`_
-  `Hindsight Experience Replay`_
-  `Advanced Saving and Loading`_

.. _Getting Started: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_getting_started.ipynb
.. _Training, Saving, Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb
.. _Multiprocessing: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb
.. _Monitor Training and Plotting: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb
.. _Atari Games: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
.. _Hindsight Experience Replay: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_her.ipynb
.. _RL Baselines zoo: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb
.. _PyBullet: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb
.. _Advanced Saving and Loading: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb

.. |colab| image:: ../_static/img/colab.svg

Basic Usage: Training, Saving, Loading
--------------------------------------

In the following example, we will train, save and load a DQN model on the Lunar Lander environment.

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/saving_loading_dqn.ipynb


.. figure:: https://cdn-images-1.medium.com/max/960/1*f4VZPKOI0PYNWiwt0la0Rg.gif

  Lunar Lander Environment


.. note::
  LunarLander requires the python package ``box2d``.
  You can install it using ``apt install swig`` and then ``pip install box2d box2d-kengz``

.. warning::
  ``load`` method re-creates the model from scratch and should be called on the Algorithm without instantiating it first,
  e.g. ``model = DQN.load("dqn_lunar", env=env)`` instead of ``model = DQN(env=env)`` followed by  ``model.load("dqn_lunar")``. The latter **will not work** as ``load`` is not an in-place operation.
  If you want to load parameters without re-creating the model, e.g. to evaluate the same model
  with multiple different sets of parameters, consider using ``set_parameters`` instead.

.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import DQN
  from stable_baselines3.common.evaluation import evaluate_policy


  # Create environment
  env = gym.make("LunarLander-v2", render_mode="rgb_array")

  # Instantiate the agent
  model = DQN("MlpPolicy", env, verbose=1)
  # Train the agent and display a progress bar
  model.learn(total_timesteps=int(2e5), progress_bar=True)
  # Save the agent
  model.save("dqn_lunar")
  del model  # delete trained model to demonstrate loading

  # Load the trained agent
  # NOTE: if you have loading issue, you can pass `print_system_info=True`
  # to compare the system on which the model was trained vs the current one
  # model = DQN.load("dqn_lunar", env=env, print_system_info=True)
  model = DQN.load("dqn_lunar", env=env)

  # Evaluate the agent
  # NOTE: If you use wrappers with your environment that modify rewards,
  #       this will be reflected here. To evaluate with original rewards,
  #       wrap environment in a "Monitor" wrapper before other wrappers.
  mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)

  # Enjoy trained agent
  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _states = model.predict(obs, deterministic=True)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render("human")


Multiprocessing: Unleashing the Power of Vectorized Environments
----------------------------------------------------------------

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/multiprocessing_rl.ipynb

.. figure:: https://cdn-images-1.medium.com/max/960/1*h4WTQNVIsvMXJTCpXm_TAw.gif

  CartPole Environment


.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import PPO
  from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
  from stable_baselines3.common.env_util import make_vec_env
  from stable_baselines3.common.utils import set_random_seed

  def make_env(env_id: str, rank: int, seed: int = 0):
      """
      Utility function for multiprocessed env.

      :param env_id: the environment ID
      :param num_env: the number of environments you wish to have in subprocesses
      :param seed: the initial seed for RNG
      :param rank: index of the subprocess
      """
      def _init():
          env = gym.make(env_id, render_mode="human")
          env.reset(seed=seed + rank)
          return env
      set_random_seed(seed)
      return _init

  if __name__ == "__main__":
      env_id = "CartPole-v1"
      num_cpu = 4  # Number of processes to use
      # Create the vectorized environment
      vec_env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)])

      # Stable Baselines provides you with make_vec_env() helper
      # which does exactly the previous steps for you.
      # You can choose between `DummyVecEnv` (usually faster) and `SubprocVecEnv`
      # env = make_vec_env(env_id, n_envs=num_cpu, seed=0, vec_env_cls=SubprocVecEnv)

      model = PPO("MlpPolicy", vec_env, verbose=1)
      model.learn(total_timesteps=25_000)

      obs = vec_env.reset()
      for _ in range(1000):
          action, _states = model.predict(obs)
          obs, rewards, dones, info = vec_env.step(action)
          vec_env.render()


Multiprocessing with off-policy algorithms
------------------------------------------

.. warning::

  When using multiple environments with off-policy algorithms, you should update the ``gradient_steps``
  parameter too. Set it to ``gradient_steps=-1`` to perform as many gradient steps as transitions collected.
  There is usually a compromise between wall-clock time and sample efficiency,
  see this `example in PR #439 <https://github.com/DLR-RM/stable-baselines3/pull/439#issuecomment-961796799>`_


.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import SAC
  from stable_baselines3.common.env_util import make_vec_env

  vec_env = make_vec_env("Pendulum-v0", n_envs=4, seed=0)

  # We collect 4 transitions per call to `env.step()`
  # and performs 2 gradient steps per call to `env.step()`
  # if gradient_steps=-1, then we would do 4 gradients steps per call to `env.step()`
  model = SAC("MlpPolicy", vec_env, train_freq=1, gradient_steps=2, verbose=1)
  model.learn(total_timesteps=10_000)


Dict Observations
-----------------

You can use environments with dictionary observation spaces. This is useful in the case where one can't directly
concatenate observations such as an image from a camera combined with a vector of servo sensor data (e.g., rotation angles).
Stable Baselines3 provides ``SimpleMultiObsEnv`` as an example of this kind of setting.
The environment is a simple grid world, but the observations for each cell come in the form of dictionaries.
These dictionaries are randomly initialized on the creation of the environment and contain a vector observation and an image observation.

.. code-block:: python

  from stable_baselines3 import PPO
  from stable_baselines3.common.envs import SimpleMultiObsEnv


  # Stable Baselines provides SimpleMultiObsEnv as an example environment with Dict observations
  env = SimpleMultiObsEnv(random_start=False)

  model = PPO("MultiInputPolicy", env, verbose=1)
  model.learn(total_timesteps=100_000)


Callbacks: Monitoring Training
------------------------------

.. note::

	We recommend reading the `Callback section <callbacks.html>`_

You can define a custom callback function that will be called inside the agent.
This could be useful when you want to monitor training, for instance display live
learning curves in Tensorboard or save the best agent.
If your callback returns False, training is aborted early.

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/monitor_training.ipynb


.. code-block:: python

  import os

  import gymnasium as gym
  import numpy as np
  import matplotlib.pyplot as plt

  from stable_baselines3 import TD3
  from stable_baselines3.common import results_plotter
  from stable_baselines3.common.monitor import Monitor
  from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
  from stable_baselines3.common.noise import NormalActionNoise
  from stable_baselines3.common.callbacks import BaseCallback


  class SaveOnBestTrainingRewardCallback(BaseCallback):
      """
      Callback for saving a model (the check is done every ``check_freq`` steps)
      based on the training reward (in practice, we recommend using ``EvalCallback``).

      :param check_freq:
      :param log_dir: Path to the folder where the model will be saved.
        It must contains the file created by the ``Monitor`` wrapper.
      :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
      """
      def __init__(self, check_freq: int, log_dir: str, verbose: int = 1):
          super().__init__(verbose)
          self.check_freq = check_freq
          self.log_dir = log_dir
          self.save_path = os.path.join(log_dir, "best_model")
          self.best_mean_reward = -np.inf

      def _init_callback(self) -> None:
          # Create folder if needed
          if self.save_path is not None:
              os.makedirs(self.save_path, exist_ok=True)

      def _on_step(self) -> bool:
          if self.n_calls % self.check_freq == 0:

            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), "timesteps")
            if len(x) > 0:
                # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose >= 1:
                  print(f"Num timesteps: {self.num_timesteps}")
                  print(f"Best mean reward: {self.best_mean_reward:.2f} - Last mean reward per episode: {mean_reward:.2f}")

                # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose >= 1:
                      print(f"Saving new best model to {self.save_path}")
                    self.model.save(self.save_path)

          return True

  # Create log dir
  log_dir = "tmp/"
  os.makedirs(log_dir, exist_ok=True)

  # Create and wrap the environment
  env = gym.make("LunarLanderContinuous-v2")
  env = Monitor(env, log_dir)

  # Add some action noise for exploration
  n_actions = env.action_space.shape[-1]
  action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
  # Because we use parameter noise, we should use a MlpPolicy with layer normalization
  model = TD3("MlpPolicy", env, action_noise=action_noise, verbose=0)
  # Create the callback: check every 1000 steps
  callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)
  # Train the agent
  timesteps = 1e5
  model.learn(total_timesteps=int(timesteps), callback=callback)

  plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS, "TD3 LunarLander")
  plt.show()


Callbacks: Evaluate Agent Performance
-------------------------------------
To periodically evaluate an agent's performance on a separate test environment, use ``EvalCallback``.
You can control the evaluation frequency with ``eval_freq`` to monitor your agent's progress during training.

.. code-block:: python

  import os
  import gymnasium as gym

  from stable_baselines3 import SAC
  from stable_baselines3.common.callbacks import EvalCallback
  from stable_baselines3.common.env_util import make_vec_env

  env_id = "Pendulum-v1"
  n_training_envs = 1
  n_eval_envs = 5

  # Create log dir where evaluation results will be saved
  eval_log_dir = "./eval_logs/"
  os.makedirs(eval_log_dir, exist_ok=True)

  # Initialize a vectorized training environment with default parameters
  train_env = make_vec_env(env_id, n_envs=n_training_envs, seed=0)

  # Separate evaluation env, with different parameters passed via env_kwargs
  # Eval environments can be vectorized to speed up evaluation.
  eval_env = make_vec_env(env_id, n_envs=n_eval_envs, seed=0,
                          env_kwargs={'g':0.7})

  # Create callback that evaluates agent for 5 episodes every 500 training environment steps.
  # When using multiple training environments, agent will be evaluated every
  # eval_freq calls to train_env.step(), thus it will be evaluated every
  # (eval_freq * n_envs) training steps. See EvalCallback doc for more information.
  eval_callback = EvalCallback(eval_env, best_model_save_path=eval_log_dir,
                                log_path=eval_log_dir, eval_freq=max(500 // n_training_envs, 1),
                                n_eval_episodes=5, deterministic=True,
                                render=False)

  model = SAC("MlpPolicy", train_env)
  model.learn(5000, callback=eval_callback)


Atari Games
-----------

.. figure:: ../_static/img/breakout.gif

  Trained A2C agent on Breakout

.. figure:: https://cdn-images-1.medium.com/max/960/1*UHYJE7lF8IDZS_U5SsAFUQ.gif

 Pong Environment


Training a RL agent on Atari games is straightforward thanks to ``make_atari_env`` helper function.
It will do `all the preprocessing <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>`_
and multiprocessing for you. To install the Atari environments, run the command ``pip install gymnasium[atari,accept-rom-license]`` to install the Atari environments and ROMs, or install Stable Baselines3 with ``pip install stable-baselines3[extra]`` to install this and other optional dependencies.

.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/atari_games.ipynb
..

.. code-block:: python

  from stable_baselines3.common.env_util import make_atari_env
  from stable_baselines3.common.vec_env import VecFrameStack
  from stable_baselines3 import A2C

  import ale_py

  # There already exists an environment generator
  # that will make and wrap atari environments correctly.
  # Here we are also multi-worker training (n_envs=4 => 4 environments)
  vec_env = make_atari_env("PongNoFrameskip-v4", n_envs=4, seed=0)
  # Frame-stacking with 4 frames
  vec_env = VecFrameStack(vec_env, n_stack=4)

  model = A2C("CnnPolicy", vec_env, verbose=1)
  model.learn(total_timesteps=25_000)

  obs = vec_env.reset()
  while True:
      action, _states = model.predict(obs, deterministic=False)
      obs, rewards, dones, info = vec_env.step(action)
      vec_env.render("human")


PyBullet: Normalizing input features
------------------------------------

Normalizing input features may be essential to successful training of an RL agent
(by default, images are scaled, but other types of input are not),
for instance when training on `PyBullet <https://github.com/bulletphysics/bullet3/>`__ environments.
For this, there is a wrapper ``VecNormalize`` that will compute a running average and standard deviation of the input features (it can do the same for rewards).


.. note::

	you need to install pybullet envs with ``pip install pybullet_envs_gymnasium``


.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/pybullet.ipynb


.. code-block:: python

    from pathlib import Path

    import pybullet_envs_gymnasium

    from stable_baselines3.common.vec_env import VecNormalize
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3 import PPO

    # Alternatively, you can use the MuJoCo equivalent "HalfCheetah-v4"
    vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
    # Automatically normalize the input features and reward
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    model = PPO("MlpPolicy", vec_env)
    model.learn(total_timesteps=2000)

    # Don't forget to save the VecNormalize statistics when saving the agent
    log_dir = Path("/tmp/")
    model.save(log_dir / "ppo_halfcheetah")
    stats_path = log_dir / "vec_normalize.pkl"
    vec_env.save(stats_path)

    # To demonstrate loading
    del model, vec_env

    # Load the saved statistics
    vec_env = make_vec_env("HalfCheetahBulletEnv-v0", n_envs=1)
    vec_env = VecNormalize.load(stats_path, vec_env)
    #  do not update them at test time
    vec_env.training = False
    # reward normalization is not needed at test time
    vec_env.norm_reward = False

    # Load the agent
    model = PPO.load(log_dir / "ppo_halfcheetah", env=vec_env)


Hindsight Experience Replay (HER)
---------------------------------

For this example, we are using `Highway-Env <https://github.com/eleurent/highway-env>`_ by `@eleurent <https://github.com/eleurent>`_.


.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/stable_baselines_her.ipynb


.. figure:: https://raw.githubusercontent.com/eleurent/highway-env/gh-media/docs/media/parking-env.gif

   The highway-parking-v0 environment.

The parking env is a goal-conditioned continuous control task, in which the vehicle must park in a given space with the appropriate heading.

.. note::

  The hyperparameters in the following example were optimized for that environment.


.. code-block:: python

  import gymnasium as gym
  import highway_env
  import numpy as np

  from stable_baselines3 import HerReplayBuffer, SAC, DDPG, TD3
  from stable_baselines3.common.noise import NormalActionNoise

  env = gym.make("parking-v0")

  # Create 4 artificial transitions per real transition
  n_sampled_goal = 4

  # SAC hyperparams:
  model = SAC(
      "MultiInputPolicy",
      env,
      replay_buffer_class=HerReplayBuffer,
      replay_buffer_kwargs=dict(
        n_sampled_goal=n_sampled_goal,
        goal_selection_strategy="future",
      ),
      verbose=1,
      buffer_size=int(1e6),
      learning_rate=1e-3,
      gamma=0.95,
      batch_size=256,
      policy_kwargs=dict(net_arch=[256, 256, 256]),
  )

  model.learn(int(2e5))
  model.save("her_sac_highway")

  # Load saved model
  # Because it needs access to `env.compute_reward()`
  # HER must be loaded with the env
  env = gym.make("parking-v0", render_mode="human") # Change the render mode
  model = SAC.load("her_sac_highway", env=env)

  obs, info = env.reset()

  # Evaluate the agent
  episode_reward = 0
  for _ in range(100):
      action, _ = model.predict(obs, deterministic=True)
      obs, reward, terminated, truncated, info = env.step(action)
      episode_reward += reward
      if terminated or truncated or info.get("is_success", False):
          print("Reward:", episode_reward, "Success?", info.get("is_success", False))
          episode_reward = 0.0
          obs, info = env.reset()


Learning Rate Schedule
----------------------

All algorithms allow you to pass a learning rate schedule that takes as input the current progress remaining (from 1 to 0).
``PPO``'s ``clip_range``` parameter also accepts such schedule.

The `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ already includes
linear and constant schedules.


.. code-block:: python

  from typing import Callable

  from stable_baselines3 import PPO


  def linear_schedule(initial_value: float) -> Callable[[float], float]:
      """
      Linear learning rate schedule.

      :param initial_value: Initial learning rate.
      :return: schedule that computes
        current learning rate depending on remaining progress
      """
      def func(progress_remaining: float) -> float:
          """
          Progress will decrease from 1 (beginning) to 0.

          :param progress_remaining:
          :return: current learning rate
          """
          return progress_remaining * initial_value

      return func

  # Initial learning rate of 0.001
  model = PPO("MlpPolicy", "CartPole-v1", learning_rate=linear_schedule(0.001), verbose=1)
  model.learn(total_timesteps=20_000)
  # By default, `reset_num_timesteps` is True, in which case the learning rate schedule resets.
  # progress_remaining = 1.0 - (num_timesteps / total_timesteps)
  model.learn(total_timesteps=10_000, reset_num_timesteps=True)


Advanced Saving and Loading
---------------------------------

In this example, we show how to use a policy independently from a model (and how to save it, load it) and save/load a replay buffer.

By default, the replay buffer is not saved when calling ``model.save()``, in order to save space on the disk (a replay buffer can be up to several GB when using images).
However, SB3 provides a ``save_replay_buffer()`` and ``load_replay_buffer()`` method to save it separately.


.. note::

	For training model after loading it, we recommend loading the replay buffer to ensure stable learning (for off-policy algorithms).
	You also need to pass ``reset_num_timesteps=True`` to ``learn`` function which initializes the environment
	and agent for training if a new environment was created since saving the model.


.. image:: ../_static/img/colab-badge.svg
   :target: https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/advanced_saving_loading.ipynb


.. code-block:: python

  from stable_baselines3 import SAC
  from stable_baselines3.common.evaluation import evaluate_policy
  from stable_baselines3.sac.policies import MlpPolicy

  # Create the model and the training environment
  model = SAC("MlpPolicy", "Pendulum-v1", verbose=1,
              learning_rate=1e-3)

  # train the model
  model.learn(total_timesteps=6000)

  # save the model
  model.save("sac_pendulum")

  # the saved model does not contain the replay buffer
  loaded_model = SAC.load("sac_pendulum")
  print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

  # now save the replay buffer too
  model.save_replay_buffer("sac_replay_buffer")

  # load it into the loaded_model
  loaded_model.load_replay_buffer("sac_replay_buffer")

  # now the loaded replay is not empty anymore
  print(f"The loaded_model has {loaded_model.replay_buffer.size()} transitions in its buffer")

  # Save the policy independently from the model
  # Note: if you don't save the complete model with `model.save()`
  # you cannot continue training afterward
  policy = model.policy
  policy.save("sac_policy_pendulum")

  # Retrieve the environment
  env = model.get_env()

  # Evaluate the policy
  mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=10, deterministic=True)

  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

  # Load the policy independently from the model
  saved_policy = MlpPolicy.load("sac_policy_pendulum")

  # Evaluate the loaded policy
  mean_reward, std_reward = evaluate_policy(saved_policy, env, n_eval_episodes=10, deterministic=True)

  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")



Accessing and modifying model parameters
----------------------------------------

You can access model's parameters via ``set_parameters`` and ``get_parameters`` functions,
or via ``model.policy.state_dict()`` (and ``load_state_dict()``),
which use dictionaries that map variable names to PyTorch tensors.

These functions are useful when you need to e.g. evaluate large set of models with same network structure,
visualize different layers of the network or modify parameters manually.

Policies also offers a simple way to save/load weights as a NumPy vector, using ``parameters_to_vector()``
and ``load_from_vector()`` method.

Following example demonstrates reading parameters, modifying some of them and loading them to model
by implementing `evolution strategy (es) <http://blog.otoro.net/2017/10/29/visual-evolution-strategies/>`_
for solving the ``CartPole-v1`` environment. The initial guess for parameters is obtained by running
A2C policy gradient updates on the model.

.. code-block:: python

  from typing import Dict

  import gymnasium as gym
  import numpy as np
  import torch as th

  from stable_baselines3 import A2C
  from stable_baselines3.common.evaluation import evaluate_policy


  def mutate(params: Dict[str, th.Tensor]) -> Dict[str, th.Tensor]:
      """Mutate parameters by adding normal noise to them"""
      return dict((name, param + th.randn_like(param)) for name, param in params.items())


  # Create policy with a small network
  model = A2C(
      "MlpPolicy",
      "CartPole-v1",
      ent_coef=0.0,
      policy_kwargs={"net_arch": [32]},
      seed=0,
      learning_rate=0.05,
  )

  # Use traditional actor-critic policy gradient updates to
  # find good initial parameters
  model.learn(total_timesteps=10_000)

  # Include only variables with "policy", "action" (policy) or "shared_net" (shared layers)
  # in their name: only these ones affect the action.
  # NOTE: you can retrieve those parameters using model.get_parameters() too
  mean_params = dict(
      (key, value)
      for key, value in model.policy.state_dict().items()
      if ("policy" in key or "shared_net" in key or "action" in key)
  )

  # population size of 50 invdiduals
  pop_size = 50
  # Keep top 10%
  n_elite = pop_size // 10
  # Retrieve the environment
  vec_env = model.get_env()

  for iteration in range(10):
      # Create population of candidates and evaluate them
      population = []
      for population_i in range(pop_size):
          candidate = mutate(mean_params)
          # Load new policy parameters to agent.
          # Tell function that it should only update parameters
          # we give it (policy parameters)
          model.policy.load_state_dict(candidate, strict=False)
          # Evaluate the candidate
          fitness, _ = evaluate_policy(model, vec_env)
          population.append((candidate, fitness))
      # Take top 10% and use average over their parameters as next mean parameter
      top_candidates = sorted(population, key=lambda x: x[1], reverse=True)[:n_elite]
      mean_params = dict(
          (
              name,
              th.stack([candidate[0][name] for candidate in top_candidates]).mean(dim=0),
          )
          for name in mean_params.keys()
      )
      mean_fitness = sum(top_candidate[1] for top_candidate in top_candidates) / n_elite
      print(f"Iteration {iteration + 1:<3} Mean top fitness: {mean_fitness:.2f}")
      print(f"Best fitness: {top_candidates[0][1]:.2f}")


SB3 with Isaac Lab, Brax, Procgen, EnvPool
------------------------------------------

Some massively parallel simulations such as `EnvPool <https://github.com/sail-sg/envpool>`_, `Isaac Lab <https://github.com/isaac-sim/IsaacLab>`_, `Brax <https://github.com/google/brax>`_ or `ProcGen <https://github.com/Farama-Foundation/Procgen2>`_ already produce a vectorized environment to speed up data collection (see discussion in `issue #314 <https://github.com/DLR-RM/stable-baselines3/issues/314>`_).

To use SB3 with these tools, you need to wrap the env with tool-specific ``VecEnvWrapper`` that pre-processes the data for SB3,
you can find links to some of these wrappers in `issue #772 <https://github.com/DLR-RM/stable-baselines3/issues/772#issuecomment-1048657002>`_.

- Isaac Lab wrapper: `link <https://github.com/isaac-sim/IsaacLab/blob/main/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/utils/wrappers/sb3.py>`__
- Brax: `link <https://gist.github.com/araffin/a7a576ec1453e74d9bb93120918ef7e7>`__
- EnvPool: `link <https://github.com/sail-sg/envpool/blob/main/examples/sb3_examples/ppo.py>`__


SB3 with DeepMind Control (dm_control)
--------------------------------------

If you want to use SB3 with `dm_control <https://github.com/google-deepmind/dm_control>`_, you need to use two wrappers (one from `shimmy <https://github.com/Farama-Foundation/Shimmy>`_, one pre-built one) to convert it to a Gymnasium compatible environment:

.. code-block:: python

    import shimmy
    import stable_baselines3 as sb3
    from dm_control import suite
    from gymnasium.wrappers import FlattenObservation

    # Available envs:
    # suite._DOMAINS and suite.dog.SUITE

    env = suite.load(domain_name="dog", task_name="run")
    gym_env = FlattenObservation(shimmy.DmControlCompatibilityV0(env))

    model = sb3.PPO("MlpPolicy", gym_env, verbose=1)
    model.learn(10_000, progress_bar=True)



Record a Video
--------------

Record a mp4 video (here using a random agent).

.. note::

  It requires ``ffmpeg`` or ``avconv`` to be installed on the machine.

.. code-block:: python

  import gymnasium as gym
  from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

  env_id = "CartPole-v1"
  video_folder = "logs/videos/"
  video_length = 100

  vec_env = DummyVecEnv([lambda: gym.make(env_id, render_mode="rgb_array")])

  obs = vec_env.reset()

  # Record the video starting at the first step
  vec_env = VecVideoRecorder(vec_env, video_folder,
                         record_video_trigger=lambda x: x == 0, video_length=video_length,
                         name_prefix=f"random-agent-{env_id}")

  vec_env.reset()
  for _ in range(video_length + 1):
    action = [vec_env.action_space.sample()]
    obs, _, _, _ = vec_env.step(action)
  # Save the video
  vec_env.close()


Bonus: Make a GIF of a Trained Agent
------------------------------------

.. code-block:: python

  import imageio
  import numpy as np

  from stable_baselines3 import A2C

  model = A2C("MlpPolicy", "LunarLander-v2").learn(100_000)

  images = []
  obs = model.env.reset()
  img = model.env.render(mode="rgb_array")
  for i in range(350):
      images.append(img)
      action, _ = model.predict(obs)
      obs, _, _ ,_ = model.env.step(action)
      img = model.env.render(mode="rgb_array")

  imageio.mimsave("lander_a2c.gif", [np.array(img) for i, img in enumerate(images) if i%2 == 0], fps=29)



================================================
FILE: docs/guide/export.rst
================================================
.. _export:


Exporting models
================

After training an agent, you may want to deploy/use it in another language
or framework, like `tensorflowjs <https://github.com/tensorflow/tfjs>`_.
Stable Baselines3 does not include tools to export models to other frameworks, but
this document aims to cover parts that are required for exporting along with
more detailed stories from users of Stable Baselines3.


Background
----------

In Stable Baselines3, the controller is stored inside policies which convert
observations into actions. Each learning algorithm (e.g. DQN, A2C, SAC)
contains a policy object which represents the currently learned behavior,
accessible via ``model.policy``.

Policies hold enough information to do the inference (i.e. predict actions),
so it is enough to export these policies (cf :ref:`examples <examples>`)
to do inference in another framework.

.. warning::
  When using CNN policies, the observation is normalized during pre-preprocessing.
  This pre-processing is done *inside* the policy (dividing by 255 to have values in [0, 1])


Export to ONNX
-----------------


If you are using PyTorch 2.0+ and ONNX Opset 14+, you can easily export SB3 policies using the following code:


.. warning::

  The following returns normalized actions and doesn't include the `post-processing <https://github.com/DLR-RM/stable-baselines3/blob/a9273f968eaf8c6e04302a07d803eebfca6e7e86/stable_baselines3/common/policies.py#L370-L377>`_ step that is done with continuous actions
  (clip or unscale the action to the correct space).


.. code-block:: python

  import torch as th
  from typing import Tuple

  from stable_baselines3 import PPO
  from stable_baselines3.common.policies import BasePolicy


  class OnnxableSB3Policy(th.nn.Module):
      def __init__(self, policy: BasePolicy):
          super().__init__()
          self.policy = policy

      def forward(self, observation: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
          # NOTE: Preprocessing is included, but postprocessing
          # (clipping/inscaling actions) is not,
          # If needed, you also need to transpose the images so that they are channel first
          # use deterministic=False if you want to export the stochastic policy
          # policy() returns `actions, values, log_prob` for PPO
          return self.policy(observation, deterministic=True)


  # Example: model = PPO("MlpPolicy", "Pendulum-v1")
  PPO("MlpPolicy", "Pendulum-v1").save("PathToTrainedModel")
  model = PPO.load("PathToTrainedModel.zip", device="cpu")

  onnx_policy = OnnxableSB3Policy(model.policy)

  observation_size = model.observation_space.shape
  dummy_input = th.randn(1, *observation_size)
  th.onnx.export(
      onnx_policy,
      dummy_input,
      "my_ppo_model.onnx",
      opset_version=17,
      input_names=["input"],
  )

  ##### Load and test with onnx

  import onnx
  import onnxruntime as ort
  import numpy as np

  onnx_path = "my_ppo_model.onnx"
  onnx_model = onnx.load(onnx_path)
  onnx.checker.check_model(onnx_model)

  observation = np.zeros((1, *observation_size)).astype(np.float32)
  ort_sess = ort.InferenceSession(onnx_path)
  actions, values, log_prob = ort_sess.run(None, {"input": observation})

  print(actions, values, log_prob)

  # Check that the predictions are the same
  with th.no_grad():
      print(model.policy(th.as_tensor(observation), deterministic=True))

For exporting ``MultiInputPolicy``, please have a look at `GH#1873 <https://github.com/DLR-RM/stable-baselines3/issues/1873#issuecomment-2710776085>`_.

For SAC the procedure is similar. The example shown only exports the actor network as the actor is sufficient to roll out the trained policies.

.. code-block:: python

  import torch as th

  from stable_baselines3 import SAC


  class OnnxablePolicy(th.nn.Module):
      def __init__(self, actor: th.nn.Module):
          super().__init__()
          self.actor = actor

      def forward(self, observation: th.Tensor) -> th.Tensor:
          # NOTE: You may have to postprocess (unnormalize) actions
          # to the correct bounds (see commented code below)
          return self.actor(observation, deterministic=True)


  # Example: model = SAC("MlpPolicy", "Pendulum-v1")
  SAC("MlpPolicy", "Pendulum-v1").save("PathToTrainedModel.zip")
  model = SAC.load("PathToTrainedModel.zip", device="cpu")
  onnxable_model = OnnxablePolicy(model.policy.actor)

  observation_size = model.observation_space.shape
  dummy_input = th.randn(1, *observation_size)
  th.onnx.export(
      onnxable_model,
      dummy_input,
      "my_sac_actor.onnx",
      opset_version=17,
      input_names=["input"],
  )

  ##### Load and test with onnx

  import onnxruntime as ort
  import numpy as np

  onnx_path = "my_sac_actor.onnx"

  observation = np.zeros((1, *observation_size)).astype(np.float32)
  ort_sess = ort.InferenceSession(onnx_path)
  scaled_action = ort_sess.run(None, {"input": observation})[0]

  print(scaled_action)

  # Post-process: rescale to correct space
  # Rescale the action from [-1, 1] to [low, high]
  # low, high = model.action_space.low, model.action_space.high
  # post_processed_action = low + (0.5 * (scaled_action + 1.0) * (high - low))

  # Check that the predictions are the same
  with th.no_grad():
      print(model.actor(th.as_tensor(observation), deterministic=True))


For more discussion around the topic, please refer to `GH#383 <https://github.com/DLR-RM/stable-baselines3/issues/383>`_ and `GH#1349 <https://github.com/DLR-RM/stable-baselines3/issues/1349>`_.



Trace/Export to C++
-------------------

You can use PyTorch JIT to trace and save a trained model that can be re-used in other applications
(for instance inference code written in C++).

There is a draft PR in the RL Zoo about C++ export: https://github.com/DLR-RM/rl-baselines3-zoo/pull/228

.. code-block:: python

  # See "ONNX export" for imports and OnnxablePolicy
  jit_path = "sac_traced.pt"

  # Trace and optimize the module
  traced_module = th.jit.trace(onnxable_model.eval(), dummy_input)
  frozen_module = th.jit.freeze(traced_module)
  frozen_module = th.jit.optimize_for_inference(frozen_module)
  th.jit.save(frozen_module, jit_path)

  ##### Load and test with torch

  import torch as th

  dummy_input = th.randn(1, *observation_size)
  loaded_module = th.jit.load(jit_path)
  action_jit = loaded_module(dummy_input)


Export to tensorflowjs / ONNX-JS
--------------------------------

TODO: contributors help is welcomed!
Probably a good starting point: https://github.com/elliotwaite/pytorch-to-javascript-with-onnx-js


Export to TFLite / Coral (Edge TPU)
-----------------------------------

Full example code: https://github.com/chunky/sb3_to_coral

Google created a chip called the "Coral" for deploying AI to the
edge. It's available in a variety of form factors, including USB (using
the Coral on a Raspberry Pi, with a SB3-developed model, was the original
motivation for the code example above).

The Coral chip is fast, with very low power consumption, but only has limited
on-device training abilities. More information is on the webpage here:
https://coral.ai.

To deploy to a Coral, one must work via TFLite, and quantize the
network to reflect the Coral's capabilities. The full chain to go from
SB3 to Coral is: SB3 (Torch) => ONNX => TensorFlow => TFLite => Coral.

The code linked above is a complete, minimal, example that:

1. Creates a model using SB3
2. Follows the path of exports all the way to TFLite and Google Coral
3. Demonstrates the forward pass for most exported variants

There are a number of pitfalls along the way to the complete conversion
that this example covers, including:

- Making the Gym's observation work with ONNX properly
- Quantising the TFLite model appropriately to align with Gym
  while still taking advantage of Coral
- Using OnnxablePolicy described as described in the above example


Manual export
-------------

You can also manually export required parameters (weights) and construct the
network in your desired framework.

You can access parameters of the model via agents'
:func:`get_parameters <stable_baselines3.common.base_class.BaseAlgorithm.get_parameters>` function.
As policies are also PyTorch modules, you can also access ``model.policy.state_dict()`` directly.
To find the architecture of the networks for each algorithm, best is to check the ``policies.py`` file located
in their respective folders.

.. note::

  In most cases, we recommend using PyTorch methods ``state_dict()`` and ``load_state_dict()`` from the policy,
  unless you need to access the optimizers' state dict too. In that case, you need to call ``get_parameters()``.



================================================
FILE: docs/guide/imitation.rst
================================================
.. _imitation:

Imitation Learning
==================

The `imitation <https://github.com/HumanCompatibleAI/imitation>`__ library implements
imitation learning algorithms on top of Stable-Baselines3, including:

  - Behavioral Cloning
  - `DAgger <https://arxiv.org/abs/1011.0686>`_ with synthetic examples
  - `Adversarial Inverse Reinforcement Learning <https://arxiv.org/abs/1710.11248>`_ (AIRL)
  - `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_  (GAIL)
  - `Deep RL from Human Preferences <https://arxiv.org/abs/1706.03741>`_ (DRLHP)

You can install imitation with ``pip install imitation``. The `imitation
documentation <https://imitation.readthedocs.io/en/latest/>`_ has more details
on how to use the library, including `a quick start guide
<https://imitation.readthedocs.io/en/latest/getting-started/first-steps.html>`_
for the impatient.



================================================
FILE: docs/guide/install.rst
================================================
.. _install:

Installation
============


Prerequisites
-------------

Stable-Baselines3 requires python 3.9+ and PyTorch >= 2.3

Windows
~~~~~~~

We recommend using `Anaconda <https://conda.io/docs/user-guide/install/windows.html>`_ for Windows users for easier installation of Python packages and required libraries. You need an environment with Python version 3.8 or above.

For a quick start you can move straight to installing Stable-Baselines3 in the next step.

.. note::

	Trying to create Atari environments may result to vague errors related to missing DLL files and modules. This is an
	issue with atari-py package. `See this discussion for more information <https://github.com/openai/atari-py/issues/65>`_.


Stable Release
~~~~~~~~~~~~~~
To install Stable Baselines3 with pip, execute:

.. code-block:: bash

    pip install stable-baselines3[extra]

.. note::
        Some shells such as Zsh require quotation marks around brackets, i.e. ``pip install 'stable-baselines3[extra]'`` `More information <https://stackoverflow.com/a/30539963>`_.


This includes an optional dependencies like Tensorboard, OpenCV or ``ale-py`` to train on Atari games. If you do not need those, you can use:

.. code-block:: bash

    pip install stable-baselines3


.. note::

  If you need to work with OpenCV on a machine without a X-server (for instance inside a docker image),
  you will need to install ``opencv-python-headless``, see `issue #298 <https://github.com/DLR-RM/stable-baselines3/issues/298>`_.


Bleeding-edge version
---------------------

.. code-block:: bash

	pip install git+https://github.com/DLR-RM/stable-baselines3

with extras:

.. code-block:: bash

  pip install "stable_baselines3[extra,tests,docs] @ git+https://github.com/DLR-RM/stable-baselines3"


Development version
-------------------

To contribute to Stable-Baselines3, with support for running tests and building the documentation.

.. code-block:: bash

    git clone https://github.com/DLR-RM/stable-baselines3 && cd stable-baselines3
    pip install -e .[docs,tests,extra]


Using Docker Images
-------------------

If you are looking for docker images with stable-baselines already installed in it,
we recommend using images from `RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

Otherwise, the following images contained all the dependencies for stable-baselines3 but not the stable-baselines3 package itself.
They are made for development.

Use Built Images
~~~~~~~~~~~~~~~~

GPU image (requires `nvidia-docker`_):

.. code-block:: bash

   docker pull stablebaselines/stable-baselines3

CPU only:

.. code-block:: bash

   docker pull stablebaselines/stable-baselines3-cpu

Build the Docker Images
~~~~~~~~~~~~~~~~~~~~~~~~

Build GPU image (with nvidia-docker):

.. code-block:: bash

   make docker-gpu

Build CPU image:

.. code-block:: bash

   make docker-cpu

Note: if you are using a proxy, you need to pass extra params during
build and do some `tweaks`_:

.. code-block:: bash

   --network=host --build-arg HTTP_PROXY=http://your.proxy.fr:8080/ --build-arg http_proxy=http://your.proxy.fr:8080/ --build-arg HTTPS_PROXY=https://your.proxy.fr:8080/ --build-arg https_proxy=https://your.proxy.fr:8080/

Run the images (CPU/GPU)
~~~~~~~~~~~~~~~~~~~~~~~~

Run the nvidia-docker GPU image

.. code-block:: bash

   docker run -it --runtime=nvidia --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3 bash -c 'cd /home/mamba/stable-baselines3/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./scripts/run_docker_gpu.sh pytest tests/

Run the docker CPU image

.. code-block:: bash

   docker run -it --rm --network host --ipc=host --name test --mount src="$(pwd)",target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3-cpu bash -c 'cd /home/mamba/stable-baselines3/ && pytest tests/'

Or, with the shell file:

.. code-block:: bash

   ./scripts/run_docker_cpu.sh pytest tests/

Explanation of the docker command:

-  ``docker run -it`` create an instance of an image (=container), and
   run it interactively (so ctrl+c will work)
-  ``--rm`` option means to remove the container once it exits/stops
   (otherwise, you will have to use ``docker rm``)
-  ``--network host`` don't use network isolation, this allow to use
   tensorboard/visdom on host machine
-  ``--ipc=host`` Use the host system’s IPC namespace. IPC (POSIX/SysV IPC) namespace provides
   separation of named shared memory segments, semaphores and message
   queues.
-  ``--name test`` give explicitly the name ``test`` to the container,
   otherwise it will be assigned a random name
-  ``--mount src=...`` give access of the local directory (``pwd``
   command) to the container (it will be map to ``/home/mamba/stable-baselines``), so
   all the logs created in the container in this folder will be kept
-  ``bash -c '...'`` Run command inside the docker image, here run the tests
   (``pytest tests/``)

.. _nvidia-docker: https://github.com/NVIDIA/nvidia-docker
.. _tweaks: https://stackoverflow.com/questions/23111631/cannot-download-docker-images-behind-a-proxy



================================================
FILE: docs/guide/integrations.rst
================================================
.. _integrations:

============
Integrations
============

Weights & Biases
================

Weights & Biases provides a callback for experiment tracking that allows to visualize and share results.

The full documentation is available here: https://docs.wandb.ai/guides/integrations/other/stable-baselines-3

.. code-block:: python

  import gymnasium as gym
  import wandb
  from wandb.integration.sb3 import WandbCallback

  from stable_baselines3 import PPO

  config = {
      "policy_type": "MlpPolicy",
      "total_timesteps": 25000,
      "env_id": "CartPole-v1",
  }
  run = wandb.init(
      project="sb3",
      config=config,
      sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
      # monitor_gym=True,  # auto-upload the videos of agents playing the game
      # save_code=True,  # optional
  )

  model = PPO(config["policy_type"], config["env_id"], verbose=1, tensorboard_log=f"runs/{run.id}")
  model.learn(
      total_timesteps=config["total_timesteps"],
      callback=WandbCallback(
          model_save_path=f"models/{run.id}",
          verbose=2,
      ),
  )
  run.finish()


Hugging Face 🤗
===============
The Hugging Face Hub 🤗 is a central place where anyone can share and explore models. It allows you to host your saved models 💾.

You can see the list of stable-baselines3 saved models here: https://huggingface.co/models?library=stable-baselines3
Most of them are available via the RL Zoo.

Official pre-trained models are saved in the SB3 organization on the hub: https://huggingface.co/sb3

We wrote a tutorial on how to use 🤗 Hub and Stable-Baselines3
`here <https://colab.research.google.com/github/huggingface/huggingface_sb3/blob/main/notebooks/sb3_huggingface.ipynb>`_.


Installation
-------------

.. code-block:: bash

 pip install huggingface_sb3


.. note::

 If you use the `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_, pushing/loading models from the hub are already integrated:

 .. code-block:: bash


     # Download model and save it into the logs/ folder
     # Only use TRUST_REMOTE_CODE=True with HF models that can be trusted (here the SB3 organization)
     TRUST_REMOTE_CODE=True python -m rl_zoo3.load_from_hub --algo a2c --env LunarLander-v2 -orga sb3 -f logs/
     # Test the agent
     python -m rl_zoo3.enjoy --algo a2c --env LunarLander-v2  -f logs/
     # Push model, config and hyperparameters to the hub
     python -m rl_zoo3.push_to_hub --algo a2c --env LunarLander-v2 -f logs/ -orga sb3 -m "Initial commit"



Download a model from the Hub
-----------------------------
You need to copy the repo-id that contains your saved model.
For instance ``sb3/demo-hf-CartPole-v1``:

.. code-block:: python

  import os

  import gymnasium as gym

  from huggingface_sb3 import load_from_hub
  from stable_baselines3 import PPO
  from stable_baselines3.common.evaluation import evaluate_policy


  # Allow the use of `pickle.load()` when downloading model from the hub
  # Please make sure that the organization from which you download can be trusted
  os.environ["TRUST_REMOTE_CODE"] = "True"

  # Retrieve the model from the hub
  ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
  ## filename = name of the model zip file from the repository
  checkpoint = load_from_hub(
      repo_id="sb3/demo-hf-CartPole-v1",
      filename="ppo-CartPole-v1.zip",
  )
  model = PPO.load(checkpoint)

  # Evaluate the agent and watch it
  eval_env = gym.make("CartPole-v1")
  mean_reward, std_reward = evaluate_policy(
      model, eval_env, render=True, n_eval_episodes=5, deterministic=True, warn=False
  )
  print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

You need to define two parameters:

- ``repo-id``: the name of the Hugging Face repo you want to download.
- ``filename``: the file you want to download.


Upload a model to the Hub
-------------------------

You can easily upload your models using two different functions:

1. ``package_to_hub()``: save the model, evaluate it, generate a model card and record a replay video of your agent before pushing the complete repo to the Hub.

2. ``push_to_hub()``: simply push a file to the Hub.


First, you need to be logged in to Hugging Face to upload a model:

- If you're using Colab/Jupyter Notebooks:

.. code-block:: python

 from huggingface_hub import notebook_login
 notebook_login()


- Otherwise:

.. code-block:: bash

 huggingface-cli login


Then, in this example, we train a PPO agent to play CartPole-v1 and push it to a new repo ``sb3/demo-hf-CartPole-v1``

With ``package_to_hub()``
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

  from stable_baselines3 import PPO
  from stable_baselines3.common.env_util import make_vec_env

  from huggingface_sb3 import package_to_hub

  # Create the environment
  env_id = "CartPole-v1"
  env = make_vec_env(env_id, n_envs=1)

  # Create the evaluation environment
  eval_env = make_vec_env(env_id, n_envs=1)

  # Instantiate the agent
  model = PPO("MlpPolicy", env, verbose=1)

  # Train the agent
  model.learn(total_timesteps=int(5000))

  # This method save, evaluate, generate a model card and record a replay video of your agent before pushing the repo to the hub
  package_to_hub(model=model,
               model_name="ppo-CartPole-v1",
               model_architecture="PPO",
               env_id=env_id,
               eval_env=eval_env,
               repo_id="sb3/demo-hf-CartPole-v1",
               commit_message="Test commit")

You need to define seven parameters:

- ``model``: your trained model.
- ``model_architecture``: name of the architecture of your model (DQN, PPO, A2C, SAC…).
- ``env_id``: name of the environment.
- ``eval_env``: environment used to evaluate the agent.
- ``repo-id``: the name of the Hugging Face repo you want to create or update. It’s <your huggingface username>/<the repo name>.
- ``commit-message``.
- ``filename``: the file you want to push to the Hub.

With ``push_to_hub()``
^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python


  from stable_baselines3 import PPO
  from stable_baselines3.common.env_util import make_vec_env

  from huggingface_sb3 import push_to_hub

  # Create the environment
  env_id = "CartPole-v1"
  env = make_vec_env(env_id, n_envs=1)

  # Instantiate the agent
  model = PPO("MlpPolicy", env, verbose=1)

  # Train the agent
  model.learn(total_timesteps=int(5000))

  # Save the model
  model.save("ppo-CartPole-v1")

  # Push this saved model .zip file to the hf repo
  # If this repo does not exists it will be created
  ## repo_id = id of the model repository from the Hugging Face Hub (repo_id = {organization}/{repo_name})
  ## filename: the name of the file == "name" inside model.save("ppo-CartPole-v1")
  push_to_hub(
    repo_id="sb3/demo-hf-CartPole-v1",
    filename="ppo-CartPole-v1.zip",
    commit_message="Added CartPole-v1 model trained with PPO",
  )

You need to define three parameters:

- ``repo-id``: the name of the Hugging Face repo you want to create or update. It’s <your huggingface username>/<the repo name>.
- ``filename``: the file you want to push to the Hub.
- ``commit-message``.

MLFLow
======

If you want to use `MLFLow <https://github.com/mlflow/mlflow>`_ to track your SB3 experiments,
you can adapt the following code which defines a custom logger output:

.. code-block:: python

  import sys
  from typing import Any, Dict, Tuple, Union

  import mlflow
  import numpy as np

  from stable_baselines3 import SAC
  from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger


  class MLflowOutputFormat(KVWriter):
      """
      Dumps key/value pairs into MLflow's numeric format.
      """

      def write(
          self,
          key_values: Dict[str, Any],
          key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
          step: int = 0,
      ) -> None:

          for (key, value), (_, excluded) in zip(
              sorted(key_values.items()), sorted(key_excluded.items())
          ):

              if excluded is not None and "mlflow" in excluded:
                  continue

              if isinstance(value, np.ScalarType):
                  if not isinstance(value, str):
                      mlflow.log_metric(key, value, step)


  loggers = Logger(
      folder=None,
      output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
  )

  with mlflow.start_run():
      model = SAC("MlpPolicy", "Pendulum-v1", verbose=2)
      # Set custom logger
      model.set_logger(loggers)
      model.learn(total_timesteps=10000, log_interval=1)



================================================
FILE: docs/guide/migration.rst
================================================
.. _migration:

================================
Migrating from Stable-Baselines
================================


This is a guide to migrate from Stable-Baselines (SB2) to Stable-Baselines3 (SB3).

It also references the main changes.


Overview
========

Overall Stable-Baselines3 (SB3) keeps the high-level API of Stable-Baselines (SB2).
Most of the changes are to ensure more consistency and are internal ones.
Because of the backend change, from Tensorflow to PyTorch, the internal code is much more readable and easy to debug
at the cost of some speed (dynamic graph vs static graph., see `Issue #90 <https://github.com/DLR-RM/stable-baselines3/issues/90>`_)
However, the algorithms were extensively benchmarked on Atari games and continuous control PyBullet envs
(see `Issue #48 <https://github.com/DLR-RM/stable-baselines3/issues/48>`_  and `Issue #49 <https://github.com/DLR-RM/stable-baselines3/issues/49>`_)
so you should not expect performance drop when switching from SB2 to SB3.


How to migrate?
===============

In most cases, replacing ``from stable_baselines`` by ``from stable_baselines3`` will be sufficient.
Some files were moved to the common folder (cf below) and could result to import errors.
Some algorithms were removed because of their complexity to improve the maintainability of the project.
We recommend reading this guide carefully to understand all the changes that were made.
You can also take a look at the `rl-zoo3 <https://github.com/DLR-RM/rl-baselines3-zoo>`_ and compare the imports
to the `rl-zoo <https://github.com/araffin/rl-baselines-zoo>`_ of SB2 to have a concrete example of successful migration.


.. note::

  If you experience massive slow-down switching to PyTorch, you may need to play with the number of threads used,
  using ``torch.set_num_threads(1)`` or ``OMP_NUM_THREADS=1``, see `issue #122 <https://github.com/DLR-RM/stable-baselines3/issues/122>`_
  and `issue #90 <https://github.com/DLR-RM/stable-baselines3/issues/90>`_.


Breaking Changes
================


- SB3 requires python 3.7+ (instead of python 3.5+ for SB2)
- Dropped MPI support
- Dropped layer normalized policies (``MlpLnLstmPolicy``, ``CnnLnLstmPolicy``)
- LSTM policies (```MlpLstmPolicy```, ```CnnLstmPolicy```) are not supported for the time being
  (see `PR #53 <https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53>`_ for a recurrent PPO implementation)
- Dropped parameter noise for DDPG and DQN
- PPO is now closer to the original implementation (no clipping of the value function by default), cf PPO section below
- Orthogonal initialization is only used by A2C/PPO
- The features extractor (CNN extractor) is shared between policy and q-networks for DDPG/SAC/TD3 and only the policy loss used to update it (much faster)
- Tensorboard legacy logging was dropped in favor of having one logger for the terminal and Tensorboard (cf :ref:`Tensorboard integration <tensorboard>`)
- We dropped ACKTR/ACER support because of their complexity compared to simpler alternatives (PPO, SAC, TD3) performing as good.
- We dropped GAIL support as we are focusing on model-free RL only, you can however take a look at the :ref:`imitation project <imitation>` which implements
  GAIL and other imitation learning algorithms on top of SB3.
- ``action_probability`` is currently not implemented in the base class
- ``pretrain()`` method for behavior cloning was removed (see `issue #27 <https://github.com/DLR-RM/stable-baselines3/issues/27>`_)

You can take a look at the `issue about SB3 implementation design <https://github.com/hill-a/stable-baselines/issues/576>`_ for more details.


Moved Files
-----------

- ``bench/monitor.py`` -> ``common/monitor.py``
- ``logger.py`` -> ``common/logger.py``
- ``results_plotter.py`` -> ``common/results_plotter.py``
- ``common/cmd_util.py`` -> ``common/env_util.py``

Utility functions are no longer exported from ``common`` module, you should import them with their absolute path, e.g.:

.. code-block:: python

  from stable_baselines3.common.env_util import make_atari_env, make_vec_env
  from stable_baselines3.common.utils import set_random_seed

instead of ``from stable_baselines3.common import make_atari_env``



Changes and renaming in parameters
----------------------------------

Base-class (all algorithms)
^^^^^^^^^^^^^^^^^^^^^^^^^^^

- ``load_parameters`` -> ``set_parameters``

  - ``get/set_parameters`` return a dictionary mapping object names
    to their respective PyTorch tensors and other objects representing
    their parameters, instead of simpler mapping of parameter name to
    a NumPy array. These functions also return PyTorch tensors rather
    than NumPy arrays.


Policies
^^^^^^^^

- ``cnn_extractor`` -> ``features_extractor``, as ``features_extractor`` in now used with ``MlpPolicy`` too

A2C
^^^

- ``epsilon`` -> ``rms_prop_eps``
- ``lr_schedule`` is part of ``learning_rate`` (it can be a callable).
- ``alpha``, ``momentum`` are modifiable through ``policy_kwargs`` key ``optimizer_kwargs``.

.. warning::

	PyTorch implementation of RMSprop `differs from Tensorflow's <https://github.com/pytorch/pytorch/issues/23796>`_,
	which leads to `different and potentially more unstable results <https://github.com/DLR-RM/stable-baselines3/pull/110#issuecomment-663255241>`_.
	Use ``stable_baselines3.common.sb2_compat.rmsprop_tf_like.RMSpropTFLike`` optimizer to match the results
	with TensorFlow's implementation. This can be done through ``policy_kwargs``: ``A2C(policy_kwargs=dict(optimizer_class=RMSpropTFLike, optimizer_kwargs=dict(eps=1e-5)))``


PPO
^^^

- ``cliprange`` -> ``clip_range``
- ``cliprange_vf`` -> ``clip_range_vf``
- ``nminibatches`` -> ``batch_size``

.. warning::

	``nminibatches`` gave different batch size depending on the number of environments:  ``batch_size = (n_steps * n_envs) // nminibatches``


- ``clip_range_vf`` behavior for PPO is slightly different: Set it to ``None`` (default) to deactivate clipping (in SB2, you had to pass ``-1``, ``None`` meant to use ``clip_range`` for the clipping)
- ``lam`` -> ``gae_lambda``
- ``noptepochs`` -> ``n_epochs``

PPO default hyperparameters are the one tuned for continuous control environment.
We recommend taking a look at the :ref:`RL Zoo <rl_zoo>` for hyperparameters tuned for Atari games.


DQN
^^^

Only the vanilla DQN is implemented right now but extensions will follow.
Default hyperparameters are taken from the Nature paper, except for the optimizer and learning rate that were taken from Stable Baselines defaults.

DDPG
^^^^

DDPG now follows the same interface as SAC/TD3.
For state/reward normalization, you should use ``VecNormalize`` as for all other algorithms.

SAC/TD3
^^^^^^^

SAC/TD3 now accept any number of critics, e.g. ``policy_kwargs=dict(n_critics=3)``, instead of only two before.


.. note::

	SAC/TD3 default hyperparameters (including network architecture) now match the ones from the original papers.
	DDPG is using TD3 defaults.


SAC
^^^

SAC implementation matches the latest version of the original implementation: it uses two Q function networks and two target Q function networks
instead of two Q function networks and one Value function network (SB2 implementation, first version of the original implementation).
Despite this change, no change in performance should be expected.

.. note::

	SAC ``predict()`` method has now ``deterministic=False`` by default for consistency.
	To match SB2 behavior, you need to explicitly pass ``deterministic=True``


HER
^^^

The ``HER`` implementation now only supports online sampling of the new goals. This is done in a vectorized version.
The goal selection strategy ``RANDOM`` is no longer supported.


New logger API
--------------

- Methods were renamed in the logger:

  - ``logkv`` -> ``record``, ``writekvs`` -> ``write``, ``writeseq`` ->  ``write_sequence``,
  - ``logkvs`` -> ``record_dict``, ``dumpkvs`` -> ``dump``,
  - ``getkvs`` -> ``get_log_dict``, ``logkv_mean`` -> ``record_mean``,


Internal Changes
----------------

Please read the :ref:`Developer Guide <developer>` section.


New Features (SB3 vs SB2)
=========================

- Much cleaner and consistent base code (and no more warnings =D!) and static type checks
- Independent saving/loading/predict for policies
- A2C now supports Generalized Advantage Estimation (GAE) and advantage normalization (both are deactivated by default)
- Generalized State-Dependent Exploration (gSDE) exploration is available for A2C/PPO/SAC. It allows using RL directly on real robots (cf https://arxiv.org/abs/2005.05719)
- Better saving/loading: optimizers are now included in the saved parameters and there are two new methods ``save_replay_buffer`` and ``load_replay_buffer`` for the replay buffer when using off-policy algorithms (DQN/DDPG/SAC/TD3)
- You can pass ``optimizer_class`` and ``optimizer_kwargs`` to ``policy_kwargs`` in order to easily
  customize optimizers
- Seeding now works properly to have deterministic results
- Replay buffer does not grow, allocate everything at build time (faster)
- We added a memory efficient replay buffer variant (pass ``optimize_memory_usage=True`` to the constructor), it reduces drastically the memory used especially when using images
- You can specify an arbitrary number of critics for SAC/TD3 (e.g. ``policy_kwargs=dict(n_critics=3)``)



================================================
FILE: docs/guide/quickstart.rst
================================================
.. _quickstart:

===============
Getting Started
===============

.. note::

  Stable-Baselines3 (SB3) uses :ref:`vectorized environments (VecEnv) <vec_env>` internally.
  Please read the associated section to learn more about its features and differences compared to a single Gym environment.


Most of the library tries to follow a sklearn-like syntax for the Reinforcement Learning algorithms.

Here is a quick example of how to train and run A2C on a CartPole environment:

.. code-block:: python

  import gymnasium as gym

  from stable_baselines3 import A2C

  env = gym.make("CartPole-v1", render_mode="rgb_array")

  model = A2C("MlpPolicy", env, verbose=1)
  model.learn(total_timesteps=10_000)

  vec_env = model.get_env()
  obs = vec_env.reset()
  for i in range(1000):
      action, _state = model.predict(obs, deterministic=True)
      obs, reward, done, info = vec_env.step(action)
      vec_env.render("human")
      # VecEnv resets automatically
      # if done:
      #   obs = vec_env.reset()

.. note::

	You can find explanations about the logger output and names in the :ref:`Logger <logger>` section.


Or just train a model with a one line if
`the environment is registered in Gymnasium <https://gymnasium.farama.org/tutorials/gymnasium_basics/environment_creation/#registering-envs>`_ and if
the policy is registered:

.. code-block:: python

    from stable_baselines3 import A2C

    model = A2C("MlpPolicy", "CartPole-v1").learn(10000)



================================================
FILE: docs/guide/rl.rst
================================================
.. _rl:

================================
Reinforcement Learning Resources
================================


Stable-Baselines3 assumes that you already understand the basic concepts of Reinforcement Learning (RL).

However, if you want to learn about RL, there are several good resources to get started:

- `OpenAI Spinning Up <https://spinningup.openai.com/en/latest/>`_
- `The Deep Reinforcement Learning Course <https://huggingface.co/learn/deep-rl-course/unit0/introduction>`_
- `David Silver's course <http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html>`_
- `Lilian Weng's blog <https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html>`_
- `Berkeley's Deep RL Bootcamp <https://sites.google.com/view/deep-rl-bootcamp/lectures>`_
- `Berkeley's Deep Reinforcement Learning course <http://rail.eecs.berkeley.edu/deeprlcourse/>`_
- `DQN tutorial <https://github.com/araffin/rlss23-dqn-tutorial>`_
- `Decisions & Dragons - FAQ for RL foundations <https://www.decisionsanddragons.com>`_
- `More resources <https://github.com/dennybritz/reinforcement-learning>`_



================================================
FILE: docs/guide/rl_tips.rst
================================================
.. _rl_tips:

======================================
Reinforcement Learning Tips and Tricks
======================================

The aim of this section is to help you run reinforcement learning experiments.
It covers general advice about RL (where to start, which algorithm to choose, how to evaluate an algorithm, ...),
as well as tips and tricks when using a custom environment or implementing an RL algorithm.

.. note::

  We have a `video on YouTube <https://www.youtube.com/watch?v=Ikngt0_DXJg>`_ that covers
  this section in more details. You can also find the `slides here <https://araffin.github.io/slides/rlvs-tips-tricks/>`_.


.. note::

	We also have a `video on Designing and Running Real-World RL Experiments <https://youtu.be/eZ6ZEpCi6D8>`_, slides `can be found online <https://araffin.github.io/slides/design-real-rl-experiments/>`_.


General advice when using Reinforcement Learning
================================================

TL;DR
-----

1. Read about RL and Stable Baselines3
2. Do quantitative experiments and hyperparameter tuning if needed
3. Evaluate the performance using a separate test environment (remember to check wrappers!)
4. For better performance, increase the training budget


Like any other subject, if you want to work with RL, you should first read about it (we have a dedicated `resource page <rl.html>`_ to get you started)
to understand what you are using. We also recommend you read Stable Baselines3 (SB3) documentation and do the `tutorial <https://github.com/araffin/rl-tutorial-jnrr19>`_.
It covers basic usage and guide you towards more advanced concepts of the library (e.g. callbacks and wrappers).

Reinforcement Learning differs from other machine learning methods in several ways. The data used to train the agent is collected
through interactions with the environment by the agent itself (compared to supervised learning where you have a fixed dataset for instance).
This dependence can lead to vicious circle: if the agent collects poor quality data (e.g., trajectories with no rewards), then it will not improve and continue to amass
bad trajectories.

This factor, among others, explains that results in RL may vary from one run to another (i.e., when only the seed of the pseudo-random generator changes).
For this reason, you should always do several runs to have quantitative results.

Good results in RL are generally dependent on finding appropriate hyperparameters. Recent algorithms (PPO, SAC, TD3, DroQ) normally require little hyperparameter tuning,
however, *don't expect the default ones to work* on any environment.

Therefore, we *highly recommend you* to take a look at the `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ (or the original papers) for tuned hyperparameters.
A best practice when you apply RL to a new problem is to do automatic hyperparameter optimization. Again, this is included in the `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_.

When applying RL to a custom problem, you should always normalize the input to the agent (e.g. using ``VecNormalize`` for PPO/A2C)
and look at common preprocessing done on other environments (e.g. for `Atari <https://danieltakeshi.github.io/2016/11/25/frame-skipping-and-preprocessing-for-deep-q-networks-on-atari-2600-games/>`_, frame-stack, ...).
Please refer to *Tips and Tricks when creating a custom environment* paragraph below for more advice related to custom environments.


Current Limitations of RL
-------------------------

You have to be aware of the current `limitations <https://www.alexirpan.com/2018/02/14/rl-hard.html>`_ of reinforcement learning.


Model-free RL algorithms (i.e. all the algorithms implemented in SB) are usually *sample inefficient*. They require a lot of samples (sometimes millions of interactions) to learn something useful.
That's why most of the successes in RL were achieved on games or in simulation only. For instance, in this `work <https://www.youtube.com/watch?v=aTDkYFZFWug>`_ by ETH Zurich, the ANYmal robot was trained in simulation only, and then tested in the real world.

As a general advice, to obtain better performances, you should augment the budget of the agent (number of training timesteps).


In order to achieve the desired behavior, expert knowledge is often required to design an adequate reward function.
This *reward engineering* (or *RewArt* as coined by `Freek Stulp <http://www.freekstulp.net/>`_), necessitates several iterations. As a good example of reward shaping,
you can take a look at `Deep Mimic paper <https://xbpeng.github.io/projects/DeepMimic/index.html>`_ which combines imitation learning and reinforcement learning to do acrobatic moves.

One last limitation of RL is the instability of training. That is to say, you can observe during training a huge drop in performance.
This behavior is particularly present in ``DDPG``, that's why its extension ``TD3`` tries to tackle that issue.
Other method, like ``TRPO`` or ``PPO`` make use of a *trust region* to minimize that problem by avoiding too large update.


How to evaluate an RL algorithm?
--------------------------------

.. note::

  Pay attention to environment wrappers when evaluating your agent and comparing results to others' results. Modifications to episode rewards
  or lengths may also affect evaluation results which may not be desirable. Check ``evaluate_policy`` helper function in :ref:`Evaluation Helper <eval>` section.

Because most algorithms use exploration noise during training, you need a separate test environment to evaluate the performance
of your agent at a given time. It is recommended to periodically evaluate your agent for ``n`` test episodes (``n`` is usually between 5 and 20)
and average the reward per episode to have a good estimate.

.. note::

	We provide an ``EvalCallback`` for doing such evaluation. You can read more about it in the :ref:`Callbacks <callbacks>` section.

As some policies are stochastic by default (e.g. A2C or PPO), you should also try to set `deterministic=True` when calling the `.predict()` method,
this frequently leads to better performance.
Looking at the training curve (episode reward function of the timesteps) is a good proxy but underestimates the agent true performance.


We highly recommend reading `Empirical Design in Reinforcement Learning <https://arxiv.org/abs/2304.01315>`_, as it provides valuable insights for best practices when running RL experiments.

We also suggest reading `Deep Reinforcement Learning that Matters <https://arxiv.org/abs/1709.06560>`_ for a good discussion about RL evaluation,
and `Rliable: Better Evaluation for Reinforcement Learning <https://araffin.github.io/post/rliable/>`_ for comparing results.

You can also take a look at this `blog post <https://openlab-flowers.inria.fr/t/how-many-random-seeds-should-i-use-statistical-power-analysis-in-deep-reinforcement-learning-experiments/457>`_
and this `issue <https://github.com/hill-a/stable-baselines/issues/199>`_ by Cédric Colas.


Which algorithm should I use?
=============================

There is no silver bullet in RL, you can choose one or the other depending on your needs and problems.
The first distinction comes from your action space, i.e., do you have discrete (e.g. LEFT, RIGHT, ...)
or continuous actions (ex: go to a certain speed)?

Some algorithms are only tailored for one or the other domain: ``DQN`` supports only discrete actions, while ``SAC`` is restricted to continuous actions.

The second difference that will help you decide is whether you can parallelize your training or not.
If what matters is the wall clock training time, then you should lean towards ``A2C`` and its derivatives (PPO, ...).
Take a look at the `Vectorized Environments <vec_envs.html>`_ to learn more about training with multiple workers.

To accelerate training, you can also take a look at `SBX`_, which is SB3 + Jax, it has less features than SB3 but can be up to 20x faster than SB3 PyTorch thanks to JIT compilation of the gradient update.

In sparse reward settings, we either recommend using either dedicated methods like HER (see below) or population-based algorithms like ARS (available in our :ref:`contrib repo <sb3_contrib>`).

To sum it up:

Discrete Actions
----------------

.. note::

	This covers ``Discrete``, ``MultiDiscrete``, ``Binary`` and ``MultiBinary`` spaces


Discrete Actions - Single Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``DQN`` with extensions (double DQN, prioritized replay, ...) are the recommended algorithms.
We notably provide ``QR-DQN`` in our :ref:`contrib repo <sb3_contrib>`.
``DQN`` is usually slower to train (regarding wall clock time) but is the most sample efficient (because of its replay buffer).

Discrete Actions - Multiprocessed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You should give a try to ``PPO`` or ``A2C``.


Continuous Actions
------------------

Continuous Actions - Single Process
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Current State Of The Art (SOTA) algorithms are ``SAC``, ``TD3``, ``CrossQ`` and ``TQC`` (available in our :ref:`contrib repo <sb3_contrib>` and :ref:`SBX (SB3 + Jax) repo <sbx>`).
Please use the hyperparameters in the `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ for best results.

If you want an extremely sample-efficient algorithm, we recommend using the `DroQ configuration <https://twitter.com/araffin2/status/1575439865222660098>`_ in `SBX`_ (it does many gradient steps per step in the environment).


Continuous Actions - Multiprocessed
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Take a look at ``PPO``, ``TRPO`` (available in our :ref:`contrib repo <sb3_contrib>`) or ``A2C``. Again, don't forget to take the hyperparameters from the `RL zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ for continuous actions problems (cf *Bullet* envs).

.. note::

  Normalization is critical for those algorithms



Goal Environment
-----------------

If your environment follows the ``GoalEnv`` interface (cf :ref:`HER <her>`), then you should use
HER + (SAC/TD3/DDPG/DQN/QR-DQN/TQC) depending on the action space.


.. note::

	The ``batch_size`` is an important hyperparameter for experiments with :ref:`HER <her>`



Tips and Tricks when creating a custom environment
==================================================

If you want to learn about how to create a custom environment, we recommend you read this `page <custom_env.html>`_.
We also provide a `colab notebook <https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb>`_ for a concrete example of creating a custom gym environment.

Some basic advice:

- always normalize your observation space if you can, i.e. if you know the boundaries
- normalize your action space and make it symmetric if it is continuous (see potential problem below) A good practice is to rescale your actions so that they lie in [-1, 1]. This does not limit you, as you can easily rescale the action within the environment.
- start with a shaped reward (i.e. informative reward) and a simplified version of your problem
- debug with random actions to check if your environment works and follows the gym interface (with ``check_env``, see below)

Two important things to keep in mind when creating a custom environment are avoiding breaking the Markov assumption
and properly handle termination due to a timeout (maximum number of steps in an episode).
For example, if there is a time delay between action and observation (e.g. due to wifi communication), you should provide a history of observations as input.

Termination due to timeout (max number of steps per episode) needs to be handled separately.
You should return ``truncated = True``.
If you are using the gym ``TimeLimit`` wrapper, this will be done automatically.
You can read `Time Limit in RL <https://arxiv.org/abs/1712.00378>`_, take a look at the `Designing and Running Real-World RL Experiments video <https://youtu.be/eZ6ZEpCi6D8>`_ or `RL Tips and Tricks video <https://www.youtube.com/watch?v=Ikngt0_DXJg>`_ for more details.


We provide a helper to check that your environment runs without error:

.. code-block:: python

	from stable_baselines3.common.env_checker import check_env

	env = CustomEnv(arg1, ...)
	# It will check your custom environment and output additional warnings if needed
	check_env(env)


If you want to quickly try a random agent on your environment, you can also do:

.. code-block:: python

  env = YourEnv()
  obs, info = env.reset()
  n_steps = 10
  for _ in range(n_steps):
      # Random action
      action = env.action_space.sample()
      obs, reward, terminated, truncated, info = env.step(action)
      if done:
          obs, info = env.reset()


**Why should I normalize the action space?**


Most reinforcement learning algorithms rely on a `Gaussian distribution <https://araffin.github.io/post/sac-massive-sim/>`_ (initially centered at 0 with std 1) for continuous actions.
So, if you forget to normalize the action space when using a custom environment,
this can `harm learning <https://araffin.github.io/post/sac-massive-sim/>`_ and can be difficult to debug (cf attached image and `issue #473 <https://github.com/hill-a/stable-baselines/issues/473>`_).

.. figure:: ../_static/img/mistake.png


Another consequence of using a Gaussian distribution is that the action range is not bounded.
That's why clipping is usually used as a bandage to stay in a valid interval.
A better solution would be to use a squashing function (cf ``SAC``) or a Beta distribution (cf `issue #112 <https://github.com/hill-a/stable-baselines/issues/112>`_).

.. note::

	This statement is not true for ``DDPG`` or ``TD3`` because they don't rely on any probability distribution.



Tips and Tricks when implementing an RL algorithm
=================================================

.. note::

  We have a `video on YouTube about reliable RL <https://www.youtube.com/watch?v=7-PUg9EAa3Y>`_ that covers
  this section in more details. You can also find the `slides online <https://araffin.github.io/slides/tips-reliable-rl/>`_.


When you try to reproduce a RL paper by implementing the algorithm, the `nuts and bolts of RL research <http://joschu.net/docs/nuts-and-bolts.pdf>`_
by John Schulman are quite useful (`video <https://www.youtube.com/watch?v=8EcdaCk9KaQ>`_).

We *recommend following those steps to have a working RL algorithm*:

1. Read the original paper several times
2. Read existing implementations (if available)
3. Try to have some "sign of life" on toy problems
4. Validate the implementation by making it run on harder and harder envs (you can compare results against the RL zoo).
   You usually need to run hyperparameter optimization for that step.

You need to be particularly careful on the shape of the different objects you are manipulating (a broadcast mistake will fail silently cf. `issue #75 <https://github.com/hill-a/stable-baselines/pull/76>`_)
and when to stop the gradient propagation.

Don't forget to handle termination due to timeout separately (see remark in the custom environment section above),
you can also take a look at `Issue #284 <https://github.com/DLR-RM/stable-baselines3/issues/284>`_ and `Issue #633 <https://github.com/DLR-RM/stable-baselines3/issues/633>`_.

A personal pick (by @araffin) for environments with gradual difficulty in RL with continuous actions:

1. Pendulum (easy to solve)
2. HalfCheetahBullet (medium difficulty with local minima and shaped reward)
3. BipedalWalkerHardcore (if it works on that one, then you can have a cookie)

in RL with discrete actions:

1. CartPole-v1 (easy to be better than random agent, harder to achieve maximal performance)
2. LunarLander
3. Pong (one of the easiest Atari game)
4. other Atari games (e.g. Breakout)

.. _SBX: https://github.com/araffin/sbx



================================================
FILE: docs/guide/rl_zoo.rst
================================================
.. _rl_zoo:

==================
RL Baselines3 Zoo
==================

`RL Baselines3 Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ is a training framework for Reinforcement Learning (RL).

It provides scripts for training, evaluating agents, tuning hyperparameters, plotting results and recording videos.

In addition, it includes a collection of tuned hyperparameters for common environments and RL algorithms, and agents trained with those settings.

Goals of this repository:

1. Provide a simple interface to train and enjoy RL agents
2. Benchmark the different Reinforcement Learning algorithms
3. Provide tuned hyperparameters for each environment and RL algorithm
4. Have fun with the trained agents!

Documentation is available online: https://rl-baselines3-zoo.readthedocs.io/

Installation
------------

Option 1: install the python package ``pip install rl_zoo3``

or:

1. Clone the repository:

::

  git clone --recursive https://github.com/DLR-RM/rl-baselines3-zoo
  cd rl-baselines3-zoo/


.. note::

	You can remove the ``--recursive`` option if you don't want to download the trained agents


.. note::

  If you only need the training/plotting scripts and additional callbacks/wrappers from the RL Zoo, you can also install it via pip: ``pip install rl_zoo3``


2. Install dependencies
::

   apt-get install swig cmake ffmpeg
   # full dependencies
   pip install -r requirements.txt
   # minimal dependencies
   pip install -e .


Train an Agent
--------------

The hyperparameters for each environment are defined in
``hyperparameters/algo_name.yml``.

If the environment exists in this file, then you can train an agent
using:

::

 python -m rl_zoo3.train --algo algo_name --env env_id

For example (with evaluation and checkpoints):

::

 python -m rl_zoo3.train --algo ppo --env CartPole-v1 --eval-freq 10000 --save-freq 50000


Continue training (here, load pretrained agent for Breakout and continue
training for 5000 steps):

::

 python -m rl_zoo3.train --algo a2c --env BreakoutNoFrameskip-v4 -i trained_agents/a2c/BreakoutNoFrameskip-v4_1/BreakoutNoFrameskip-v4.zip -n 5000


Enjoy a Trained Agent
---------------------

If the trained agent exists, then you can see it in action using:

::

  python -m rl_zoo3.enjoy --algo algo_name --env env_id

For example, enjoy A2C on Breakout during 5000 timesteps:

::

  python -m rl_zoo3.enjoy --algo a2c --env BreakoutNoFrameskip-v4 --folder rl-trained-agents/ -n 5000


Hyperparameter Optimization
---------------------------

We use `Optuna <https://optuna.org/>`_ for optimizing the hyperparameters.


Tune the hyperparameters for PPO, using a random sampler and median pruner, 2 parallels jobs,
with a budget of 1000 trials and a maximum of 50000 steps:

::

  python -m rl_zoo3.train --algo ppo --env MountainCar-v0 -n 50000 -optimize --n-trials 1000 --n-jobs 2 \
    --sampler random --pruner median


Colab Notebook: Try it Online!
------------------------------

You can train agents online using Google `colab notebook <https://colab.research.google.com/github/Stable-Baselines-Team/rl-colab-notebooks/blob/sb3/rl-baselines-zoo.ipynb>`_.


.. note::

	You can find more information about the rl baselines3 zoo in the repo `README <https://github.com/DLR-RM/rl-baselines3-zoo>`_. For instance, how to record a video of a trained agent.



================================================
FILE: docs/guide/save_format.rst
================================================
.. _save_format:


On saving and loading
=====================

Stable Baselines3 (SB3) stores both neural network parameters and algorithm-related parameters such as
exploration schedule, number of environments and observation/action space. This allows continual learning and easy
use of trained agents without training, but it is not without its issues. Following describes the format
used to save agents in SB3 along with its pros and shortcomings.

Terminology used in this page:

-  *parameters* refer to neural network parameters (also called "weights"). This is a dictionary
   mapping variable name to a PyTorch tensor.
-  *data* refers to RL algorithm parameters, e.g. learning rate, exploration schedule, action/observation space.
   These depend on the algorithm used. This is a dictionary mapping classes variable names to their values.


Zip-archive
-----------

A zip-archived JSON dump, PyTorch state dictionaries and PyTorch variables. The data dictionary (class parameters)
is stored as a JSON file, model parameters and optimizers are serialized with ``torch.save()`` function and these files
are stored under a single .zip archive.

Any objects that are not JSON serializable are serialized with cloudpickle and stored as base64-encoded
string in the JSON file, along with some information that was stored in the serialization. This allows
inspecting stored objects without deserializing the object itself.

This format allows skipping elements in the file, i.e. we can skip deserializing objects that are
broken/non-serializable.
This can be done via ``custom_objects`` argument to load functions.

.. note::

  If you encounter loading issue, for instance pickle issues or error after loading
  (see `#171 <https://github.com/DLR-RM/stable-baselines3/issues/171>`_ or `#573 <https://github.com/DLR-RM/stable-baselines3/issues/573>`_),
  you can pass ``print_system_info=True``
  to compare the system on which the model was trained vs the current one
  ``model = PPO.load("ppo_saved", print_system_info=True)``


File structure:

::

  saved_model.zip/
  ├── data              JSON file of class-parameters (dictionary)
  ├── *.optimizer.pth   PyTorch optimizers serialized
  ├── policy.pth        PyTorch state dictionary of the policy saved
  ├── pytorch_variables.pth Additional PyTorch variables
  ├── _stable_baselines3_version contains the SB3 version with which the model was saved
  ├── system_info.txt contains system info (os, python version, ...) on which the model was saved


Pros:

- More robust to unserializable objects (one bad object does not break everything).
- Saved files can be inspected/extracted with zip-archive explorers and by other languages.


Cons:

- More complex implementation.
- Still relies partly on cloudpickle for complex objects (e.g. custom functions)
  with can lead to `incompatibilities <https://github.com/DLR-RM/stable-baselines3/issues/172>`_ between Python versions.



================================================
FILE: docs/guide/sb3_contrib.rst
================================================
.. _sb3_contrib:

==================
SB3 Contrib
==================

We implement experimental features in a separate contrib repository:
`SB3-Contrib`_

This allows Stable-Baselines3 (SB3) to maintain a stable and compact core, while still
providing the latest features, like RecurrentPPO (PPO LSTM), Truncated Quantile Critics (TQC), Augmented Random Search (ARS), Trust Region Policy Optimization (TRPO) or
Quantile Regression DQN (QR-DQN).

Why create this repository?
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Over the span of stable-baselines and stable-baselines3, the community
has been eager to contribute in form of better logging utilities,
environment wrappers, extended support (e.g. different action spaces)
and learning algorithms.

However sometimes these utilities were too niche to be considered for
stable-baselines or proved to be too difficult to integrate well into
the existing code without creating a mess. sb3-contrib aims to fix this by not
requiring the neatest code integration with existing code and not
setting limits on what is too niche: almost everything remotely useful
goes!
We hope this allows us to provide reliable implementations
following stable-baselines usual standards (consistent style, documentation, etc)
beyond the relatively small scope of utilities in the main repository.

Features
--------

See documentation for the full list of included features.

**RL Algorithms**:

- `Augmented Random Search (ARS) <https://arxiv.org/abs/1803.07055>`_
- `Quantile Regression DQN (QR-DQN)`_
- `PPO with invalid action masking (Maskable PPO) <https://arxiv.org/abs/2006.14171>`_
- `PPO with recurrent policy (RecurrentPPO aka PPO LSTM) <https://ppo-details.cleanrl.dev//2021/11/05/ppo-implementation-details/>`_
- `Truncated Quantile Critics (TQC)`_
- `Trust Region Policy Optimization (TRPO) <https://arxiv.org/abs/1502.05477>`_
- `Batch Normalization in Deep Reinforcement Learning (CrossQ) <https://openreview.net/forum?id=PczQtTsTIX>`_


**Gym Wrappers**:

- `Time Feature Wrapper`_

Documentation
-------------

Documentation is available online: https://sb3-contrib.readthedocs.io/

Installation
------------

To install Stable-Baselines3 contrib with pip, execute:

::

   pip install sb3-contrib

We recommend to use the ``master`` version of Stable Baselines3 and SB3-Contrib.

To install Stable Baselines3 ``master`` version:

::

   pip install git+https://github.com/DLR-RM/stable-baselines3

To install Stable Baselines3 contrib ``master`` version:

::

  pip install git+https://github.com/Stable-Baselines-Team/stable-baselines3-contrib


Example
-------

SB3-Contrib follows the SB3 API and folder structure. So, if you are familiar with SB3,
using SB3-Contrib should be easy too.

Here is an example of training a Quantile Regression DQN (QR-DQN) agent on the CartPole environment.

.. code-block:: python

  from sb3_contrib import QRDQN

  policy_kwargs = dict(n_quantiles=50)
  model = QRDQN("MlpPolicy", "CartPole-v1", policy_kwargs=policy_kwargs, verbose=1)
  model.learn(total_timesteps=10000, log_interval=4)
  model.save("qrdqn_cartpole")



.. _SB3-Contrib: https://github.com/Stable-Baselines-Team/stable-baselines3-contrib
.. _Truncated Quantile Critics (TQC): https://arxiv.org/abs/2005.04269
.. _Quantile Regression DQN (QR-DQN): https://arxiv.org/abs/1710.10044
.. _Time Feature Wrapper: https://arxiv.org/abs/1712.00378



================================================
FILE: docs/guide/sbx.rst
================================================
.. _sbx:

==========================
Stable Baselines Jax (SBX)
==========================

`Stable Baselines Jax (SBX) <https://github.com/araffin/sbx>`_ is a proof of concept version of Stable-Baselines3 in Jax.

It provides a minimal number of features compared to SB3 but can be much faster (up to 20x times!): https://twitter.com/araffin2/status/1590714558628253698

Implemented algorithms:

- Soft Actor-Critic (SAC) and SAC-N
- Truncated Quantile Critics (TQC)
- Dropout Q-Functions for Doubly Efficient Reinforcement Learning (DroQ)
- Proximal Policy Optimization (PPO)
- Deep Q Network (DQN)
- Twin Delayed DDPG (TD3)
- Deep Deterministic Policy Gradient (DDPG)
- Batch Normalization in Deep Reinforcement Learning (CrossQ)
- Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning (SimBa)


As SBX follows SB3 API, it is also compatible with the `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_.
For that you will need to create two files:

``train_sbx.py``:

.. code-block:: python

  import rl_zoo3
  import rl_zoo3.train
  from rl_zoo3.train import train
  from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

  rl_zoo3.ALGOS["ddpg"] = DDPG
  rl_zoo3.ALGOS["dqn"] = DQN
  # See SBX readme to use DroQ configuration
  # rl_zoo3.ALGOS["droq"] = DroQ
  rl_zoo3.ALGOS["sac"] = SAC
  rl_zoo3.ALGOS["ppo"] = PPO
  rl_zoo3.ALGOS["td3"] = TD3
  rl_zoo3.ALGOS["tqc"] = TQC
  rl_zoo3.ALGOS["crossq"] = CrossQ
  rl_zoo3.train.ALGOS = rl_zoo3.ALGOS
  rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS


  if __name__ == "__main__":
      train()

Then you can call ``python train_sbx.py --algo sac --env Pendulum-v1`` and use the RL Zoo CLI.


``enjoy_sbx.py``:

.. code-block:: python

  import rl_zoo3
  import rl_zoo3.enjoy
  from rl_zoo3.enjoy import enjoy
  from sbx import DDPG, DQN, PPO, SAC, TD3, TQC, CrossQ

  rl_zoo3.ALGOS["ddpg"] = DDPG
  rl_zoo3.ALGOS["dqn"] = DQN
  # See SBX readme to use DroQ configuration
  # rl_zoo3.ALGOS["droq"] = DroQ
  rl_zoo3.ALGOS["sac"] = SAC
  rl_zoo3.ALGOS["ppo"] = PPO
  rl_zoo3.ALGOS["td3"] = TD3
  rl_zoo3.ALGOS["tqc"] = TQC
  rl_zoo3.ALGOS["crossq"] = CrossQ
  rl_zoo3.enjoy.ALGOS = rl_zoo3.ALGOS
  rl_zoo3.exp_manager.ALGOS = rl_zoo3.ALGOS


  if __name__ == "__main__":
      enjoy()



================================================
FILE: docs/guide/tensorboard.rst
================================================
.. _tensorboard:

Tensorboard Integration
=======================

Basic Usage
------------

To use Tensorboard with stable baselines3, you simply need to pass the location of the log folder to the RL agent:

.. code-block:: python

    from stable_baselines3 import A2C

    model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10_000)


You can also define custom logging name when training (by default it is the algorithm name)

.. code-block:: python

    from stable_baselines3 import A2C

    model = A2C("MlpPolicy", "CartPole-v1", verbose=1, tensorboard_log="./a2c_cartpole_tensorboard/")
    model.learn(total_timesteps=10_000, tb_log_name="first_run")
    # Pass reset_num_timesteps=False to continue the training curve in tensorboard
    # By default, it will create a new curve
    # Keep tb_log_name constant to have continuous curve (see note below)
    model.learn(total_timesteps=10_000, tb_log_name="second_run", reset_num_timesteps=False)
    model.learn(total_timesteps=10_000, tb_log_name="third_run", reset_num_timesteps=False)


.. note::
    If you specify different ``tb_log_name`` in subsequent runs, you will have split graphs, like in the figure below.
    If you want them to be continuous, you must keep the same ``tb_log_name`` (see `issue #975 <https://github.com/DLR-RM/stable-baselines3/issues/975#issuecomment-1198992211>`_).
    And, if you still managed to get your graphs split by other means, just put tensorboard log files into the same folder.

    .. image:: ../_static/img/split_graph.png
      :width: 330
      :alt: split_graph

Once the learn function is called, you can monitor the RL agent during or after the training, with the following bash command:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole_tensorboard/


.. note::

	You can find explanations about the logger output and names in the :ref:`Logger <logger>` section.


you can also add past logging folders:

.. code-block:: bash

  tensorboard --logdir ./a2c_cartpole_tensorboard/;./ppo2_cartpole_tensorboard/

It will display information such as the episode reward (when using a ``Monitor`` wrapper), the model losses and other parameter unique to some models.

.. image:: ../_static/img/Tensorboard_example.png
  :width: 600
  :alt: plotting

Logging More Values
-------------------

Using a callback, you can easily log more values with TensorBoard.
Here is a simple example on how to log both additional tensor or arbitrary scalar value:

.. code-block:: python

    import numpy as np

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback

    model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


    class TensorboardCallback(BaseCallback):
        """
        Custom callback for plotting additional values in tensorboard.
        """

        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self) -> bool:
            # Log scalar value (here a random variable)
            value = np.random.random()
            self.logger.record("random_value", value)
            return True


    model.learn(50000, callback=TensorboardCallback())


.. note::

  If you want to log values more often than the default to tensorboard, you manually call ``self.logger.dump(self.num_timesteps)`` in a callback
  (see `issue #506 <https://github.com/DLR-RM/stable-baselines3/issues/506>`_).


Logging Images
--------------

TensorBoard supports periodic logging of image data, which helps evaluating agents at various stages during training.

.. warning::
    To support image logging `pillow <https://github.com/python-pillow/Pillow>`_ must be installed otherwise, TensorBoard ignores the image and logs a warning.

Here is an example of how to render an image to TensorBoard at regular intervals:

.. code-block:: python

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Image

    model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


    class ImageRecorderCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self):
            image = self.training_env.render(mode="rgb_array")
            # "HWC" specify the dataformat of the image, here channel last
            # (H for height, W for width, C for channel)
            # See https://pytorch.org/docs/stable/tensorboard.html
            # for supported formats
            self.logger.record("trajectory/image", Image(image, "HWC"), exclude=("stdout", "log", "json", "csv"))
            return True


    model.learn(50000, callback=ImageRecorderCallback())

Logging Figures/Plots
---------------------
TensorBoard supports periodic logging of figures/plots created with matplotlib, which helps evaluate agents at various stages during training.

.. warning::
    To support figure logging `matplotlib <https://matplotlib.org/>`_ must be installed otherwise, TensorBoard ignores the figure and logs a warning.

Here is an example of how to store a plot in TensorBoard at regular intervals:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import Figure

    model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


    class FigureRecorderCallback(BaseCallback):
        def __init__(self, verbose=0):
            super().__init__(verbose)

        def _on_step(self):
            # Plot values (here a random variable)
            figure = plt.figure()
            figure.add_subplot().plot(np.random.random(3))
            # Close the figure after logging it
            self.logger.record("trajectory/figure", Figure(figure, close=True), exclude=("stdout", "log", "json", "csv"))
            plt.close()
            return True


    model.learn(50000, callback=FigureRecorderCallback())

Logging Videos
--------------

TensorBoard supports periodic logging of video data, which helps evaluate agents at various stages during training.

.. warning::
    To support video logging `moviepy <https://zulko.github.io/moviepy/>`_ must be installed otherwise, TensorBoard ignores the video and logs a warning.

Here is an example of how to render an episode and log the resulting video to TensorBoard at regular intervals:

.. code-block:: python

    from typing import Any, Dict

    import gymnasium as gym
    import torch as th
    import numpy as np

    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.logger import Video


    class VideoRecorderCallback(BaseCallback):
        def __init__(self, eval_env: gym.Env, render_freq: int, n_eval_episodes: int = 1, deterministic: bool = True):
            """
            Records a video of an agent's trajectory traversing ``eval_env`` and logs it to TensorBoard

            :param eval_env: A gym environment from which the trajectory is recorded
            :param render_freq: Render the agent's trajectory every eval_freq call of the callback.
            :param n_eval_episodes: Number of episodes to render
            :param deterministic: Whether to use deterministic or stochastic policy
            """
            super().__init__()
            self._eval_env = eval_env
            self._render_freq = render_freq
            self._n_eval_episodes = n_eval_episodes
            self._deterministic = deterministic

        def _on_step(self) -> bool:
            if self.n_calls % self._render_freq == 0:
                screens = []

                def grab_screens(_locals: Dict[str, Any], _globals: Dict[str, Any]) -> None:
                    """
                    Renders the environment in its current state, recording the screen in the captured `screens` list

                    :param _locals: A dictionary containing all local variables of the callback's scope
                    :param _globals: A dictionary containing all global variables of the callback's scope
                    """
                    # We expect `render()` to return a uint8 array with values in [0, 255] or a float array
                    # with values in [0, 1], as described in
                    # https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter.add_video
                    screen = self._eval_env.render(mode="rgb_array")
                    # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
                    screens.append(screen.transpose(2, 0, 1))

                evaluate_policy(
                    self.model,
                    self._eval_env,
                    callback=grab_screens,
                    n_eval_episodes=self._n_eval_episodes,
                    deterministic=self._deterministic,
                )
                self.logger.record(
                    "trajectory/video",
                    Video(th.from_numpy(np.asarray([screens])), fps=40),
                    exclude=("stdout", "log", "json", "csv"),
                )
            return True


    model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log="runs/", verbose=1)
    video_recorder = VideoRecorderCallback(gym.make("CartPole-v1"), render_freq=5000)
    model.learn(total_timesteps=int(5e4), callback=video_recorder)

Logging Hyperparameters
-----------------------

TensorBoard supports logging of hyperparameters in its HPARAMS tab, which helps to compare agents trainings.

.. warning::
    To display hyperparameters in the HPARAMS section, a ``metric_dict`` must be given (as well as a ``hparam_dict``).


Here is an example of how to save hyperparameters in TensorBoard:

.. code-block:: python

    from stable_baselines3 import A2C
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import HParam


    class HParamCallback(BaseCallback):
        """
        Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
        """

        def _on_training_start(self) -> None:
            hparam_dict = {
                "algorithm": self.model.__class__.__name__,
                "learning rate": self.model.learning_rate,
                "gamma": self.model.gamma,
            }
            # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
            # Tensorbaord will find & display metrics from the `SCALARS` tab
            metric_dict = {
                "rollout/ep_len_mean": 0,
                "train/value_loss": 0.0,
            }
            self.logger.record(
                "hparams",
                HParam(hparam_dict, metric_dict),
                exclude=("stdout", "log", "json", "csv"),
            )

        def _on_step(self) -> bool:
            return True


    model = A2C("MlpPolicy", "CartPole-v1", tensorboard_log="runs/", verbose=1)
    model.learn(total_timesteps=int(5e4), callback=HParamCallback())

Directly Accessing The Summary Writer
-------------------------------------

If you would like to log arbitrary data (in one of the formats supported by `pytorch <https://pytorch.org/docs/stable/tensorboard.html>`_), you
can get direct access to the underlying SummaryWriter in a callback:

.. warning::
    This is method is not recommended and should only be used by advanced users.

.. note::

  If you want a concrete example, you can watch `how to log lap time with donkeycar env <https://www.youtube.com/watch?v=v8j2bpcE4Rg&t=4619s>`_,
  or read the code in the `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo/blob/feat/gym-donkeycar/rl_zoo3/callbacks.py#L251-L270>`_.
  You might also want to take a look at `issue #1160 <https://github.com/DLR-RM/stable-baselines3/issues/1160>`_ and `issue #1219 <https://github.com/DLR-RM/stable-baselines3/issues/1219>`_.


.. code-block:: python

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.logger import TensorBoardOutputFormat



    model = SAC("MlpPolicy", "Pendulum-v1", tensorboard_log="/tmp/sac/", verbose=1)


    class SummaryWriterCallback(BaseCallback):

        def _on_training_start(self):
            self._log_freq = 1000  # log every 1000 calls

            output_formats = self.logger.output_formats
            # Save reference to tensorboard formatter object
            # note: the failure case (not formatter found) is not handled here, should be done with try/except.
            self.tb_formatter = next(formatter for formatter in output_formats if isinstance(formatter, TensorBoardOutputFormat))

        def _on_step(self) -> bool:
            if self.n_calls % self._log_freq == 0:
                # You can have access to info from the env using self.locals.
                # for instance, when using one env (index 0 of locals["infos"]):
                # lap_count = self.locals["infos"][0]["lap_count"]
                # self.tb_formatter.writer.add_scalar("train/lap_count", lap_count, self.num_timesteps)

                self.tb_formatter.writer.add_text("direct_access", "this is a value", self.num_timesteps)
                self.tb_formatter.writer.flush()


    model.learn(50000, callback=SummaryWriterCallback())



================================================
FILE: docs/guide/vec_envs.rst
================================================
.. _vec_env:

.. automodule:: stable_baselines3.common.vec_env

Vectorized Environments
=======================

Vectorized Environments are a method for stacking multiple independent environments into a single environment.
Instead of training an RL agent on 1 environment per step, it allows us to train it on ``n`` environments per step.
Because of this, ``actions`` passed to the environment are now a vector (of dimension ``n``).
It is the same for ``observations``, ``rewards`` and end of episode signals (``dones``).
In the case of non-array observation spaces such as ``Dict`` or ``Tuple``, where different sub-spaces
may have different shapes, the sub-observations are vectors (of dimension ``n``).

============= ======= ============ ======== ========= ================
Name          ``Box`` ``Discrete`` ``Dict`` ``Tuple`` Multi Processing
============= ======= ============ ======== ========= ================
DummyVecEnv   ✔️       ✔️           ✔️        ✔️         ❌️
SubprocVecEnv ✔️       ✔️           ✔️        ✔️         ✔️
============= ======= ============ ======== ========= ================

.. note::

	Vectorized environments are required when using wrappers for frame-stacking or normalization.

.. note::

	When using vectorized environments, the environments are automatically reset at the end of each episode.
	Thus, the observation returned for the i-th environment when ``done[i]`` is true will in fact be the first observation of the next episode, not the last observation of the episode that has just terminated.
	You can access the "real" final observation of the terminated episode—that is, the one that accompanied the ``done`` event provided by the underlying environment—using the ``terminal_observation`` keys in the info dicts returned by the ``VecEnv``.


.. warning::

  When defining a custom ``VecEnv`` (for instance, using gym3 ``ProcgenEnv``), you should provide ``terminal_observation`` keys in the info dicts returned by the ``VecEnv``
  (cf. note above).


.. warning::

    When using ``SubprocVecEnv``, users must wrap the code in an ``if __name__ == "__main__":`` if using the ``forkserver`` or ``spawn`` start method (default on Windows).
    On Linux, the default start method is ``fork`` which is not thread safe and can create deadlocks.

    For more information, see Python's `multiprocessing guidelines <https://docs.python.org/3/library/multiprocessing.html#the-spawn-and-forkserver-start-methods>`_.


VecEnv API vs Gym API
---------------------

For consistency across Stable-Baselines3 (SB3) versions and because of its special requirements and features,
SB3 VecEnv API is not the same as Gym API.
SB3 VecEnv API is actually close to Gym 0.21 API but differs to Gym 0.26+ API:

- the ``reset()`` method only returns the observation (``obs = vec_env.reset()``) and not a tuple, the info at reset are stored in ``vec_env.reset_infos``.

- only the initial call to ``vec_env.reset()`` is required, environments are reset automatically afterward (and ``reset_infos`` is updated automatically).

- the ``vec_env.step(actions)`` method expects an array as input
  (with a batch size corresponding to the number of environments) and returns a 4-tuple (and not a 5-tuple): ``obs, rewards, dones, infos`` instead of ``obs, reward, terminated, truncated, info``
  where ``dones = terminated or truncated`` (for each env).
  ``obs, rewards, dones`` are NumPy arrays with shape ``(n_envs, shape_for_single_env)`` (so with a batch dimension).
  Additional information is passed via the ``infos`` value which is a list of dictionaries.

- at the end of an episode, ``infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated``
  tells the user if an episode was truncated or not:
  you should bootstrap if ``infos[env_idx]["TimeLimit.truncated"] is True`` (episode over due to a timeout/truncation)
  or ``dones[env_idx] is False`` (episode not finished).
  Note: compared to Gym 0.26+ ``infos[env_idx]["TimeLimit.truncated"]`` and ``terminated`` `are mutually exclusive <https://github.com/openai/gym/issues/3102>`_.
  The conversion from SB3 to Gym API is

  .. code-block:: python

    # done is True at the end of an episode
    # dones[env_idx] = terminated[env_idx] or truncated[env_idx]
    # In SB3, truncated and terminated are mutually exclusive
    # infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated
    # terminated[env_idx] tells you whether you should bootstrap or not:
    # when the episode has not ended or when the termination was a timeout/truncation
    terminated[env_idx] = dones[env_idx] and not infos[env_idx]["TimeLimit.truncated"]
    should_bootstrap[env_idx] = not terminated[env_idx]


- at the end of an episode, because the environment resets automatically,
  we provide ``infos[env_idx]["terminal_observation"]`` which contains the last observation
  of an episode (and can be used when bootstrapping, see note in the previous section)

- to overcome the current Gymnasium limitation (only one render mode allowed per env instance, see `issue #100 <https://github.com/Farama-Foundation/Gymnasium/issues/100>`_),
  we recommend using ``render_mode="rgb_array"`` since we can both have the image as a numpy array and display it with OpenCV.
  if no mode is passed or ``mode="rgb_array"`` is passed when calling ``vec_env.render`` then we use the default mode, otherwise, we use the OpenCV display.
  Note that if ``render_mode != "rgb_array"``, you can only call ``vec_env.render()`` (without argument or with ``mode=env.render_mode``).

- the ``reset()`` method doesn't take any parameter. If you want to seed the pseudo-random generator or pass options,
  you should call ``vec_env.seed(seed=seed)``/``vec_env.set_options(options)`` and ``obs = vec_env.reset()`` afterward (seed and options are discarded after each call to ``reset()``).

- methods and attributes of the underlying Gym envs can be accessed, called and set using ``vec_env.get_attr("attribute_name")``,
  ``vec_env.env_method("method_name", args1, args2, kwargs1=kwargs1)`` and ``vec_env.set_attr("attribute_name", new_value)``.


Modifying Vectorized Environments Attributes
--------------------------------------------

If you plan to `modify the attributes of an environment <https://github.com/DLR-RM/stable-baselines3/issues/1573>`_ while it is used (e.g., modifying an attribute specifying the task carried out for a portion of training when doing multi-task learning, or
a parameter of the environment dynamics), you must expose a setter method.
In fact, directly accessing the environment attribute in the callback can lead to unexpected behavior because environments can be wrapped (using gym or VecEnv wrappers, the ``Monitor`` wrapper being one example).

Consider the following example for a custom env:

.. code-block:: python

	import gymnasium as gym
	from gymnasium import spaces

	from stable_baselines3.common.env_util import make_vec_env


	class MyMultiTaskEnv(gym.Env):

	  def __init__(self):
	      super().__init__()
	      """
	      A state and action space for robotic locomotion.
	      The multi-task twist is that the policy would need to adapt to different terrains, each with its own
	      friction coefficient, mu.
	      The friction coefficient is the only parameter that changes between tasks.
	      mu is a scalar between 0 and 1, and during training a callback is used to update mu.
	      """
	      ...

	  def step(self, action):
	    # Do something, depending on the action and current value of mu the next state is computed
	    return self._get_obs(), reward, done, truncated, info

	  def set_mu(self, new_mu: float) -> None:
	      # Note: this value should be used only at the next reset
	      self.mu = new_mu

	# Example of wrapped env
	# env is of type <TimeLimit<OrderEnforcing<PassiveEnvChecker<CartPoleEnv<CartPole-v1>>>>>
	env = gym.make("CartPole-v1")
	# To access the base env, without wrapper, you should use `.unwrapped`
	# or env.get_wrapper_attr("gravity") to include wrappers
	env.unwrapped.gravity
	# SB3 uses VecEnv for training, where `env.unwrapped.x = new_value` cannot be used to set an attribute
	# therefore, you should expose a setter like `set_mu` to properly set an attribute
	vec_env = make_vec_env(MyMultiTaskEnv)
	# Print current mu value
	# Note: you should use vec_env.env_method("get_wrapper_attr", "mu") in Gymnasium v1.0
	print(vec_env.env_method("get_wrapper_attr", "mu"))
	# Change `mu` attribute via the setter
	vec_env.env_method("set_mu", "mu", 0.1)
	# If the variable exists, you can also use `set_wrapper_attr` to set it
	assert vec_env.has_attr("mu")
	vec_env.env_method("set_wrapper_attr", "mu", 0.1)


In this example ``env.mu`` cannot be accessed/changed directly because it is wrapped in a ``VecEnv`` and because it could be wrapped with other wrappers (see `GH#1573 <https://github.com/DLR-RM/stable-baselines3/issues/1573>`_ for a longer explanation).
Instead, the callback should use the ``set_mu`` method via the ``env_method`` method for Vectorized Environments.

.. code-block:: python

	from itertools import cycle

	class ChangeMuCallback(BaseCallback):
	  """
	  This callback changes the value of mu during training looping
	  through a list of values until training is aborted.
	  The environment is implemented so that the impact of changing
	  the value of mu mid-episode is visible only after the episode is over
	  and the reset method has been called.
	  """"
	  def __init__(self):
	    super().__init__()
	    # An iterator that contains the different of the friction coefficient
	    self.mus = cycle([0.1, 0.2, 0.5, 0.13, 0.9])

	  def _on_step(self):
	    # Note: in practice, you should not change this value at every step
	    # but rather depending on some events/metrics like agent performance/episode termination
	    # both accessible via the `self.logger` or `self.locals` variables
	    self.training_env.env_method("set_mu", next(self.mus))

This callback can then be used to safely modify environment attributes during training since
it calls the environment setter method.


Vectorized Environments Wrappers
--------------------------------

If you want to alter or augment a ``VecEnv`` without redefining it completely (e.g. stack multiple frames, monitor the ``VecEnv``, normalize the observation, ...), you can use ``VecEnvWrapper`` for that.
They are the vectorized equivalents (i.e., they act on multiple environments at the same time) of ``gym.Wrapper``.

You can find below an example for extracting one key from the observation:

.. code-block:: python

	import numpy as np

	from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvStepReturn, VecEnvWrapper


	class VecExtractDictObs(VecEnvWrapper):
	    """
	    A vectorized wrapper for filtering a specific key from dictionary observations.
	    Similar to Gym's FilterObservation wrapper:
	        https://github.com/openai/gym/blob/master/gym/wrappers/filter_observation.py

	    :param venv: The vectorized environment
	    :param key: The key of the dictionary observation
	    """

	    def __init__(self, venv: VecEnv, key: str):
	        self.key = key
	        super().__init__(venv=venv, observation_space=venv.observation_space.spaces[self.key])

	    def reset(self) -> np.ndarray:
	        obs = self.venv.reset()
	        return obs[self.key]

	    def step_async(self, actions: np.ndarray) -> None:
	        self.venv.step_async(actions)

	    def step_wait(self) -> VecEnvStepReturn:
	        obs, reward, done, info = self.venv.step_wait()
	        return obs[self.key], reward, done, info

	env = DummyVecEnv([lambda: gym.make("FetchReach-v1")])
	# Wrap the VecEnv
	env = VecExtractDictObs(env, key="observation")


.. note::
   When creating a vectorized environment, you can also specify ordinary gymnasium
   wrappers to wrap each of the sub-environments. See the
   :func:`make_vec_env <stable_baselines3.common.env_util.make_vec_env>`
   documentation for details.
   Example:

   .. code-block:: python

    from gymnasium.wrappers import RescaleAction
    from stable_baselines3.common.env_util import make_vec_env

    # Use gym wrapper for each sub-env of the VecEnv
    wrapper_kwargs = dict(min_action=-1.0, max_action=1.0)
    vec_env = make_vec_env(
        "Pendulum-v1", n_envs=2, wrapper_class=RescaleAction, wrapper_kwargs=wrapper_kwargs
    )



VecEnv
------

.. autoclass:: VecEnv
  :members:

DummyVecEnv
-----------

.. autoclass:: DummyVecEnv
  :members:

SubprocVecEnv
-------------

.. autoclass:: SubprocVecEnv
  :members:

Wrappers
--------

VecFrameStack
~~~~~~~~~~~~~

.. autoclass:: VecFrameStack
  :members:

StackedObservations
~~~~~~~~~~~~~~~~~~~

.. autoclass:: stable_baselines3.common.vec_env.stacked_observations.StackedObservations
  :members:

VecNormalize
~~~~~~~~~~~~

.. autoclass:: VecNormalize
  :members:


VecVideoRecorder
~~~~~~~~~~~~~~~~

.. autoclass:: VecVideoRecorder
  :members:


VecCheckNan
~~~~~~~~~~~~~~~~

.. autoclass:: VecCheckNan
  :members:


VecTransposeImage
~~~~~~~~~~~~~~~~~

.. autoclass:: VecTransposeImage
  :members:

VecMonitor
~~~~~~~~~~~~~~~~~

.. autoclass:: VecMonitor
  :members:

VecExtractDictObs
~~~~~~~~~~~~~~~~~

.. autoclass:: VecExtractDictObs
  :members:



================================================
FILE: docs/misc/changelog.rst
================================================
.. _changelog:

Changelog
==========

Release 2.6.1a0 (WIP)
--------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^
- Fixed docker GPU image (PyTorch GPU was not installed)

`SB3-Contrib`_
^^^^^^^^^^^^^^

`RL Zoo`_
^^^^^^^^^

`SBX`_ (SB3 + Jax)
^^^^^^^^^^^^^^^^^^

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^

Documentation:
^^^^^^^^^^^^^^
- Clarify ``evaluate_policy`` documentation


Release 2.6.0 (2025-03-24)
--------------------------

**New ``LogEveryNTimesteps`` callback and ``has_attr`` method, refactored hyperparameter optimization**

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Added ``has_attr`` method for ``VecEnv`` to check if an attribute exists
- Added ``LogEveryNTimesteps`` callback to dump logs every N timesteps (note: you need to pass ``log_interval=None`` to avoid any interference)
- Added Gymnasium v1.1 support

Bug Fixes:
^^^^^^^^^^
- `SubProcVecEnv` will now exit gracefully (without big traceback) when using `KeyboardInterrupt`

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Renamed ``_dump_logs()`` to ``dump_logs()``
- Fixed issues with ``SubprocVecEnv`` and ``MaskablePPO`` by using ``vec_env.has_attr()`` (pickling issues, mask function not present)

`RL Zoo`_
^^^^^^^^^
- Refactored hyperparameter optimization. The Optuna `Journal storage backend <https://optuna.readthedocs.io/en/stable/reference/generated/optuna.storages.JournalStorage.html>`__ is now supported (recommended default) and you can easily load tuned hyperparameter via the new ``--trial-id`` argument of ``train.py``.
- Save the exact command line used to launch a training
- Added support for special vectorized env (e.g. Brax, IsaacSim) by allowing to override the ``VecEnv`` class use to instantiate the env in the ``ExperimentManager``
- Allow to disable auto-logging by passing ``--log-interval -2`` (useful when logging things manually)
- Added Gymnasium v1.1 support
- Fixed use of old HF api in ``get_hf_trained_models()``

`SBX`_ (SB3 + Jax)
^^^^^^^^^^^^^^^^^^
- Updated PPO to support ``net_arch``, and additional fixes
- Fixed entropy coeff wrongly logged for SAC and derivatives.
- Fixed PPO ``predict()`` for env that were not normalized (action spaces with limits != [-1, 1])
- PPO now logs the standard deviation

Deprecations:
^^^^^^^^^^^^^
- ``algo._dump_logs()`` is deprecated in favor of ``algo.dump_logs()`` and will be removed in SB3 v2.7.0

Others:
^^^^^^^
- Updated black from v24 to v25
- Improved error messages when checking Box space equality (loading ``VecNormalize``)
- Updated test to reflect how ``set_wrapper_attr`` should be used now

Documentation:
^^^^^^^^^^^^^^
- Clarify the use of Gym wrappers with ``make_vec_env`` in the section on Vectorized Environments (@pstahlhofen)
- Updated callback doc for ``EveryNTimesteps``
- Added doc on how to set env attributes via ``VecEnv`` calls
- Added ONNX export example for ``MultiInputPolicy`` (@darkopetrovic)


Release 2.5.0 (2025-01-27)
--------------------------

**New algorithm: SimBa in SBX, NumPy 2.0 support**


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Increased minimum required version of PyTorch to 2.3.0
- Removed support for Python 3.8

New Features:
^^^^^^^^^^^^^
- Added support for NumPy v2.0: ``VecNormalize`` now cast normalized rewards to float32, updated bit flipping env to avoid overflow issues too
- Added official support for Python 3.12

`SBX`_ (SB3 + Jax)
^^^^^^^^^^^^^^^^^^
- Added SimBa Policy: Simplicity Bias for Scaling Up Parameters in DRL
- Added support for parameter resets

Others:
^^^^^^^
- Updated Dockerfile

Documentation:
^^^^^^^^^^^^^^
- Added Decisions and Dragons to resources. (@jmacglashan)
- Updated PyBullet example, now compatible with Gymnasium
- Added link to policies for ``policy_kwargs`` parameter (@kplers)
- Add FootstepNet Envs to the project page (@cgaspard3333)
- Added FRASA to the project page (@MarcDcls)
- Fixed atari example (@chrisgao99)
- Add a note about ``Discrete`` action spaces with ``start!=0``
- Update doc for massively parallel simulators (Isaac Lab, Brax, ...)
- Add dm_control example

Release 2.4.1 (2024-12-20)
--------------------------

Bug Fixes:
^^^^^^^^^^
- Fixed a bug introduced in v2.4.0 where the ``VecVideoRecorder`` would override videos


Release 2.4.0 (2024-11-18)
--------------------------

**New algorithm: CrossQ in SB3 Contrib, Gymnasium v1.0 support**

.. note::

  DQN (and QR-DQN) models saved with SB3 < 2.4.0 will show a warning about
  truncation of optimizer state when loaded with SB3 >= 2.4.0.
  To suppress the warning, simply save the model again.
  You can find more info in `PR #1963 <https://github.com/DLR-RM/stable-baselines3/pull/1963>`_

.. warning::

    Stable-Baselines3 (SB3) v2.4.0 will be the last one supporting Python 3.8 (end of life in October 2024)
    and PyTorch < 2.3.
    We highly recommended you to upgrade to Python >= 3.9 and PyTorch >= 2.3 (compatible with NumPy v2).


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Increased minimum required version of Gymnasium to 0.29.1

New Features:
^^^^^^^^^^^^^
- Added support for ``pre_linear_modules`` and ``post_linear_modules`` in ``create_mlp`` (useful for adding normalization layers, like in DroQ or CrossQ)
- Enabled np.ndarray logging for TensorBoardOutputFormat as histogram (see GH#1634) (@iwishwasaneagle)
- Updated env checker to warn users when using multi-dim array to define `MultiDiscrete` spaces
- Added support for Gymnasium v1.0

Bug Fixes:
^^^^^^^^^^
- Fixed memory leak when loading learner from storage, ``set_parameters()`` does not try to load the object data anymore
  and only loads the PyTorch parameters (@peteole)
- Cast type in compute gae method to avoid error when using torch compile (@amjames)
- ``CallbackList`` now sets the ``.parent`` attribute of child callbacks to its own ``.parent``. (will-maclean)
- Fixed error when loading a model that has ``net_arch`` manually set to ``None``   (@jak3122)
- Set requirement numpy<2.0 until PyTorch is compatible (https://github.com/pytorch/pytorch/issues/107302)
- Updated DQN optimizer input to only include q_network parameters, removing the target_q_network ones (@corentinlger)
- Fixed ``test_buffers.py::test_device`` which was not actually checking the device of tensors (@rhaps0dy)


`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added ``CrossQ`` algorithm, from "Batch Normalization in Deep Reinforcement Learning" paper (@danielpalen)
- Added ``BatchRenorm`` PyTorch layer used in ``CrossQ`` (@danielpalen)
- Updated QR-DQN optimizer input to only include quantile_net parameters (@corentinlger)
- Fixed loading QRDQN changes `target_update_interval` (@jak3122)

`RL Zoo`_
^^^^^^^^^
- Updated defaults hyperparameters for TQC/SAC for Swimmer-v4 (decrease gamma for more consistent results)

`SBX`_ (SB3 + Jax)
^^^^^^^^^^^^^^^^^^
- Added CNN support for DQN
- Bug fix for SAC and related algorithms, optimize log of ent coeff to be consistent with SB3

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed various typos (@cschindlbeck)
- Remove unnecessary SDE noise resampling in PPO update (@brn-dev)
- Updated PyTorch version on CI to 2.3.1
- Added a warning to recommend using CPU with on policy algorithms (A2C/PPO) and ``MlpPolicy``
- Switched to uv to download packages faster on GitHub CI
- Updated dependencies for read the doc
- Removed unnecessary ``copy_obs_dict`` method for ``SubprocVecEnv``, remove the use of ordered dict and rename ``flatten_obs`` to ``stack_obs``

Documentation:
^^^^^^^^^^^^^^
- Updated PPO doc to recommend using CPU with ``MlpPolicy``
- Clarified documentation about planned features and citing software
- Added a note about the fact we are optimizing log of ent coeff for SAC

Release 2.3.2 (2024-04-27)
--------------------------

Bug Fixes:
^^^^^^^^^^
- Reverted ``torch.load()`` to be called ``weights_only=False`` as it caused loading issue with old version of PyTorch.


Documentation:
^^^^^^^^^^^^^^
- Added ER-MRL to the project page (@corentinlger)
- Updated Tensorboard Logging Videos documentation (@NickLucche)


Release 2.3.1 (2024-04-22)
--------------------------

Bug Fixes:
^^^^^^^^^^
- Cast return value of learning rate schedule to float, to avoid issue when loading model because of ``weights_only=True`` (@markscsmith)

Documentation:
^^^^^^^^^^^^^^
- Updated SBX documentation (CrossQ and deprecated DroQ)
- Updated RL Tips and Tricks section

Release 2.3.0 (2024-03-31)
--------------------------

**New defaults hyperparameters for DDPG, TD3 and DQN**

.. warning::

  Because of ``weights_only=True``, this release breaks loading of policies when using PyTorch 1.13.
  Please upgrade to PyTorch >= 2.0 or upgrade SB3 version (we reverted the change in SB3 2.3.2)


Breaking Changes:
^^^^^^^^^^^^^^^^^
- The defaults hyperparameters of ``TD3`` and ``DDPG`` have been changed to be more consistent with ``SAC``

.. code-block:: python

  # SB3 < 2.3.0 default hyperparameters
  # model = TD3("MlpPolicy", env, train_freq=(1, "episode"), gradient_steps=-1, batch_size=100)
  # SB3 >= 2.3.0:
  model = TD3("MlpPolicy", env, train_freq=1, gradient_steps=1, batch_size=256)

.. note::

	Two inconsistencies remain: the default network architecture for ``TD3/DDPG`` is ``[400, 300]`` instead of ``[256, 256]`` for SAC (for backward compatibility reasons, see `report on the influence of the network size <https://wandb.ai/openrlbenchmark/sbx/reports/SBX-TD3-Influence-of-policy-net--Vmlldzo2NDg1Mzk3>`_) and the default learning rate is 1e-3 instead of 3e-4 for SAC (for performance reasons, see `W&B report on the influence of the lr <https://wandb.ai/openrlbenchmark/sbx/reports/SBX-TD3-RL-Zoo-v2-3-0a0-vs-SB3-TD3-RL-Zoo-2-2-1---Vmlldzo2MjUyNTQx>`_)



- The default ``learning_starts`` parameter of ``DQN`` have been changed to be consistent with the other offpolicy algorithms


.. code-block:: python

  # SB3 < 2.3.0 default hyperparameters, 50_000 corresponded to Atari defaults hyperparameters
  # model = DQN("MlpPolicy", env, learning_starts=50_000)
  # SB3 >= 2.3.0:
  model = DQN("MlpPolicy", env, learning_starts=100)

- For safety, ``torch.load()`` is now called with ``weights_only=True`` when loading torch tensors,
  policy ``load()`` still uses ``weights_only=False`` as gymnasium imports are required for it to work
- When using ``huggingface_sb3``, you will now need to set ``TRUST_REMOTE_CODE=True`` when downloading models from the hub, as ``pickle.load`` is not safe.


New Features:
^^^^^^^^^^^^^
- Log success rate ``rollout/success_rate`` when available for on policy algorithms (@corentinlger)

Bug Fixes:
^^^^^^^^^^
- Fixed ``monitor_wrapper`` argument that was not passed to the parent class, and dones argument that wasn't passed to ``_update_into_buffer`` (@corentinlger)

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added ``rollout_buffer_class`` and ``rollout_buffer_kwargs`` arguments to MaskablePPO
- Fixed ``train_freq`` type annotation for tqc and qrdqn (@Armandpl)
- Fixed ``sb3_contrib/common/maskable/*.py`` type annotations
- Fixed ``sb3_contrib/ppo_mask/ppo_mask.py`` type annotations
- Fixed ``sb3_contrib/common/vec_env/async_eval.py`` type annotations
- Add some additional notes about ``MaskablePPO`` (evaluation and multi-process) (@icheered)


`RL Zoo`_
^^^^^^^^^
- Updated defaults hyperparameters for TD3/DDPG to be more consistent with SAC
- Upgraded MuJoCo envs hyperparameters to v4 (pre-trained agents need to be updated)
- Added test dependencies to `setup.py` (@power-edge)
- Simplify dependencies of `requirements.txt` (remove duplicates from `setup.py`)

`SBX`_ (SB3 + Jax)
^^^^^^^^^^^^^^^^^^
- Added support for ``MultiDiscrete`` and ``MultiBinary`` action spaces to PPO
- Added support for large values for gradient_steps to SAC, TD3, and TQC
- Fix  ``train()`` signature and update type hints
- Fix replay buffer device at load time
- Added flatten layer
- Added ``CrossQ``

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Updated black from v23 to v24
- Updated ruff to >= v0.3.1
- Updated env checker for (multi)discrete spaces with non-zero start.

Documentation:
^^^^^^^^^^^^^^
- Added a paragraph on modifying vectorized environment parameters via setters (@fracapuano)
- Updated callback code example
- Updated export to ONNX documentation, it is now much simpler to export SB3 models with newer ONNX Opset!
- Added video link to "Practical Tips for Reliable Reinforcement Learning" video
- Added ``render_mode="human"`` in the README example (@marekm4)
- Fixed docstring signature for sum_independent_dims (@stagoverflow)
- Updated docstring description for ``log_interval`` in the base class (@rushitnshah).

Release 2.2.1 (2023-11-17)
--------------------------
**Support for options at reset, bug fixes and better error messages**

.. note::

  SB3 v2.2.0 was yanked after a breaking change was found in `GH#1751 <https://github.com/DLR-RM/stable-baselines3/issues/1751>`_.
  Please use SB3 v2.2.1 and not v2.2.0.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched to ``ruff`` for sorting imports (isort is no longer needed), black and ruff version now require a minimum version
- Dropped ``x is False`` in favor of ``not x``, which means that callbacks that wrongly returned None (instead of a boolean) will cause the training to stop (@iwishiwasaneagle)

New Features:
^^^^^^^^^^^^^
- Improved error message of the ``env_checker`` for env wrongly detected as GoalEnv (``compute_reward()`` is defined)
- Improved error message when mixing Gym API with VecEnv API (see GH#1694)
- Add support for setting ``options`` at reset with VecEnv via the ``set_options()`` method. Same as seeds logic, options are reset at the end of an episode (@ReHoss)
- Added ``rollout_buffer_class`` and ``rollout_buffer_kwargs`` arguments to on-policy algorithms (A2C and PPO)


Bug Fixes:
^^^^^^^^^^
- Prevents using squash_output and not use_sde in ActorCritcPolicy (@PatrickHelm)
- Performs unscaling of actions in collect_rollout in OnPolicyAlgorithm (@PatrickHelm)
- Moves VectorizedActionNoise into ``_setup_learn()`` in OffPolicyAlgorithm (@PatrickHelm)
- Prevents out of bound error on Windows if no seed is passed (@PatrickHelm)
- Calls ``callback.update_locals()`` before ``callback.on_rollout_end()`` in OnPolicyAlgorithm (@PatrickHelm)
- Fixed replay buffer device after loading in OffPolicyAlgorithm (@PatrickHelm)
- Fixed ``render_mode`` which was not properly loaded when using ``VecNormalize.load()``
- Fixed success reward dtype in ``SimpleMultiObsEnv`` (@NixGD)
- Fixed check_env for Sequence observation space (@corentinlger)
- Prevents instantiating BitFlippingEnv with conflicting observation spaces (@kylesayrs)
- Fixed ResourceWarning when loading and saving models (files were not closed), please note that only path are closed automatically,
  the behavior stay the same for tempfiles (they need to be closed manually),
  the behavior is now consistent when loading/saving replay buffer

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added ``set_options`` for ``AsyncEval``
- Added ``rollout_buffer_class`` and ``rollout_buffer_kwargs`` arguments to TRPO

`RL Zoo`_
^^^^^^^^^
- Removed `gym` dependency, the package is still required for some pretrained agents.
- Added `--eval-env-kwargs` to `train.py` (@Quentin18)
- Added `ppo_lstm` to hyperparams_opt.py (@technocrat13)
- Upgraded to `pybullet_envs_gymnasium>=0.4.0`
- Removed old hacks (for instance limiting offpolicy algorithms to one env at test time)
- Updated docker image, removed support for X server
- Replaced deprecated `optuna.suggest_uniform(...)` by `optuna.suggest_float(..., low=..., high=...)`

`SBX`_ (SB3 + Jax)
^^^^^^^^^^^^^^^^^^
- Added ``DDPG`` and ``TD3`` algorithms

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``stable_baselines3/common/callbacks.py`` type hints
- Fixed ``stable_baselines3/common/utils.py`` type hints
- Fixed ``stable_baselines3/common/vec_envs/vec_transpose.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_video_recorder.py`` type hints
- Fixed ``stable_baselines3/common/save_util.py`` type hints
- Updated docker images to  Ubuntu Jammy using micromamba 1.5
- Fixed ``stable_baselines3/common/buffers.py`` type hints
- Fixed ``stable_baselines3/her/her_replay_buffer.py`` type hints
- Buffers do no call an additional ``.copy()`` when storing new transitions
- Fixed ``ActorCriticPolicy.extract_features()`` signature by adding an optional ``features_extractor`` argument
- Update dependencies (accept newer Shimmy/Sphinx version and remove ``sphinx_autodoc_typehints``)
- Fixed ``stable_baselines3/common/off_policy_algorithm.py`` type hints
- Fixed ``stable_baselines3/common/distributions.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_normalize.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/__init__.py`` type hints
- Switched to PyTorch 2.1.0 in the CI (fixes type annotations)
- Fixed ``stable_baselines3/common/policies.py`` type hints
- Switched to ``mypy`` only for checking types
- Added tests to check consistency when saving/loading files

Documentation:
^^^^^^^^^^^^^^
- Updated RL Tips and Tricks (include recommendation for evaluation, added links to DroQ, ARS and SBX).
- Fixed various typos and grammar mistakes
- Added PokemonRedExperiments to the project page
- Fixed an out-of-date command for installing Atari in examples

Release 2.1.0 (2023-08-17)
--------------------------

**Float64 actions , Gymnasium 0.29 support and bug fixes**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed Python 3.7 support
- SB3 now requires PyTorch >= 1.13

New Features:
^^^^^^^^^^^^^
- Added Python 3.11 support
- Added Gymnasium 0.29 support (@pseudo-rnd-thoughts)

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed MaskablePPO ignoring ``stats_window_size`` argument
- Added Python 3.11 support

`RL Zoo`_
^^^^^^^^^
- Upgraded to Huggingface-SB3 >= 2.3
- Added Python 3.11 support


Bug Fixes:
^^^^^^^^^^
- Relaxed check in logger, that was causing issue on Windows with colorama
- Fixed off-policy algorithms with continuous float64 actions (see #1145) (@tobirohrer)
- Fixed ``env_checker.py`` warning messages for out of bounds in complex observation spaces (@Gabo-Tor)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Updated GitHub issue templates
- Fix typo in gym patch error message (@lukashass)
- Refactor ``test_spaces.py`` tests

Documentation:
^^^^^^^^^^^^^^
- Fixed callback example (@BertrandDecoster)
- Fixed policy network example (@kyle-he)
- Added mobile-env as new community project (@stefanbschneider)
- Added [DeepNetSlice](https://github.com/AlexPasqua/DeepNetSlice) to community projects (@AlexPasqua)


Release 2.0.0 (2023-06-22)
--------------------------

**Gymnasium support**

.. warning::

  Stable-Baselines3 (SB3) v2.0 will be the last one supporting python 3.7 (end of life in June 2023).
  We highly recommended you to upgrade to Python >= 3.8.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched to Gymnasium as primary backend, Gym 0.21 and 0.26 are still supported via the ``shimmy`` package (@carlosluis, @arjun-kg, @tlpss)
- The deprecated ``online_sampling`` argument of ``HerReplayBuffer`` was removed
- Removed deprecated ``stack_observation_space`` method of ``StackedObservations``
- Renamed environment output observations in ``evaluate_policy`` to prevent shadowing the input observations during callbacks (@npit)
- Upgraded wrappers and custom environment to Gymnasium
- Refined the ``HumanOutputFormat`` file check: now it verifies if the object is an instance of ``io.TextIOBase`` instead of only checking for the presence of a ``write`` method.
- Because of new Gym API (0.26+), the random seed passed to ``vec_env.seed(seed=seed)`` will only be effective after then ``env.reset()`` call.

New Features:
^^^^^^^^^^^^^
- Added Gymnasium support (Gym 0.21 and 0.26 are supported via the ``shimmy`` package)

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed QRDQN update interval for multi envs


`RL Zoo`_
^^^^^^^^^
- Gym 0.26+ patches to continue working with pybullet and TimeLimit wrapper
- Renamed `CarRacing-v1` to `CarRacing-v2` in hyperparameters
- Huggingface push to hub now accepts a `--n-timesteps` argument to adjust the length of the video
- Fixed `record_video` steps (before it was stepping in a closed env)
- Dropped Gym 0.21 support

Bug Fixes:
^^^^^^^^^^
- Fixed ``VecExtractDictObs`` does not handle terminal observation (@WeberSamuel)
- Set NumPy version to ``>=1.20`` due to use of ``numpy.typing`` (@troiganto)
- Fixed loading DQN changes ``target_update_interval`` (@tobirohrer)
- Fixed env checker to properly reset the env before calling ``step()`` when checking
  for ``Inf`` and ``NaN`` (@lutogniew)
- Fixed HER ``truncate_last_trajectory()`` (@lbergmann1)
- Fixed HER desired and achieved goal order in reward computation (@JonathanKuelz)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``stable_baselines3/a2c/*.py`` type hints
- Fixed ``stable_baselines3/ppo/*.py`` type hints
- Fixed ``stable_baselines3/sac/*.py`` type hints
- Fixed ``stable_baselines3/td3/*.py`` type hints
- Fixed ``stable_baselines3/common/base_class.py`` type hints
- Fixed ``stable_baselines3/common/logger.py`` type hints
- Fixed ``stable_baselines3/common/envs/*.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_monitor|vec_extract_dict_obs|util.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/base_vec_env.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_frame_stack.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/dummy_vec_env.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/subproc_vec_env.py`` type hints
- Upgraded docker images to use mamba/micromamba and CUDA 11.7
- Updated env checker to reflect what subset of Gymnasium is supported and improve GoalEnv checks
- Improve type annotation of wrappers
- Tests envs are now checked too
- Added render test for ``VecEnv`` and ``VecEnvWrapper``
- Update issue templates and env info saved with the model
- Changed ``seed()`` method return type from ``List`` to ``Sequence``
- Updated env checker doc and requirements for tuple spaces/goal envs

Documentation:
^^^^^^^^^^^^^^
- Added Deep RL Course link to the Deep RL Resources page
- Added documentation about ``VecEnv`` API vs Gym API
- Upgraded tutorials to Gymnasium API
- Make it more explicit when using ``VecEnv`` vs Gym env
- Added UAV_Navigation_DRL_AirSim to the project page (@heleidsn)
- Added ``EvalCallback`` example (@sidney-tio)
- Update custom env documentation
- Added `pink-noise-rl` to projects page
- Fix custom policy example, ``ortho_init`` was ignored
- Added SBX page


Release 1.8.0 (2023-04-07)
--------------------------

**Multi-env HerReplayBuffer, Open RL Benchmark, Improved env checker**

.. warning::

  Stable-Baselines3 (SB3) v1.8.0 will be the last one to use Gym as a backend.
  Starting with v2.0.0, Gymnasium will be the default backend (though SB3 will have compatibility layers for Gym envs).
  You can find a migration guide here: https://gymnasium.farama.org/content/migration-guide/.
  If you want to try the SB3 v2.0 alpha version, you can take a look at `PR #1327 <https://github.com/DLR-RM/stable-baselines3/pull/1327>`_.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed shared layers in ``mlp_extractor`` (@AlexPasqua)
- Refactored ``StackedObservations`` (it now handles dict obs, ``StackedDictObservations`` was removed)
- You must now explicitly pass a ``features_extractor`` parameter when calling ``extract_features()``
- Dropped offline sampling for ``HerReplayBuffer``
- As ``HerReplayBuffer`` was refactored to support multiprocessing, previous replay buffer are incompatible with this new version
- ``HerReplayBuffer`` doesn't require a ``max_episode_length`` anymore

New Features:
^^^^^^^^^^^^^
- Added ``repeat_action_probability`` argument in ``AtariWrapper``.
- Only use ``NoopResetEnv`` and ``MaxAndSkipEnv`` when needed in ``AtariWrapper``
- Added support for dict/tuple observations spaces for ``VecCheckNan``, the check is now active in the ``env_checker()`` (@DavyMorgan)
- Added multiprocessing support for ``HerReplayBuffer``
- ``HerReplayBuffer`` now supports all datatypes supported by ``ReplayBuffer``
- Provide more helpful failure messages when validating the ``observation_space`` of custom gym environments using ``check_env`` (@FieteO)
- Added ``stats_window_size`` argument to control smoothing in rollout logging (@jonasreiher)


`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added warning about potential crashes caused by ``check_env`` in the ``MaskablePPO`` docs (@AlexPasqua)
- Fixed ``sb3_contrib/qrdqn/*.py`` type hints
- Removed shared layers in ``mlp_extractor`` (@AlexPasqua)

`RL Zoo`_
^^^^^^^^^
- `Open RL Benchmark <https://github.com/openrlbenchmark/openrlbenchmark/issues/7>`_
- Upgraded to new `HerReplayBuffer` implementation that supports multiple envs
- Removed `TimeFeatureWrapper` for Panda and Fetch envs, as the new replay buffer should handle timeout.
- Tuned hyperparameters for RecurrentPPO on Swimmer
- Documentation is now built using Sphinx and hosted on read the doc
- Removed `use_auth_token` for push to hub util
- Reverted from v3 to v2 for HumanoidStandup, Reacher, InvertedPendulum and InvertedDoublePendulum since they were not part of the mujoco refactoring (see https://github.com/openai/gym/pull/1304)
- Fixed `gym-minigrid` policy (from `MlpPolicy` to `MultiInputPolicy`)
- Replaced deprecated `optuna.suggest_loguniform(...)` by `optuna.suggest_float(..., log=True)`
- Switched to `ruff` and `pyproject.toml`
- Removed `online_sampling` and `max_episode_length` argument when using `HerReplayBuffer`

Bug Fixes:
^^^^^^^^^^
- Fixed Atari wrapper that missed the reset condition (@luizapozzobon)
- Added the argument ``dtype`` (default to ``float32``) to the noise for consistency with gym action (@sidney-tio)
- Fixed PPO train/n_updates metric not accounting for early stopping (@adamfrly)
- Fixed loading of normalized image-based environments
- Fixed ``DictRolloutBuffer.add`` with multidimensional action space (@younik)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``tests/test_tensorboard.py`` type hint
- Fixed ``tests/test_vec_normalize.py`` type hint
- Fixed ``stable_baselines3/common/monitor.py`` type hint
- Added tests for StackedObservations
- Removed Gitlab CI file
- Moved from ``setup.cg`` to ``pyproject.toml`` configuration file
- Switched from ``flake8`` to ``ruff``
- Upgraded AutoROM to latest version
- Fixed ``stable_baselines3/dqn/*.py`` type hints
- Added ``extra_no_roms`` option for package installation without Atari Roms

Documentation:
^^^^^^^^^^^^^^
- Renamed ``load_parameters`` to ``set_parameters`` (@DavyMorgan)
- Clarified documentation about subproc multiprocessing for A2C (@Bonifatius94)
- Fixed typo in ``A2C`` docstring (@AlexPasqua)
- Renamed timesteps to episodes for ``log_interval`` description (@theSquaredError)
- Removed note about gif creation for Atari games (@harveybellini)
- Added information about default network architecture
- Update information about Gymnasium support

Release 1.7.0 (2023-01-10)
--------------------------

.. warning::

  Shared layers in MLP policy (``mlp_extractor``) are now deprecated for PPO, A2C and TRPO.
  This feature will be removed in SB3 v1.8.0 and the behavior of ``net_arch=[64, 64]``
  will create **separate** networks with the same architecture, to be consistent with the off-policy algorithms.


.. note::

  A2C and PPO saved with SB3 < 1.7.0 will show a warning about
  missing keys in the state dict when loaded with SB3 >= 1.7.0.
  To suppress the warning, simply save the model again.
  You can find more info in `issue #1233 <https://github.com/DLR-RM/stable-baselines3/issues/1233>`_


Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed deprecated ``create_eval_env``, ``eval_env``, ``eval_log_path``, ``n_eval_episodes`` and ``eval_freq`` parameters,
  please use an ``EvalCallback`` instead
- Removed deprecated ``sde_net_arch`` parameter
- Removed ``ret`` attributes in ``VecNormalize``, please use ``returns`` instead
- ``VecNormalize`` now updates the observation space when normalizing images

New Features:
^^^^^^^^^^^^^
- Introduced mypy type checking
- Added option to have non-shared features extractor between actor and critic in on-policy algorithms (@AlexPasqua)
- Added ``with_bias`` argument to ``create_mlp``
- Added support for multidimensional ``spaces.MultiBinary`` observations
- Features extractors now properly support unnormalized image-like observations (3D tensor)
  when passing ``normalize_images=False``
- Added ``normalized_image`` parameter to ``NatureCNN`` and ``CombinedExtractor``
- Added support for Python 3.10

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed a bug in ``RecurrentPPO`` where the lstm states where incorrectly reshaped for ``n_lstm_layers > 1`` (thanks @kolbytn)
- Fixed ``RuntimeError: rnn: hx is not contiguous`` while predicting terminal values for ``RecurrentPPO`` when ``n_lstm_layers > 1``

`RL Zoo`_
^^^^^^^^^
- Added support for python file for configuration
- Added ``monitor_kwargs`` parameter

Bug Fixes:
^^^^^^^^^^
- Fixed ``ProgressBarCallback`` under-reporting (@dominicgkerr)
- Fixed return type of ``evaluate_actions`` in ``ActorCritcPolicy`` to reflect that entropy is an optional tensor (@Rocamonde)
- Fixed type annotation of ``policy`` in ``BaseAlgorithm`` and ``OffPolicyAlgorithm``
- Allowed model trained with Python 3.7 to be loaded with Python 3.8+ without the ``custom_objects`` workaround
- Raise an error when the same gym environment instance is passed as separate environments when creating a vectorized environment with more than one environment. (@Rocamonde)
- Fix type annotation of ``model`` in ``evaluate_policy``
- Fixed ``Self`` return type using ``TypeVar``
- Fixed the env checker, the key was not passed when checking images from Dict observation space
- Fixed ``normalize_images`` which was not passed to parent class in some cases
- Fixed ``load_from_vector`` that was broken with newer PyTorch version when passing PyTorch tensor

Deprecations:
^^^^^^^^^^^^^
- You should now explicitly pass a ``features_extractor`` parameter when calling ``extract_features()``
- Deprecated shared layers in ``MlpExtractor`` (@AlexPasqua)

Others:
^^^^^^^
- Used issue forms instead of issue templates
- Updated the PR template to associate each PR with its peer in RL-Zoo3 and SB3-Contrib
- Fixed flake8 config to be compatible with flake8 6+
- Goal-conditioned environments are now characterized by the availability of the ``compute_reward`` method, rather than by their inheritance to ``gym.GoalEnv``
- Replaced ``CartPole-v0`` by ``CartPole-v1`` is tests
- Fixed ``tests/test_distributions.py`` type hints
- Fixed ``stable_baselines3/common/type_aliases.py`` type hints
- Fixed ``stable_baselines3/common/torch_layers.py`` type hints
- Fixed ``stable_baselines3/common/env_util.py`` type hints
- Fixed ``stable_baselines3/common/preprocessing.py`` type hints
- Fixed ``stable_baselines3/common/atari_wrappers.py`` type hints
- Fixed ``stable_baselines3/common/vec_env/vec_check_nan.py`` type hints
- Exposed modules in ``__init__.py`` with the ``__all__`` attribute (@ZikangXiong)
- Upgraded GitHub CI/setup-python to v4 and checkout to v3
- Set tensors construction directly on the device (~8% speed boost on GPU)
- Monkey-patched ``np.bool = bool`` so gym 0.21 is compatible with NumPy 1.24+
- Standardized the use of ``from gym import spaces``
- Modified ``get_system_info`` to avoid issue linked to copy-pasting on GitHub issue

Documentation:
^^^^^^^^^^^^^^
- Updated Hugging Face Integration page (@simoninithomas)
- Changed ``env`` to ``vec_env`` when environment is vectorized
- Updated custom policy docs to better explain the ``mlp_extractor``'s dimensions (@AlexPasqua)
- Updated custom policy documentation (@athatheo)
- Improved tensorboard callback doc
- Clarify doc when using image-like input
- Added RLeXplore to the project page (@yuanmingqi)


Release 1.6.2 (2022-10-10)
--------------------------

**Progress bar in the learn() method, RL Zoo3 is now a package**

Breaking Changes:
^^^^^^^^^^^^^^^^^

New Features:
^^^^^^^^^^^^^
- Added ``progress_bar`` argument in the ``learn()`` method, displayed using TQDM and rich packages
- Added progress bar callback
- The `RL Zoo <https://github.com/DLR-RM/rl-baselines3-zoo>`_ can now be installed as a package (``pip install rl_zoo3``)

`SB3-Contrib`_
^^^^^^^^^^^^^^

`RL Zoo`_
^^^^^^^^^
- RL Zoo is now a python package and can be installed using ``pip install rl_zoo3``

Bug Fixes:
^^^^^^^^^^
- ``self.num_timesteps`` was initialized properly only after the first call to ``on_step()`` for callbacks
- Set importlib-metadata version to ``~=4.13`` to be compatible with ``gym=0.21``

Deprecations:
^^^^^^^^^^^^^
- Added deprecation warning if parameters ``eval_env``, ``eval_freq`` or ``create_eval_env`` are used (see #925) (@tobirohrer)

Others:
^^^^^^^
- Fixed type hint of the ``env_id`` parameter in ``make_vec_env`` and ``make_atari_env`` (@AlexPasqua)

Documentation:
^^^^^^^^^^^^^^
- Extended docstring of the ``wrapper_class`` parameter in ``make_vec_env`` (@AlexPasqua)

Release 1.6.1 (2022-09-29)
---------------------------

**Bug fix release**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched minimum tensorboard version to 2.9.1

New Features:
^^^^^^^^^^^^^
- Support logging hyperparameters to tensorboard (@timothe-chaumont)
- Added checkpoints for replay buffer and ``VecNormalize`` statistics (@anand-bala)
- Added option for ``Monitor`` to append to existing file instead of overriding (@sidney-tio)
- The env checker now raises an error when using dict observation spaces and observation keys don't match observation space keys

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Fixed the issue of wrongly passing policy arguments when using ``CnnLstmPolicy`` or ``MultiInputLstmPolicy`` with ``RecurrentPPO`` (@mlodel)

Bug Fixes:
^^^^^^^^^^
- Fixed issue where ``PPO`` gives NaN if rollout buffer provides a batch of size 1 (@hughperkins)
- Fixed the issue that ``predict`` does not always return action as ``np.ndarray`` (@qgallouedec)
- Fixed division by zero error when computing FPS when a small number of time has elapsed in operating systems with low-precision timers.
- Added multidimensional action space support (@qgallouedec)
- Fixed missing verbose parameter passing in the ``EvalCallback`` constructor (@burakdmb)
- Fixed the issue that when updating the target network in DQN, SAC, TD3, the ``running_mean`` and ``running_var`` properties of batch norm layers are not updated (@honglu2875)
- Fixed incorrect type annotation of the replay_buffer_class argument in ``common.OffPolicyAlgorithm`` initializer, where an instance instead of a class was required (@Rocamonde)
- Fixed loading saved model with different number of environments
- Removed ``forward()`` abstract method declaration from ``common.policies.BaseModel`` (already defined in ``torch.nn.Module``) to fix type errors in subclasses (@Rocamonde)
- Fixed the return type of ``.load()`` and ``.learn()`` methods in ``BaseAlgorithm`` so that they now use ``TypeVar`` (@Rocamonde)
- Fixed an issue where keys with different tags but the same key raised an error in ``common.logger.HumanOutputFormat`` (@Rocamonde and @AdamGleave)
- Set importlib-metadata version to `~=4.13`

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed ``DictReplayBuffer.next_observations`` typing (@qgallouedec)
- Added support for ``device="auto"`` in buffers and made it default (@qgallouedec)
- Updated ``ResultsWriter`` (used internally by ``Monitor`` wrapper) to automatically create missing directories when ``filename`` is a path (@dominicgkerr)

Documentation:
^^^^^^^^^^^^^^
- Added an example of callback that logs hyperparameters to tensorboard. (@timothe-chaumont)
- Fixed typo in docstring "nature" -> "Nature" (@Melanol)
- Added info on split tensorboard logs into (@Melanol)
- Fixed typo in ppo doc (@francescoluciano)
- Fixed typo in install doc(@jlp-ue)
- Clarified and standardized verbosity documentation
- Added link to a GitHub issue in the custom policy documentation (@AlexPasqua)
- Update doc on exporting models (fixes and added torch jit)
- Fixed typos (@Akhilez)
- Standardized the use of ``"`` for string representation in documentation

Release 1.6.0 (2022-07-11)
---------------------------

**Recurrent PPO (PPO LSTM), better defaults for learning from pixels with SAC/TD3**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Changed the way policy "aliases" are handled ("MlpPolicy", "CnnPolicy", ...), removing the former
  ``register_policy`` helper, ``policy_base`` parameter and using ``policy_aliases`` static attributes instead (@Gregwar)
- SB3 now requires PyTorch >= 1.11
- Changed the default network architecture when using ``CnnPolicy`` or ``MultiInputPolicy`` with SAC or DDPG/TD3,
  ``share_features_extractor`` is now set to False by default and the ``net_arch=[256, 256]`` (instead of ``net_arch=[]`` that was before)

New Features:
^^^^^^^^^^^^^


`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added Recurrent PPO (PPO LSTM). See https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53


Bug Fixes:
^^^^^^^^^^
- Fixed saving and loading large policies greater than 2GB (@jkterry1, @ycheng517)
- Fixed final goal selection strategy that did not sample the final achieved goal (@qgallouedec)
- Fixed a bug with special characters in the tensorboard log name (@quantitative-technologies)
- Fixed a bug in ``DummyVecEnv``'s and ``SubprocVecEnv``'s seeding function. None value was unchecked (@ScheiklP)
- Fixed a bug where ``EvalCallback`` would crash when trying to synchronize ``VecNormalize`` stats when observation normalization was disabled
- Added a check for unbounded actions
- Fixed issues due to newer version of protobuf (tensorboard) and sphinx
- Fix exception causes all over the codebase (@cool-RR)
- Prohibit simultaneous use of optimize_memory_usage and handle_timeout_termination due to a bug (@MWeltevrede)
- Fixed a bug in ``kl_divergence`` check that would fail when using numpy arrays with MultiCategorical distribution

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Upgraded to Python 3.7+ syntax using ``pyupgrade``
- Removed redundant double-check for nested observations from ``BaseAlgorithm._wrap_env`` (@TibiGG)

Documentation:
^^^^^^^^^^^^^^
- Added link to gym doc and gym env checker
- Fix typo in PPO doc (@bcollazo)
- Added link to PPO ICLR blog post
- Added remark about breaking Markov assumption and timeout handling
- Added doc about MLFlow integration via custom logger (@git-thor)
- Updated Huggingface integration doc
- Added copy button for code snippets
- Added doc about EnvPool and Isaac Gym support


Release 1.5.0 (2022-03-25)
---------------------------

**Bug fixes, early stopping callback**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Switched minimum Gym version to 0.21.0

New Features:
^^^^^^^^^^^^^
- Added ``StopTrainingOnNoModelImprovement`` to callback collection (@caburu)
- Makes the length of keys and values in ``HumanOutputFormat`` configurable,
  depending on desired maximum width of output.
- Allow PPO to turn of advantage normalization (see `PR #763 <https://github.com/DLR-RM/stable-baselines3/pull/763>`_) @vwxyzjn

`SB3-Contrib`_
^^^^^^^^^^^^^^
- coming soon: Cross Entropy Method, see https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/62

Bug Fixes:
^^^^^^^^^^
- Fixed a bug in ``VecMonitor``. The monitor did not consider the ``info_keywords`` during stepping (@ScheiklP)
- Fixed a bug in ``HumanOutputFormat``. Distinct keys truncated to the same prefix would overwrite each others value,
  resulting in only one being output. This now raises an error (this should only affect a small fraction of use cases
  with very long keys.)
- Routing all the ``nn.Module`` calls through implicit rather than explicit forward as per pytorch guidelines (@manuel-delverme)
- Fixed a bug in ``VecNormalize`` where error occurs when ``norm_obs`` is set to False for environment with dictionary observation  (@buoyancy99)
- Set default ``env`` argument to ``None`` in ``HerReplayBuffer.sample`` (@qgallouedec)
- Fix ``batch_size`` typing in ``DQN`` (@qgallouedec)
- Fixed sample normalization in ``DictReplayBuffer`` (@qgallouedec)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Fixed pytest warnings
- Removed parameter ``remove_time_limit_termination`` in off policy algorithms since it was dead code (@Gregwar)

Documentation:
^^^^^^^^^^^^^^
- Added doc on Hugging Face integration (@simoninithomas)
- Added furuta pendulum project to project list (@armandpl)
- Fix indentation 2 spaces to 4 spaces in custom env documentation example (@Gautam-J)
- Update MlpExtractor docstring (@gianlucadecola)
- Added explanation of the logger output
- Update ``Directly Accessing The Summary Writer`` in tensorboard integration (@xy9485)

Release 1.4.0 (2022-01-18)
---------------------------

*TRPO, ARS and multi env training for off-policy algorithms*

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Dropped python 3.6 support (as announced in previous release)
- Renamed ``mask`` argument of the ``predict()`` method to ``episode_start`` (used with RNN policies only)
- local variables ``action``, ``done`` and ``reward`` were renamed to their plural form for offpolicy algorithms (``actions``, ``dones``, ``rewards``),
  this may affect custom callbacks.
- Removed ``episode_reward`` field from ``RolloutReturn()`` type


.. warning::

    An update to the ``HER`` algorithm is planned to support multi-env training and remove the max episode length constrain.
    (see `PR #704 <https://github.com/DLR-RM/stable-baselines3/pull/704>`_)
    This will be a backward incompatible change (model trained with previous version of ``HER`` won't work with the new version).



New Features:
^^^^^^^^^^^^^
- Added ``norm_obs_keys`` param for ``VecNormalize`` wrapper to configure which observation keys to normalize (@kachayev)
- Added experimental support to train off-policy algorithms with multiple envs (note: ``HerReplayBuffer`` currently not supported)
- Handle timeout termination properly for on-policy algorithms (when using ``TimeLimit``)
- Added ``skip`` option to ``VecTransposeImage`` to skip transforming the channel order when the heuristic is wrong
- Added ``copy()`` and ``combine()`` methods to ``RunningMeanStd``

`SB3-Contrib`_
^^^^^^^^^^^^^^
- Added Trust Region Policy Optimization (TRPO) (@cyprienc)
- Added Augmented Random Search (ARS) (@sgillen)
- Coming soon: PPO LSTM, see https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/pull/53

Bug Fixes:
^^^^^^^^^^
- Fixed a bug where ``set_env()`` with ``VecNormalize`` would result in an error with off-policy algorithms (thanks @cleversonahum)
- FPS calculation is now performed based on number of steps performed during last ``learn`` call, even when ``reset_num_timesteps`` is set to ``False`` (@kachayev)
- Fixed evaluation script for recurrent policies (experimental feature in SB3 contrib)
- Fixed a bug where the observation would be incorrectly detected as non-vectorized instead of throwing an error
- The env checker now properly checks and warns about potential issues for continuous action spaces when the boundaries are too small or when the dtype is not float32
- Fixed a bug in ``VecFrameStack`` with channel first image envs, where the terminal observation would be wrongly created.

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Added a warning in the env checker when not using ``np.float32`` for continuous actions
- Improved test coverage and error message when checking shape of observation
- Added ``newline="\n"`` when opening CSV monitor files so that each line ends with ``\r\n`` instead of ``\r\r\n`` on Windows while Linux environments are not affected (@hsuehch)
- Fixed ``device`` argument inconsistency (@qgallouedec)

Documentation:
^^^^^^^^^^^^^^
- Add drivergym to projects page (@theDebugger811)
- Add highway-env to projects page (@eleurent)
- Add tactile-gym to projects page (@ac-93)
- Fix indentation in the RL tips page (@cove9988)
- Update GAE computation docstring
- Add documentation on exporting to TFLite/Coral
- Added JMLR paper and updated citation
- Added link to RL Tips and Tricks video
- Updated ``BaseAlgorithm.load`` docstring (@Demetrio92)
- Added a note on ``load`` behavior in the examples (@Demetrio92)
- Updated SB3 Contrib doc
- Fixed A2C and migration guide guidance on how to set epsilon with RMSpropTFLike (@thomasgubler)
- Fixed custom policy documentation (@IperGiove)
- Added doc on Weights & Biases integration

Release 1.3.0 (2021-10-23)
---------------------------

*Bug fixes and improvements for the user*

.. warning::

  This version will be the last one supporting Python 3.6 (end of life in Dec 2021).
  We highly recommended you to upgrade to Python >= 3.7.


Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``sde_net_arch`` argument in policies is deprecated and will be removed in a future version.
- ``_get_latent`` (``ActorCriticPolicy``) was removed
- All logging keys now use underscores instead of spaces (@timokau). Concretely this changes:

    - ``time/total timesteps`` to ``time/total_timesteps`` for off-policy algorithms (PPO and A2C) and the eval callback (on-policy algorithms already used the underscored version),
    - ``rollout/exploration rate`` to ``rollout/exploration_rate`` and
    - ``rollout/success rate`` to ``rollout/success_rate``.


New Features:
^^^^^^^^^^^^^
- Added methods ``get_distribution`` and ``predict_values`` for ``ActorCriticPolicy`` for A2C/PPO/TRPO (@cyprienc)
- Added methods ``forward_actor`` and ``forward_critic`` for ``MlpExtractor``
- Added ``sb3.get_system_info()`` helper function to gather version information relevant to SB3 (e.g., Python and PyTorch version)
- Saved models now store system information where agent was trained, and load functions have ``print_system_info`` parameter to help debugging load issues

Bug Fixes:
^^^^^^^^^^
- Fixed ``dtype`` of observations for ``SimpleMultiObsEnv``
- Allow `VecNormalize` to wrap discrete-observation environments to normalize reward
  when observation normalization is disabled
- Fixed a bug where ``DQN`` would throw an error when using ``Discrete`` observation and stochastic actions
- Fixed a bug where sub-classed observation spaces could not be used
- Added ``force_reset`` argument to ``load()`` and ``set_env()`` in order to be able to call ``learn(reset_num_timesteps=False)`` with a new environment

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Cap gym max version to 0.19 to avoid issues with atari-py and other breaking changes
- Improved error message when using dict observation with the wrong policy
- Improved error message when using ``EvalCallback`` with two envs not wrapped the same way.
- Added additional infos about supported python version for PyPi in ``setup.py``

Documentation:
^^^^^^^^^^^^^^
- Add Rocket League Gym to list of supported projects (@AechPro)
- Added gym-electric-motor to project page (@wkirgsn)
- Added policy-distillation-baselines to project page (@CUN-bjy)
- Added ONNX export instructions (@batu)
- Update read the doc env (fixed ``docutils`` issue)
- Fix PPO environment name (@IljaAvadiev)
- Fix custom env doc and add env registration example
- Update algorithms from SB3 Contrib
- Use underscores for numeric literals in examples to improve clarity

Release 1.2.0 (2021-09-03)
---------------------------

**Hotfix for VecNormalize, training/eval mode support**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- SB3 now requires PyTorch >= 1.8.1
- ``VecNormalize`` ``ret`` attribute was renamed to ``returns``

New Features:
^^^^^^^^^^^^^

Bug Fixes:
^^^^^^^^^^
- Hotfix for ``VecNormalize`` where the observation filter was not updated at reset (thanks @vwxyzjn)
- Fixed model predictions when using batch normalization and dropout layers by calling ``train()`` and ``eval()`` (@davidblom603)
- Fixed model training for DQN, TD3 and SAC so that their target nets always remain in evaluation mode (@ayeright)
- Passing ``gradient_steps=0`` to an off-policy algorithm will result in no gradient steps being taken (vs as many gradient steps as steps done in the environment
  during the rollout in previous versions)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Enabled Python 3.9 in GitHub CI
- Fixed type annotations
- Refactored ``predict()`` by moving the preprocessing to ``obs_to_tensor()`` method

Documentation:
^^^^^^^^^^^^^^
- Updated multiprocessing example
- Added example of ``VecEnvWrapper``
- Added a note about logging to tensorboard more often
- Added warning about simplicity of examples and link to RL zoo (@MihaiAnca13)


Release 1.1.0 (2021-07-01)
---------------------------

**Dict observation support, timeout handling and refactored HER buffer**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- All customs environments (e.g. the ``BitFlippingEnv`` or ``IdentityEnv``) were moved to ``stable_baselines3.common.envs`` folder
- Refactored ``HER`` which is now the ``HerReplayBuffer`` class that can be passed to any off-policy algorithm
- Handle timeout termination properly for off-policy algorithms (when using ``TimeLimit``)
- Renamed ``_last_dones`` and ``dones`` to ``_last_episode_starts`` and ``episode_starts`` in ``RolloutBuffer``.
- Removed ``ObsDictWrapper`` as ``Dict`` observation spaces are now supported

.. code-block:: python

  her_kwargs = dict(n_sampled_goal=2, goal_selection_strategy="future", online_sampling=True)
  # SB3 < 1.1.0
  # model = HER("MlpPolicy", env, model_class=SAC, **her_kwargs)
  # SB3 >= 1.1.0:
  model = SAC("MultiInputPolicy", env, replay_buffer_class=HerReplayBuffer, replay_buffer_kwargs=her_kwargs)

- Updated the KL Divergence estimator in the PPO algorithm to be positive definite and have lower variance (@09tangriro)
- Updated the KL Divergence check in the PPO algorithm to be before the gradient update step rather than after end of epoch (@09tangriro)
- Removed parameter ``channels_last`` from ``is_image_space`` as it can be inferred.
- The logger object is now an attribute ``model.logger`` that be set by the user using ``model.set_logger()``
- Changed the signature of ``logger.configure`` and ``utils.configure_logger``, they now return a ``Logger`` object
- Removed ``Logger.CURRENT`` and ``Logger.DEFAULT``
- Moved ``warn(), debug(), log(), info(), dump()`` methods to the ``Logger`` class
- ``.learn()`` now throws an import error when the user tries to log to tensorboard but the package is not installed

New Features:
^^^^^^^^^^^^^
- Added support for single-level ``Dict`` observation space (@JadenTravnik)
- Added ``DictRolloutBuffer`` ``DictReplayBuffer`` to support dictionary observations (@JadenTravnik)
- Added ``StackedObservations`` and ``StackedDictObservations`` that are used within ``VecFrameStack``
- Added simple 4x4 room Dict test environments
- ``HerReplayBuffer`` now supports ``VecNormalize`` when ``online_sampling=False``
- Added `VecMonitor <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_monitor.py>`_ and
  `VecExtractDictObs <https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/vec_env/vec_extract_dict_obs.py>`_ wrappers
  to handle gym3-style vectorized environments (@vwxyzjn)
- Ignored the terminal observation if the it is not provided by the environment
  such as the gym3-style vectorized environments. (@vwxyzjn)
- Added policy_base as input to the OnPolicyAlgorithm for more flexibility (@09tangriro)
- Added support for image observation when using ``HER``
- Added ``replay_buffer_class`` and ``replay_buffer_kwargs`` arguments to off-policy algorithms
- Added ``kl_divergence`` helper for ``Distribution`` classes (@09tangriro)
- Added support for vector environments with ``num_envs > 1`` (@benblack769)
- Added ``wrapper_kwargs`` argument to ``make_vec_env`` (@amy12xx)

Bug Fixes:
^^^^^^^^^^
- Fixed potential issue when calling off-policy algorithms with default arguments multiple times (the size of the replay buffer would be the same)
- Fixed loading of ``ent_coef`` for ``SAC`` and ``TQC``, it was not optimized anymore (thanks @Atlis)
- Fixed saving of ``A2C`` and ``PPO`` policy when using gSDE (thanks @liusida)
- Fixed a bug where no output would be shown even if ``verbose>=1`` after passing ``verbose=0`` once
- Fixed observation buffers dtype in DictReplayBuffer (@c-rizz)
- Fixed EvalCallback tensorboard logs being logged with the incorrect timestep. They are now written with the timestep at which they were recorded. (@skandermoalla)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Added ``flake8-bugbear`` to tests dependencies to find likely bugs
- Updated ``env_checker`` to reflect support of dict observation spaces
- Added Code of Conduct
- Added tests for GAE and lambda return computation
- Updated distribution entropy test (thanks @09tangriro)
- Added sanity check ``batch_size > 1`` in PPO to avoid NaN in advantage normalization

Documentation:
^^^^^^^^^^^^^^
- Added gym pybullet drones project (@JacopoPan)
- Added link to SuperSuit in projects (@justinkterry)
- Fixed DQN example (thanks @ltbd78)
- Clarified channel-first/channel-last recommendation
- Update sphinx environment installation instructions (@tom-doerr)
- Clarified pip installation in Zsh (@tom-doerr)
- Clarified return computation for on-policy algorithms (TD(lambda) estimate was used)
- Added example for using ``ProcgenEnv``
- Added note about advanced custom policy example for off-policy algorithms
- Fixed DQN unicode checkmarks
- Updated migration guide (@juancroldan)
- Pinned ``docutils==0.16`` to avoid issue with rtd theme
- Clarified callback ``save_freq`` definition
- Added doc on how to pass a custom logger
- Remove recurrent policies from ``A2C`` docs (@bstee615)


Release 1.0 (2021-03-15)
------------------------

**First Major Version**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- Removed ``stable_baselines3.common.cmd_util`` (already deprecated), please use ``env_util`` instead

.. warning::

    A refactoring of the ``HER`` algorithm is planned together with support for dictionary observations
    (see `PR #243 <https://github.com/DLR-RM/stable-baselines3/pull/243>`_ and `#351 <https://github.com/DLR-RM/stable-baselines3/pull/351>`_)
    This will be a backward incompatible change (model trained with previous version of ``HER`` won't work with the new version).


New Features:
^^^^^^^^^^^^^
- Added support for ``custom_objects`` when loading models



Bug Fixes:
^^^^^^^^^^
- Fixed a bug with ``DQN`` predict method when using ``deterministic=False`` with image space

Documentation:
^^^^^^^^^^^^^^
- Fixed examples
- Added new project using SB3: rl_reach (@PierreExeter)
- Added note about slow-down when switching to PyTorch
- Add a note on continual learning and resetting environment

Others:
^^^^^^^
- Updated RL-Zoo to reflect the fact that is it more than a collection of trained agents
- Added images to illustrate the training loop and custom policies (created with https://excalidraw.com/)
- Updated the custom policy section


Pre-Release 0.11.1 (2021-02-27)
-------------------------------

Bug Fixes:
^^^^^^^^^^
- Fixed a bug where ``train_freq`` was not properly converted when loading a saved model



Pre-Release 0.11.0 (2021-02-27)
-------------------------------

Breaking Changes:
^^^^^^^^^^^^^^^^^
- ``evaluate_policy`` now returns rewards/episode lengths from a ``Monitor`` wrapper if one is present,
  this allows to return the unnormalized reward in the case of Atari games for instance.
- Renamed ``common.vec_env.is_wrapped`` to ``common.vec_env.is_vecenv_wrapped`` to avoid confusion
  with the new ``is_wrapped()`` helper
- Renamed ``_get_data()`` to ``_get_constructor_parameters()`` for policies (this affects independent saving/loading of policies)
- Removed ``n_episodes_rollout`` and merged it with ``train_freq``, which now accepts a tuple ``(frequency, unit)``:
- ``replay_buffer`` in ``collect_rollout`` is no more optional

.. code-block:: python

  # SB3 < 0.11.0
  # model = SAC("MlpPolicy", env, n_episodes_rollout=1, train_freq=-1)
  # SB3 >= 0.11.0:
  model = SAC("MlpPolicy", env, train_freq=(1, "episode"))



New Features:
^^^^^^^^^^^^^
- Add support for ``VecFrameStack`` to stack on first or last observation dimension, along with
  automatic check for image spaces.
- ``VecFrameStack`` now has a ``channels_order`` argument to tell if observations should be stacked
  on the first or last observation dimension (originally always stacked on last).
- Added ``common.env_util.is_wrapped`` and ``common.env_util.unwrap_wrapper`` functions for checking/unwrapping
  an environment for specific wrapper.
- Added ``env_is_wrapped()`` method for ``VecEnv`` to check if its environments are wrapped
  with given Gym wrappers.
- Added ``monitor_kwargs`` parameter to ``make_vec_env`` and ``make_atari_env``
- Wrap the environments automatically with a ``Monitor`` wrapper when possible.
- ``EvalCallback`` now logs the success rate when available (``is_success`` must be present in the info dict)
- Added new wrappers to log images and matplotlib figures to tensorboard. (@zampanteymedio)
- Add support for text records to ``Logger``. (@lorenz-h)


Bug Fixes:
^^^^^^^^^^
- Fixed bug where code added VecTranspose on channel-first image environments (thanks @qxcv)
- Fixed ``DQN`` predict method when using single ``gym.Env`` with ``deterministic=False``
- Fixed bug that the arguments order of ``explained_variance()`` in ``ppo.py`` and ``a2c.py`` is not correct (@thisray)
- Fixed bug where full ``HerReplayBuffer`` leads to an index error. (@megan-klaiber)
- Fixed bug where replay buffer could not be saved if it was too big (> 4 Gb) for python<3.8 (thanks @hn2)
- Added informative ``PPO`` construction error in edge-case scenario where ``n_steps * n_envs = 1`` (size of rollout buffer),
  which otherwise causes downstream breaking errors in training (@decodyng)
- Fixed discrete observation space support when using multiple envs with A2C/PPO (thanks @ardabbour)
- Fixed a bug for TD3 delayed update (the update was off-by-one and not delayed when ``train_freq=1``)
- Fixed numpy warning (replaced ``np.bool`` with ``bool``)
- Fixed a bug where ``VecNormalize`` was not normalizing the terminal observation
- Fixed a bug where ``VecTranspose`` was not transposing the terminal observation
- Fixed a bug where the terminal observation stored in the replay buffer was not the right one for off-policy algorithms
- Fixed a bug where ``action_noise`` was not used when using ``HER`` (thanks @ShangqunYu)

Deprecations:
^^^^^^^^^^^^^

Others:
^^^^^^^
- Add more issue templates
- Add signatures to callable type annotations (@ernestum)
- Improve error message in ``NatureCNN``
- Added checks for supported action spaces to improve clarity of error messages for the user
- Renamed variables in the ``train()`` method of ``SAC``, ``TD3`` and ``DQN`` to match SB3-Contrib.
- Updated docker base image to Ubuntu 18.04
- Set tensorboard min version to 2.2.0 (earlier version are apparently not working with PyTorch)
- Added warning for ``PPO`` when ``n_steps * n_envs`` is not a multiple of ``batch_size`` (last mini-batch truncated) (@decodyng)
- Removed some warnings in the tests

Documentation:
^^^^^^^^^^^^^^
- Updated algorithm table
- Minor docstring improvements regarding rollout (@stheid)
- Fix migration doc for ``A2C`` (epsilon parameter)
- Fix ``clip_range`` docstring
- Fix duplicated parameter in ``EvalCallback`` docstring (thanks @tfederico)
- Added example of learning rate schedule
- Added SUMO-RL as example project (@LucasAlegre)
- Fix docstring of classes in atari_wrappers.py which were inside the constructor (@LucasAlegre)
- Added SB3-Contrib page
- Fix bug in the example code of DQN (@AptX395)
- Add example on how to access the tensorboard summary writer directly. (@lorenz-h)
- Updated migration guide
- Updated custom policy doc (separate policy architecture recommended)
- Added a note about OpenCV headless version
- Corrected typo on documentation (@mschweizer)
- Provide the environment when loading the model in the examples (@lorepieri8)


Pre-Release 0.10.0 (2020-10-28)
-------------------------------

**HER with online and offline sampling, bug fixes for features extraction**

Breaking Changes:
^^^^^^^^^^^^^^^^^
- **Warning:** Renamed ``common.cmd_util`` to ``common.env_util`` for clarity (affects ``make_vec_env`` and ``make_atari_env`` functions)

New Features:
^^^^^^^^^^^^^
- Allow custom actor/critic network architectures using ``net_arch=dict(qf=[400, 300], pi=[64, 64])`` for off-policy algorithms (SAC, TD3, DDPG)
- Added Hindsight Experience Replay ``HER``. (@megan-klaiber)
- ``VecNormalize`` now supports ``gym.spaces.Dict`` observation spaces
- Support logging videos to Tensorboard (@SwamyDev)
- Added ``share_features_extractor`` argument to ``SAC`` and ``TD3`` policies

Bug Fixes:
^^^^^^^^^^
- Fix GAE computation for on-policy algorithms (off-by one for the last value) (thanks @Wovchena)
- Fixed potential issue when loading a different environment
- Fix ignoring the exclude parameter when recording logs using json, csv or log as logging format (@SwamyDev)
- Make ``make_vec_env`` support the ``env_kwargs`` argument when using an env ID str (@ManifoldFR)
- Fix model creation initializing CUDA even when `device="cpu"` is provided
- Fix ``check_env`` not checking if the env has a Dict actionspace before calling ``_check_nan`` (@wmmc88)
- Update the check for spaces unsupported by Stable Ba