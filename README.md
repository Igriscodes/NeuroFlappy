# NeuroFlappy

A real-time **Genetic Algorithm** and **Neural Network** simulation built in C++ using the [raylib](https://www.raylib.com/) library. Watch a population of 250 birds learn to navigate obstacles through trial and error across multiple generations.

## How it Works

The project implements a basic AI from scratch:

* **Neural Network**: Each bird has a brain with **5 inputs** (y-position, velocity, distance to next pipe, and pipe gap boundaries), **8 hidden nodes**, and **1 output** (flap or glide).
* **Genetic Algorithm**: After all birds die, the best performers are selected to seed the next generation. This includes **elitism** (saving the top 5%) and **crossover/mutation** to evolve better flying behavior.
* **Live Visualization**: A HUD on the top-right displays the neural network of the current "best" bird, showing real-time weight activations and node firing.

## Installation

### Prerequisites

* A C++17 compatible compiler.
* CMake (3.14 or higher).


* Raylib dependencies (automatically fetched via CMake).



### Building the Project

1. Clone the repository and navigate to the directory.
```bash
git clone https://github.com/Igriscodes/NeuroFlappy.git
```
3. Create a build folder and compile:
```bash
cd build
cmake ..
make -j12
```


3. **Assets**: Ensure you have a `build/assets/` directory containing the following images:
* `background.png`
* `bird.png`
* `pipe.png`



## Usage

Once compiled, run the executable:

```bash
./flappy
```
![demo](https://github.com/user-attachments/assets/62a7bbde-8355-4e93-ae74-2294de02d758)

### Controls

| Key | Action |
| --- | --- |
| **Left / Right Arrow** | Change simulation speed (1x to 100x) |
| **F11 / Alt+Enter** | Toggle Fullscreen |
| **ESC** | Exit simulation |

## License

[GNU Lesser General Public License v2.1](LICENSE) - Feel free to use and modify
