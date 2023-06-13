## Dependencies and Setups

Required:

- Ubuntu 20.04
- Python-3 with Pip (`apt install python3 python3-pip`)
- CUDA >= 10 if using GPU

### Python (for LaneNet Training)

1. Confirm that you are currently in the `gem/` directory
2. Navigate to the empty data directory `cd data` where we will store our lane detection training set
2. Download TuSimple lane detection [dataset](https://github.com/TuSimple/tusimple-benchmark/issues/3) (around 10 GB) by running the following commands (this will take some time...)
   ```
   wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip
   unzip train_set.zip
   python ../tusimple_transform.py --src_dir .
   ``` 
3. All configurations for the training job can be found in `train_lanenet.yaml`. Edit this file to your needs.
5. Start training by running the following command
   ```
   python train_lanenet.py --checkpoint [path to checkpoint to resume training from] [config_file] [config_name]
   ```
   [config_file] - path to configuration file

   [config_name] - name of configuration to run, either the config `train-lanenet` or `prune-lanenet-iter`

   Example
   ```
   python train_lanenet.py train_lanenet.yaml train-lanenet
   ```

### Python (for LaneNet Inference)

LaneNet will be run as a ROS node which feeds into the GEM simulator.
As ROS installs its python binding packages into the system-wide Python,
we are required to also do so (instead of using a virtual env).

1. Exit all virtual environment you're currently in. Confirm your `python3` is the system-wide one
   (e.g., using `which python3`).

1. Install `Cython` and `scikit-build` so that any packages we install next can be compiled if needed to:

   ```bash
   pip3 install Cython scikit-build
   ```

1. Install packages required for inference:

   ```bash
   pip3 install -r ./infer_req.txt
   ```

1. Install an inference engine depending on your device:

   - **[gpu]** If using a device with GPU, make sure that `nvcc` is in your PATH,
     and install `onnxruntime-gpu` by `pip install onnxruntime-gpu`;

   - **[cpu]** If using CPU for inference, install `onnxruntime` by `pip install onnxruntime`.

### ROS

GEM simulator only works with ROS Noetic which can be easily installed on Ubuntu 20.
Other Linux distros are unsupported unless you install ROS Noetic from source (not recommended).

1. Follow [this instruction](https://wiki.ros.org/noetic/Installation) to add ROS repo to your system.

1. Install the bare minimum for ROS: `sudo apt install ros-noetic-ros-base ros-noetic-gazebo-ros-pkgs`
   
   - `ros-noetic-gazebo-ros-pkgs` is required for camera to work in Gazebo.

1. Install utils and initialize `rosdep`.

   ```bash
   sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential
   sudo rosdep init
   rosdep update
   ```

1. Add setup script to your bashrc:

   ```bash
   echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
   source ~/.bashrc
   ```

## Getting Started

1. Create a `./catkin_ws/` folder here for the workspace and set up components to use from `gemsim/`:

   ```bash
   mkdir catkin_ws/
   cd catkin_ws
   mkdir src/
   ln -s (realpath ../gemsim/polaris_gem_simulator/*) src/
   ln -s (realpath ../gemsim/polaris_gem_drivers_sim/*) src/
   ```

1. Use `rosdep` to discover and install dependencies for this project,
   and install `jsk-rviz-plugins` (which `rosdep` cannot detect), then compile:

   ```bash
   rosdep install --from-paths src --ignore-src --rosdistro=noetic -y
   sudo apt install ros-noetic-jsk-rviz-plugins
   catkin_make
   ```

1. Source the workspace setup script: `source devel/setup.bash` (or add it to your bashrc)

1. Run `roslaunch gem_gazebo gem_gazebo_rviz.launch` for the simulator with race track (default world)

   It should display an RViz window and a Gazebo window (may take a while).

1. Obtain pretrained LaneNet model [here](https://drive.google.com/file/d/1OXQ43itVey6lDtowTOAwEqZPyLsGivih/view?usp=sharing) and save as `./baseline.onnx`.

1. In another 2 windows, run `python3 lanenet_node.py` (for the DNN) and
  `python3 stanley.py` (for the controller) respectively.
   This will drive the cart around on the race track.
