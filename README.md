# robot-jugg
ler
Robotic juggling. 

## Setup
1. [Install Drake for python](https://drake.mit.edu/python_bindings.html)
2. Install Jupyter Lab:
```
pip3 install jupyterlab
```
3. Install `ipywidgets`:
```
pip3 install ipywidgets
```
4. Install NodeJS (>= v10.0.0) for activating Widget Javascript:
    - On Ubuntu with `apt-get`:
    ```
    curl -sL https://deb.nodesource.com/setup_12.x | sudo -E bash -
    sudo apt-get install -y nodejs
    ```

    - On Mac with Homebrew:
    ```
    brew install nodejs
    ```
5. Enable Widget Javascript in Jupyter:
    ```    
    jupyter labextension install @jupyter-widgets/jupyterlab-manager
    ```
6. Install `manipulation` package (outside this repo), and follow the local setup steps for manipulation (already did Drake!) in `drake.html`:
    ```
    git clone --recursive git@github.com:RussTedrake/manipulation.git
    ```