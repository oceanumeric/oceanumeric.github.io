```bash
docker run \
        -it --rm \
        --gpus all --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        --shm-size=8gb \
        -p 7070:7070 \
        -v "${PWD}":/home/work \
        vsc-cuda-container-92f8d0bcb7ce38ae6a144f94ac61d015 

# install jupyter notebook
pip install jupyter
# start jupyter notebook
jupyter-lab --ip 0.0.0.0 --port=7070 --no-browser --allow-root \
        --ServerApp.token="innolab" --ServerApp.password="innolab"
```