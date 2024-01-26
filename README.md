# GPT build up

This is the repo for building up the environment for the CPU / GPU version Grid Python Toolkit.

## Build with container

### CPU version from my image (no mpi)

```bash
docker pull docker.io/greyyyhjc/gpt_cpu:coulomb
docker run --name gpt_contain --hooks-dir=/usr/share/containers/oci/hooks.d/ --runtime=nvidia -it greyyyhjc/gpt_cpu:coulomb
```


## Build on cluster

### Zaratan CPU version
```bash
./script/zaratan.cpu
```


### Zaratan GPU version
```bash
./script/zaratan.gpu
```



## Prerequisites for using GPU in the container

First, the host machine should have NVIDIA GPU and the driver should be installed. Second, we would like to use Docker to build a clean environment to install SIMULATeQCD. So the host machine should have Docker (or Podman) installed. Third, to use GPU in the container, we need to install nvidia-container-toolkit and modify some settings. The following steps are the prerequisites.

- 1. Install NVIDIA driver, check with the following command.
```bash
nvidia-smi
```

- 2. Install nvidia-container-toolkit, check with the following command.
```bash
which nvidia-container-toolkit
``` 

- 3. Check if the host machine has the directory file /usr/share/containers/oci/hooks.d/oci-nvidia-hook.json. If it doesn't exist, use the following command to create it.
```bash
Content=`cat << 'EOF'
{
    "version": "1.0.0",
    "hook": {
        "path": "/usr/bin/nvidia-container-toolkit",
        "args": ["nvidia-container-toolkit", "prestart"],
        "env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        ]
    },
    "when": {
        "always": true,
        "commands": [".*"]
    },
    "stages": ["prestart"]
}
EOF`

HookFile=/usr/share/containers/oci/hooks.d/oci-nvidia-hook.json
sudo mkdir -p `dirname $HookFile`
sudo echo "$Content" > $HookFile
```

- 4. Modify the configuration to allow users to execute and modify CUDA containers with regular user privileges.
```bash
sudo sed -i 's/^#no-cgroups = false/no-cgroups = true/;' /etc/nvidia-container-runtime/config.toml
```




