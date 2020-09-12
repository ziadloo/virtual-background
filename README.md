# Virtual Background
Virtual background for Webcams in Linux using machine learning. This source code is accompanying my
[blog post](https://challengeenthusiast.com/2020/09/11/virtual-background-for-webcams-in-linux-using-ml/).
For description on how it is developed, please refer to the blog post.

The aim of this code is to use BodyPix project to remove the background and at the same time project a helmet onto the
fact using the coordinates returned by the same BodyPix. It uses a (separate) machine learning model to estimate the
orientation of 3D model and render it in the right place.

## Prerequisites

You'll need a Linux OS (any flavor should do) with an NVIDIA GPU installed.

## How to install

The installation process is designed so you can play around with the code, if you like. It uses docker to install the
requirements without cluttering your system. Once you are done with the project, all you need to do to get rid of it is
to delete the container, its image and the downloaded folder.

1. First make sure that your NVIDIA driver is installed
1. Then, Install [`nvidia-docker2`](https://github.com/NVIDIA/nvidia-docker)
1. Navigate to the project's folder and build the docker file (heads up, it will take a while):
   ```
   docker build -t virtual_background .
   ```
1. Install [`v4l2loopback`](https://github.com/umlaeute/v4l2loopback) for your flavor of Linux
1. To setup the `v412loopback`, run the following commands in shell:
    ```
    sudo modprobe -r v4l2loopback
    sudo modprobe v4l2loopback devices=1 video_nr=20 card_label="v4l2loopback" exclusive_caps=1
    ```
   This last command is creating a virtual webcam as the device `/dev/video20` (assuming it's available, if it isn't
   just change the command) 
1. Edit the `resoures/config.json` file and change it to match your needs
1. Make sure no application is using your physical webcam
1. Run the container after you changed to the root folder of the project (for the `$(pwd)` to work properly):
    ```
    docker run -it --rm --gpus=all --name vb \
        --device=/dev/video0:/dev/video0 \
        --device=/dev/video20:/dev/video20 \
        virtual_background
    ```
1. Run the browser and enjoy

### Note

The last `docker run` is meant to let you run the code. But if you are interested in making changes to the code, you
can use this one instead which will also copy the code to the container as well:

```
docker run -it --rm --gpus=all --name vb \
    -v $(pwd)/source_code/face_orientation:/root/virtual_background/face_orientation \
    -v $(pwd)/source_code/resources:/root/virtual_background/resources \
    -v $(pwd)/source_code/src:/root/virtual_background/src \
    --device=/dev/video0:/dev/video0 \
    --device=/dev/video20:/dev/video20 \
    virtual_background
```

But this way you cannot install a new package and you are limited to the packages currently installed. If you want
to install a new package, you have to remove the `--rm` tag and `docker exec` your way to the container and do things
that way. The good thing is that if you use the `docker run` command above with `-v`s, your folders are shared with the
container and you can keep working on them and then run them in the container without needing to copy them. 