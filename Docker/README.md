# Docker
- 使用<b>Docker</b>以在<b>非Ubuntu22或Ubuntu20</b>的宿主机上使用<b>SDK</b>
- 由于众所周知的原因，Docker拉取镜像很可能失败，因此提供三种方案，<b>方案一</b>需要从dockerfile构建，可能失败，<b>方案二</b>SDK提供的Docker镜像，<b>方案三</b>使用鱼香ROS脚本构建
## Docker安装
- 安装依赖 `sudo apt update && sudo apt install -y wget`
- 安装<b>Docker</b> `wget http://fishros.com/install -O fishros && . fishros`感谢[鱼香ROS](https://fishros.org.cn/forum/)
## SDK Docker镜像构建（方案1.1）
- 使用 `dockerfile` 文件进行SDK Docker镜像构建
- 授权所有宿主机用户访问<b>X11</b> `xhost +local:root`，每次都需要执行这个，如果觉得麻烦，可以将这个命令加入 `.bashrc内`
- 构建镜像 `sudo docker build -t ${镜像名字}:${版本号} .`
    |参数|说明|
    |:-:|:-:|
    |镜像名字|镜像名字，可以自行填（必须小写）,SDK为geek-egret/fpv|
    |版本号|版本号，可以自行填写，SDK为latest|
- 清理构建缓存 `docker builder prune --all --force`
## Docker容器构建和运行（方案1.2）
- Docker容器构建
    ```
    sudo docker run -it \
    --net=host \
    --privileged \
    --device=/dev/dri \
    --device-cgroup-rule='c 189:* rmw' \
    --device-cgroup-rule='c 81:* rmw' \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -v /usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa \
    -v /etc/ssl/certs:/etc/ssl/certs \
    -v /usr/share/ca-certificates:/usr/share/ca-certificates \
    -v /home/leeeezy/Workspace/:/home/leeeezy/Workspace/ \
    -e DISPLAY=$DISPLAY \
    --user leeeezy \
    -e QT_X11_NO_MITSHM=1 \
    geek-egret/fpv:latest /bin/bash
    ```
    解释
    ```
    sudo docker run -it \
    # 共享宿主机的网络（可选，方便相机设备映射）
    --net=host \
    # 特权模式（解决GPU/设备访问权限）
    --privileged \
    # 挂载设备（保证深度相机等设备能够被Docker容器访问）
    --device=/dev/dri \
    --device-cgroup-rule='c 189:* rmw' \
    --device-cgroup-rule='c 81:* rmw' \
    -v /dev/bus/usb:/dev/bus/usb \
    -v /dev/video0:/dev/video0 \
    -v /dev/video1:/dev/video1 \
    # 挂载X11套接字（核心：让容器访问宿主机显示器）
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    # 挂载宿主机的显卡驱动（可选，提升渲染性能）
    -v /usr/lib/x86_64-linux-gnu/mesa:/usr/lib/x86_64-linux-gnu/mesa \
    # 挂载证书（否则会出现换国内源失败）
    -v /etc/ssl/certs:/etc/ssl/certs \                          (若尾巴加上 `:ro` 则为只读)
    -v /usr/share/ca-certificates:/usr/share/ca-certificates\   (若尾巴加上 `:ro` 则为只读)
    # 新增：共享宿主机的SDK目录到容器内的工作目录（第一个路径是宿主机，第二个路径是容器）
    -v /home/leeeezy/Workspace/:/home/leeeezy/Workspace/ \
    # 传递显示环境变量
    -e DISPLAY=$DISPLAY \
    # 指定普通用户运行（与Dockerfile中USER_NAME一致）
    --user leeeezy \
    # 禁用MIT-SHM（避免X11共享内存报错）
    -e QT_X11_NO_MITSHM=1 \
    # 镜像名称（与构建时一致）
    geek-egret/fpv:latest  /bin/bash
    ```
- 查看所有容器 `docker ps -a` 找到刚刚新建的容器
- 使用 `docker attach 容器的前三位ID`或`sudo docker exec -it 容器的前三位ID bash` 进入容器，若没有启动，则先使用 `docker start 容器的前三位ID`
- 换源，将容器内的 `/etc/apt/sources.list` 更换为SDK的 `sources.list`
- 安装ROS2及换源 `wget http://fishros.com/install -O fishros && . fishros`
## 使用SDK的镜像构建容器
- 使用 `sudo docker import 镜像包名称 镜像名称:版本` 加载镜像
- 使用 `sudo docker images -a` 查看所有镜像
- 使用 `sudo docker tag 镜像的前四位ID 镜像名` 给未命名镜像命名
- 使用<b>方案1.2</b>中的命令构建容器
- 导出容器到镜像 `docker export -o 输出路径.tar 容器名或ID`
## 问题
- `Cannot connect to the Docker daemon at unix:///var/run/docker.sock. Is the docker daemon running?` 守护进程未启动，使用 `sudo systemctl start docker`启动
- 如果不想每次都使用 `sudo` 命令，则需要将当前用户加入 `docker组`：`sudo groupadd Docker && sudo usermod -aG Docker $USER`

