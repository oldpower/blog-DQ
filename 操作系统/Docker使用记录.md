### 查看容器

1. **查看正在运行的容器：**
   使用 `docker ps` 命令可以列出所有当前正在运行的 Docker 容器。这会显示容器的ID、名称、使用的镜像、状态以及其他相关信息。

   ```shell
   docker ps
   ```




2. **查看所有容器（包括停止的）：**
   如果你还想查看那些已经停止的容器，可以使用带有 `-a` 参数的 `docker ps` 命令。

   ```shell
   docker ps -a
   ```
  - `ps` 表示“进程状态”（process status），用于列出容器。
  - `-a` 或 `--all` 参数表示显示所有容器，包含那些已经退出或停止的容器。

    执行该命令后，你会看到一个列表，其中包含了每个容器的详细信息，如容器ID、名称、使用的镜像、创建时间、状态以及端口映射等信息。这对于管理你的容器非常有用，无论是为了重启某个已停止的容器，删除不再需要的容器，还是仅仅是为了检查当前系统上存在哪些容器。

    如果你还想查看容器更多的细节信息，可以使用 `docker inspect [容器ID或容器名称]` 命令，这将返回关于指定容器的详细配置和状态信息。


### 查看镜像

1. **查看本地镜像：**
   要查看当前系统中存在的 Docker 镜像，可以使用 `docker images` 命令。这将展示所有已下载到本地的镜像列表，包括它们的仓库名、标签、ID、创建时间和大小等信息。

   ```shell
   docker images
   ```

### 其他有用的命令

- **查看Docker版本：**

  ```shell
  docker version
  ```

- **查看某个容器的详细信息：**

  ```shell
  docker inspect [容器ID或容器名称]
  ```

- **查看容器的日志输出：**

  ```shell
  docker logs [容器ID或容器名称]
  ```

- **查看Docker资源使用情况：**

  ```shell
  docker stats
  ```

---

### 关于缓存


1. **使用了已有的镜像**：
   - 如果你没有明确指定 `-t` 参数来给新构建的镜像命名，或者指定了一个已经存在的标签（repository:tag），Docker 不会覆盖旧的镜像，而是可能会直接使用现有的镜像。
   - **解决方法**：确保你在构建时使用了一个新的标签或覆盖了旧的标签。例如：
     ```shell
     docker build -t trt_starter:cuda11.4-cudnn8-tensorrt8.6_v1.0 .
     ```

2. **Dockerfile 或上下文未更新**：
   - 如果 Dockerfile 或者构建上下文没有变化，Docker 会利用缓存来加速构建过程，这可能导致你认为构建的是一个新镜像，但实际上它只是复用了之前的层。
   - **解决方法**：检查 Dockerfile 是否有修改，尝试清理 Docker 缓存重新构建：
     ```shell
     docker build --no-cache -t trt_starter:cuda11.4-cudnn8-tensorrt8.6_v1.0 .
     ```

3. **镜像名称或标签冲突**：
   - 可能存在多个镜像使用了相同的标签名，但它们的内容不同。在这种情况下，即使你重新构建了镜像，也可能是在引用旧的镜像。
   - **解决方法**：确认是否有重复的标签，并删除不需要的旧镜像：
     ```shell
     docker rmi trt_starter:cuda11.4-cudnn8-tensorrt8.6_v1.0
     ```
     然后再次尝试构建。

---

### 停止容器
在使用 `docker run` 启动容器后，如果你想要关闭或停止容器，有几种方法可以实现。以下是详细的操作步骤和相关命令：

#### 停止正在运行的容器

1. **通过容器名称或ID停止容器**：
   - 首先，你需要知道容器的名称或ID。可以通过 `docker ps` 查看当前正在运行的所有容器。
     ```shell
     docker ps
     ```
   - 找到你想要停止的容器的名称或ID，然后使用 `docker stop` 命令来停止它。例如，如果容器ID是 `abc123` 或者容器名为 `my_container`，你可以这样操作：
     ```shell
     docker stop abc123
     # 或者
     docker stop my_container
     ```

2. **强制停止容器**：
   - 如果容器没有响应标准的 `docker stop` 命令（默认会发送SIGTERM信号并等待10秒），你可以使用 `docker kill` 命令强制停止容器：
     ```shell
     docker kill abc123
     # 或者
     docker kill my_container
     ```

#### 交互式容器的退出

如果你是以交互模式启动的容器（即带有 `-it` 参数），比如：
```shell
docker run -it ubuntu /bin/bash
```
在这种情况下，你可以在容器内执行以下操作来退出：

1. **优雅地退出容器**：
   - 在容器的命令行中输入 `exit` 或按 `Ctrl+D` 来正常退出容器，并停止该容器。

2. **后台运行容器**：
   - 如果你希望容器在启动后直接进入后台运行，可以添加 `-d` 参数：
     ```shell
     docker run -d ubuntu /path/to/script.sh
     ```
   - 这样，容器将在后台运行，不会占用你的终端。

---

### 删除容器

**删除已停止的容器**：
  如果你想删除一个已经停止的容器，可以使用 `docker rm` 命令：
  ```shell
  docker rm my_container
  ```

### 删除镜像
    ```bash
    docker rmi image_name
    ```

---

### 重新运行


**重启停止的容器**：如果你想重新启动一个已经停止的容器，可以使用 `docker start` 命令。
```bash
 docker start my_container_name_or_id
#如果你想在启动后立即附加到容器的标准输入输出，可以使用 `docker attach` 或者 `docker exec`。

 docker start my_container_name_or_id
 docker attach my_container_name_or_id
 # 或者
 docker exec -it my_container_name_or_id /bin/bash
 ```
