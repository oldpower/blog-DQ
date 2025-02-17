
## <center>怎么查看OpenGL 版本</center>
要查看系统上安装的 OpenGL 版本，可以使用以下方法。具体步骤取决于你的操作系统和图形驱动。

---

### 方法 1：使用 `glxinfo` 命令（Linux）
`glxinfo` 是一个常用的工具，可以显示 OpenGL 和 GLX（OpenGL 扩展）的详细信息。

#### 步骤：
1. 打开终端。
2. 安装 `mesa-utils` 包（如果尚未安装）：
   ```bash
   sudo apt-get install mesa-utils
   ```
3. 运行以下命令查看 OpenGL 版本：
   ```bash
   glxinfo | grep "OpenGL version"
   ```

#### 输出示例：
```bash
OpenGL version string: 4.6.0 NVIDIA 470.82.00
```
这表示系统上的 OpenGL 版本是 4.6，使用的是 NVIDIA 驱动。

---

### 方法 2：使用 `glxgears` 命令（Linux）
`glxgears` 是一个简单的 OpenGL 测试程序，运行时会在终端中显示 OpenGL 版本。

#### 步骤：
1. 打开终端。
2. 运行以下命令：
   ```bash
   glxgears -info
   ```
3. 在输出的开头部分，你会看到类似以下的信息：
   ```bash
   GL_RENDERER   = GeForce GTX 1080/PCIe/SSE2
   GL_VERSION    = 4.6.0 NVIDIA 470.82.00
   GL_VENDOR     = NVIDIA Corporation
   ```
   其中 `GL_VERSION` 就是 OpenGL 版本。

---

### 方法 3：使用 `OpenGL Extensions Viewer`（Windows）
如果你在 Windows 系统上，可以使用第三方工具（如 OpenGL Extensions Viewer）来查看 OpenGL 版本。

#### 步骤：
1. 下载并安装 [OpenGL Extensions Viewer](https://download.cnet.com/OpenGL-Extensions-Viewer/3000-18487_4-34442.html)。
2. 运行程序，它会自动检测并显示 OpenGL 版本和显卡信息。

---

### 方法 4：使用 `System Information`（Windows）
Windows 自带的系统信息工具也可以显示 OpenGL 版本。

#### 步骤：
1. 按 `Win + R` 打开运行窗口，输入 `dxdiag`，然后按回车。
2. 在打开的 "DirectX 诊断工具" 窗口中，切换到 "显示" 选项卡。
3. 在 "驱动程序模型" 部分，你会看到 OpenGL 版本信息。

---

### 方法 5：使用代码查询 OpenGL 版本
如果你熟悉编程，可以通过编写简单的 OpenGL 程序来查询版本。

#### C++ 示例代码：
```cpp
#include <iostream>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    GLFWwindow* window = glfwCreateWindow(640, 480, "OpenGL Version", NULL, NULL);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        return -1;
    }

    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;

    glfwTerminate();
    return 0;
}
```

#### 编译和运行：
1. 安装 GLFW 和 GLEW 库。
2. 编译并运行程序，输出会显示 OpenGL 版本。

---

### 方法 6：使用 `lspci` 和 `modinfo`（Linux）
如果你只是想查看显卡型号和驱动信息，可以使用以下命令：

#### 步骤：
1. 查看显卡型号：
   ```bash
   lspci | grep VGA
   ```
2. 查看显卡驱动信息：
   ```bash
   modinfo <驱动名称>
   ```
   例如，对于 NVIDIA 显卡：
   ```bash
   modinfo nvidia
   ```

---

### 总结
- **Linux**：使用 `glxinfo` 或 `glxgears` 命令。
- **Windows**：使用 OpenGL Extensions Viewer 或 DirectX 诊断工具。
- **编程**：通过 OpenGL 程序查询版本。

如果你在 Linux 系统上，推荐使用 `glxinfo` 命令，因为它简单且直接。如果 OpenGL 版本过低（如低于 3.0），可能需要更新显卡驱动或安装支持更高 OpenGL 版本的驱动。