## 如何删除lazyvim

主要是删除这些文件:

~/.config/nvim (LazyVim config)

~/.local/share/nvim (用户数据目录下，这里有lazy安装的插件和mason安装的包)

~/.local/state/nvim (Session state directory: storage for file drafts, swap, undo, shada.)

~/.cache/nvim (Neovim cache)

```bash
rm -rf ~/.config/nvim
 
rm -rf ~/.local/share/nvim
 
rm -rf ~/.local/state/nvim
```