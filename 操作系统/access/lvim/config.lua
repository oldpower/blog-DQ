-- For new users:
-- When you run lvim firstly, please run :PackerInstall and :PackerCompile to compile the packages
-- Then a "packer_compild.lua" will be created under plugin folder
-- Please make sure the neovim version is v0.8.x
-- For a list of keyboard shortcut, please type <space> <s> <k>
-- Have fun!!

-- 模糊查找插件
reload "user.telescope"
-- 快捷键
reload "user.keymaps"
-- 语法高亮、代码解析
reload "user.treesitter"
reload "user.lsp"
reload "user.nvimtree"
reload "user.plugins"
-- 针对lsp signature插件，用于显示函数签名和参数信息
reload "user.signature"
-- 图标
reload "user.outlines"
-- 自动命令
reload "user.autocommands"

-- CMake
-- reload "user.cmake"
-- reload "user.chatgpt"


-- general 
-- vim.format_on_save.enabled = false
lvim.colorscheme = "tokyonight-moon"
-- 
lvim.transparent_window = true
-- 共享剪贴板
vim.opt.clipboard = "unnamedplus"
vim.opt.encoding = "utf-8"
lvim.use_icons=true

-- 仪表盘
lvim.builtin.alpha.active = true
lvim.builtin.alpha.mode = "dashboard"
-- lunarvim内置终端插件
lvim.builtin.terminal.active = true


