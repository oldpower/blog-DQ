-- local Path = require('plenary.path')
-- progress = ''
-- require('cmake').setup({

--     cmake_executable = 'cmake', -- CMake executable to run.
--     save_before_build = true, -- Save all buffers before building.
--     parameters_file = 'build/neovim.json', -- 这里可以改成你想要的地址和名字，配置文件
--     build_dir = tostring(Path:new('{cwd}', 'build', '{os}-{build_type}')), -- Build directory. The expressions `{cwd}`, `{os}` and `{build_type}` will be expanded with the corresponding text values. Could be a function that return the path to the build directory.
--     default_parameters = { args = {}, build_type = 'Debug',
--         run_dir = tostring(Path:new('{cwd}', 'build', '{os}-{build_type}')), dap_configurations = 'cppdbg_vscode' }, -- The default values in `parameters_file`. Can also optionally contain `run_dir` with the working directory for applications.
--     -- samples_path = tostring(script_path:parent():parent():parent() / 'samples'), -- Folder with samples. `samples` folder from the plugin directory is used by default.
--     default_projects_path = tostring(Path:new(vim.loop.os_homedir(), 'code/cpp_code/')), -- 这里填入你想要的默认的地址.
--     configure_args = { '-D', 'CMAKE_EXPORT_COMPILE_COMMANDS=1' }, -- Default arguments that will be always passed at cmake configure step. By default tells cmake to generate `compile_commands.json`. 注意：这个comple_commands.json对于clangd编译器十分重要，要不然就会出现找不到头文件的情况
--     build_args = {}, -- Default arguments that will be always passed at cmake build step.
--     on_build_output = function(lines)
--         -- Get only last line
--         local match = string.match(lines[#lines], "(%[.*%])")
--         if match then
--             progress = string.gsub(match, "%%", "%%%%")
--         end
--     end, -- Callback that will be called each time data is received by the current process. Accepts the received data as an argument.
--     quickfix = {
--         pos = 'botright', -- Where to open quickfix
--         height = 10, -- Height of the opened quickfix.
--         only_on_error = false, -- Open quickfix window only if target build failed.
--     },
--     copy_compile_commands = true, -- Copy compile_commands.json to current working directory.
--     dap_configurations = { -- Table of different DAP configurations.
--         lldb_vscode = { type = 'codelldb', request = 'launch' },
--         cppdbg_vscode = { type = 'cppdbg', request = 'launch' },
--     },
--     dap_configuration = 'cppdbg_vscode', -- DAP configuration to use if the projects `parameters_file` does not specify one.
--     dap_open_command = require('dapui').open(), -- Command to run after starting DAP session. You can set it to `false` if you don't want to open anything or `require('dapui').open` if you are using https://github.com/rcarriga/nvim-dap-ui
-- })
--
--


-- vim.list_extend(lvim.lsp.automatic_configuration.skipped_servers, { "cmake" })  -- 防止自动配置冲突

-- local cmake_lsp = {
--   name = "cmake",
--   cmd = { "cmake-language-server" },
--   filetypes = { "cmake" },
--   root_dir = require("lspconfig.util").root_pattern("CMakeLists.txt", "build"),
-- }

-- require("lvim.lsp.manager").setup("cmake", cmake_lsp)
