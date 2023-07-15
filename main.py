import fileutil


config_file_path = "yolo.cfg"
blocks = fileutil.read_configuration(config_file_path)
network_info, module_list = fileutil.create_modules(blocks)
