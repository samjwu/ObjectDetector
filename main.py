config_file_path = "yolo.cfg"
configs = []
block = {}
blocks = []

with open(config_file_path, "r") as file:
    lines = file.read().split("\n")
    
    for line in lines:
        # skip comments and empty lines
        if (line[0] == "#" or len(line) == 0):
            continue
        else:
            configs.append(line.lstrip().rstrip())

    for config in configs:
        # new block
        if config[0] == "[":
            # add previous block to list of blocks
            if block:
                blocks.append(block)
                block.clear()

            block["type"] = config[1:-1]
        else:
            key, value = config.split("=") 
            block[key] = value
    blocks.append(block)
