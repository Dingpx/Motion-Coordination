import logging


def set_logger_with_name(name="Default_name",filepath="Default_filepath.txt"):

	formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
	logger_level = logging.INFO

	logger_name = name
	logger_filehandler_path = filepath

	logger = logging.getLogger(logger_name)
	logger.setLevel(level = logger_level)
	
	handler = logging.FileHandler(logger_filehandler_path)
	handler.setLevel(logger_level)
	handler.setFormatter(formatter)
	 
	console = logging.StreamHandler()
	console.setLevel(logger_level)
	console.setFormatter(formatter)
	 
	logger.addHandler(handler)
	logger.addHandler(console)

	return logger