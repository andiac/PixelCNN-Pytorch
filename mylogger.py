import logging

class MyLogger():
    def __init__(self, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        self.fh = logging.FileHandler(f'{name}.log')
        self.fh.setLevel(logging.DEBUG)
        
        self.ch = logging.StreamHandler()
        self.ch.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        self.fh.setFormatter(formatter)
        self.ch.setFormatter(formatter)

        self.logger.addHandler(self.fh)
        self.logger.addHandler(self.ch)

    def info(self, info):
        self.logger.info(info)


if __name__ == "__main__":

    myLogger = MyLogger("yo")
    myLogger.info("blablabla")
    
