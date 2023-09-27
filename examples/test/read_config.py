"""Defines the configuration to be loaded before running any experiment"""
from configobj import ConfigObj
import string


class Config(object):
    def __init__(self, filename: string):
        """
        Read from a config file
        :param filename: name of the file to read from
        """

        self.filename = filename
        config = ConfigObj(self.filename)
        self.config = config
    
        # Comments on the experiments running
        self.comment = config["comment"]

        # Model name and location to store
        # self.model_path = config["train"]["model_path"]
        self.axis = config["train"]["axis"]
        # GT point cloud path
        self.gt_pc = config["train"]["gt_pc"]

        self.ignore_uv = config["train"].as_bool("ignore_uv")
        self.loss_type = config["train"]["loss_type"]

        self.ctrlpts_size_u = config["train"].as_int("ctrlpts_size_u")
        self.ctrlpts_size_v = config["train"].as_int("ctrlpts_size_v")
        
        self.resolution_u = config["train"].as_int("resolution_u")
        self.resolution_v = config["train"].as_int("resolution_v")
        self.sample_size_u = config["train"].as_int("sample_size_u")
        self.sample_size_v = config["train"].as_int("sample_size_v")
        
        self.out_dim_u = config["train"].as_int("out_dim_u")
        self.out_dim_v = config["train"].as_int("out_dim_v")
        # path to the model
        # self.pretrain_model_path = config["train"]["pretrain_model_path"]

        # Normals
        self.normals = config["train"].as_bool("normals")

        self.num_epochs = config["train"].as_int("num_epochs")
        # number of training examples
        # self.num_train = config["train"].as_int("num_train")
        # self.num_val = config["train"].as_int("num_val")
        # self.num_test = config["train"].as_int("num_test")

        self.num_points = config["train"].as_int("num_points")
        # Number of control points
        self.ctrpts_size = config["train"].as_int("ctrpts_size")
        # Weight to the loss function for stretching
        self.loss_weight = config["train"].as_float("loss_weight")
        # base function degree
        self.degree = config["train"].as_int("degree")

        # Number of epochs to run during training
        self.epochs = config["train"].as_int("num_epochs")

        # batch size, based on the GPU memory
        self.batch_size = config["train"].as_int("batch_size")

        # Mode of training, 1: supervised, 2: RL
        self.mode = config["train"].as_int("mode")

        # Learning rate
        self.lr1 = config["train"].as_float("lr1")
        self.lr2 = config["train"].as_float("lr2")
        
        # Number of epochs to wait before decaying the learning rate.
        self.patience = config["train"].as_int("patience")

        # Optimizer: RL training -> "sgd" or supervised training -> "adam"
        self.optim = config["train"]["optim"]

        # Epsilon for the RL training, not applicable in Supervised training
        self.accum = config["train"].as_int("accum")

        # Whether to schedule the learning rate or not
        self.lr_sch = config["train"].as_bool("lr_sch")

    def write_config(self, filename):
        """
        Write the details of the experiment in the form of a config file.
        This will be used to keep track of what experiments are running and
        what parameters have been used.
        :return:
        """
        self.config.filename = filename
        self.config.write()

    def get_all_attribute(self):
        """
        This function prints all the values of the attributes, just to cross
        check whether all the data types are correct.
        :return: Nothing, just printing
        """
        for attr, value in self.__dict__.items():
            print(attr, value)


if __name__ == "__main__":
    file = Config("config_synthetic.yml")
    print(file.write_config())