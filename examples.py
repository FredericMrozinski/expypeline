# In this notebook, we implement some very basic CNN training on the FashionMNIST dataset with PyTorch and
# ExPypeline. The goal is to explore neural architecture search and hyperparameter search with the
# mechanics of ExPypeline.

import expypeline as exp
import torch
from torchvision import datasets, models
from torchvision.transforms import ToTensor

# Below, we define different experiment steps that make up the entire experiment.

# First, we define the step for data loading the FashionMNIST dataset
def step_load_data(state: exp.ExpState):

    # ExPypeline provides multiple data directories. A global one, shared for all experiments (which we use here),
    # an experiment run-global one shared across all runs of a particular experiment, and one that is specific to
    # both the experiment and the run.
    data_dir = state.get_global_shared_data_dir()

    training_data = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=ToTensor()
    )
    test_data = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=ToTensor()
    )

    # We want to make the datasets visible to the following steps (e.g. training) in the experiment.
    # Therefore, we pass them to the given ExpState:
    state["training_data"] = training_data
    state["test_data"] = test_data

    # Set up the data loaders
    state["batch_size"] = 32
    state["training_data_loader"] = torch.utils.data.DataLoader(training_data, batch_size=state["batch_size"])
    state["test_data_loader"] = torch.utils.data.DataLoader(training_data, batch_size=state["batch_size"])

    # There is no need to return anything, ever. Instead, we write to ExpState as above.
    return


# Second, we want to define two different sets of hyperparameters that will be used for training.
# We define them by creating an ExpStep for each containing their definitions
# TODO: At some point, it would be nice to add an option for not needing functions to declare values.
# E.g.
#   pipeline.then(exp.ExpStateSpace("learning_rate", [1e-3, 1e-4]))
# But that's an idea for the future
def step_hyperparameters_1(state: exp.ExpState):
    state["learning_rate"] = 1e-3

def step_hyperparameters_2(state: exp.ExpState):
    state["learning_rate"] = 1e-4

# Next, we define two different model architectures
def step_load_alex_net(state: exp.ExpState):
    alexnet = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', weights=models.AlexNet_Weights.DEFAULT)
    alexnet.classifier[6] = torch.nn.Linear(in_features=alexnet.classifier[6].in_features, out_features=10)
    state["model"] = alexnet

def step_load_res_net(state: exp.ExpState):
    resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', weights=models.ResNet18_Weights.DEFAULT)
    resnet.fc = torch.nn.Linear(in_features=resnet.fc.in_features, out_features=10)
    state["model"] = resnet

def finetune_model(state: exp.ExpState):

    def evaluate_model():
        with torch.no_grad():
            for batch in state["test_data_loader"]:
                input, target = batch
                model_out = state["model"](input)
                print(model_out)

    evaluate_model()



# After we defined the functions that will comprise our experiment(s), we will now define the runtime
# order of the above functions. We do so by building a pipeline and wrapping every function above into
# an ExpStep
pipeline = exp.ExpStep("Data Loading", step_load_data)
pipeline = pipeline.then(exp.ExpStep("Hyperparameters 1", step_hyperparameters_1))
pipeline = pipeline.branch(exp.ExpStep("Hyperparameters 2", step_hyperparameters_2))
pipeline = pipeline.then(exp.ExpStep("Model Loading Alex Net", step_load_alex_net))
pipeline = pipeline.branch(exp.ExpStep("Model Loading ResNet", step_load_res_net))
pipeline = pipeline.then(exp.ExpStep("Finetuning model", finetune_model))


# Finally, we set up the ExpSuite, the "mother" of all experiments. It acts as the main control point.
# We further pass the directory into which all experiment related files will be written (or read from).
suite = exp.ExpSuite("/home/frederic/Coding/expy")
# We now add the experiment and its pipeline to the global queue
suite.queue_experiment("FashionMNIST Hyperparameter Tuning", pipeline)

# Lastly, before we run the experiments, we need to pass a reproducibility-handler to the ExpSuite.
# Every step gets the chance to be executed with a certain random seed to ensure reproducibility,
# independently of their run order (which is especially important if e.g. one branch fails due to
# an exception and the other branches continue to be executed). This is required because every library
# used (NumPy, PyTorch, Transformers, etc.) may have their unique way of setting seeds which is why
# the user has to specify that.
def set_seed(seed: float):
    torch.manual_seed(seed)
suite.set_reproducibility_handler(set_seed)

# Optional: Define PyTorch's cache dir at the global shared data location
torch.hub.set_dir(suite.global_shared_directory + "/cache")

# Finally, we can run the experiment
suite.run()