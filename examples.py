import expypeline as expy


# Let's run an experiment for sentence coherence:
# Generate sentence embeddings for multiple sentences.
# Hypothesis:
# If coherent, the differences between adjacent sentences change very little. In other words: they should
# make up a "smooth" curve through the hyperspace
# On the other hand if not coherent, there may be a sharp jump somewhere

def init_step(state: expy.ExpState):
    state["test"] = "a"
    print(state["test"])

def middle_step_1(state: expy.ExpState):
    print("Middle step 1")
    raise RuntimeError("well that's unfortunate")
    state["test"] = "b"
    state["test2"] = "c"
    print(state["test"])

def middle_step_2(state: expy.ExpState):
    print("Middle step 2")
    print(state["test"])

def finalization_step(state: expy.ExpState):
    print("Finalization step")
    print(state["test"])

builder = expy.ExpStep("Init step", init_step)
builder = builder.then(expy.ExpStep("Middle step 1", middle_step_1))
builder = builder.branch(expy.ExpStep("Middle step 2", middle_step_2))
builder = builder.then(expy.ExpStep("Finalization step", finalization_step))

suite = expy.ExpSuite(None)#"/home/frederic/Temporary/expy")
suite.queue_experiment("Experiment 1", builder)
suite.run()