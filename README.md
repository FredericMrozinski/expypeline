# ExPypeline
A Python framework to modularize, log, and run scientific experiments in a reproducible and stable fashion.

*Disclaimer: ExPypeline is currently still in active development and thus in a beta-state. A pip release is planned.*

## Why ExPypeline?

Running sound scientific experiments require thorough code robustness. This is especially important when experiments require
intense computational resources as re-running failed experiments is time- or cost-consuming. ExPypeline aims at minimizing
this risk of failure by serving as a strong framework to conduct experiments in Python. 

Let's look at an example use-case: Suppose we would like to train multiple neural networks on multiple given datasets to 
compare their performances. A typical script to achieve this would look something like:

```python
# ... define prerequisuites

# Run experiments
for dataset in datasets:
    for model in models:
        try:
            # Experiments should be reproducible
            reset_global_random_generator()
            experiment_result = train_model(model, dataset)
        # The script should not end if one step fails
        except Exception as e:
            experiment_result = "failure"
        
        # Exporting the results
        write_results_to_disk(experiment_results)

# ...
```
While the above code would achieve the desired, it contains boilerplate code that makes writing and reading scripts cumbersome
and potentially error-prone (e.g. if we forget to reset randomness).

ExPypeline enables the user to skip worrying about such code snippets by modularizing experiments. While we will show how
to create experiment steps in detail, we first point out that the above code could simply be reduced to:

```python
import expypeline as ep

# ... define prerequisuites

# Build experiments
dataset_pipeline = ep.Empty()
for dataset_loader in dataset_loaders:
    dataset_pipeline = dataset_pipeline.branch(dataset_loader)
model_pipeline = ep.Empty()
for model_loader in model_loaders:
    model_pipeline = model_pipeline.branch(model_loader)
full_pipeline = dataset_pipeline.then(model_pipeline).then(train_model)

# Run experiments
suite.queue_experiment("Comparing Models", full_pipeline)
suite.run()

# ...
```

What have we gained here? We have now defined experiments as "steps" and "pipelines" which are the modular units in
ExPypeline. Those can be branched, concatenated, and most importantly executed using ExPypeline's ability to take care
of stably coordinating the run.

In fact, ExPypeline can do much more than what is shown here. 
- ExPypeline automatically saves selected variable values used in experiment steps to the hard-disk in easy-to-query format. Thus, many relevant details are available by default.
- ExPypeline ensures clean logging which is automatically exported to your disk.
- ExPypeline organizes outputs cleanly on your hard-drive. Thus, you don't have to remember your own incomprehensible directory-structures ;-)
- And as explained above: 
  - ExPypeline ensures that failed experiment steps don't crash the entire script if unnecessary.
  - ExPypeline ensures reproducibility even if the order of (branched) step executions changes or a step is preceded by a failed step.

## Usage 

At this moment, ExPypeline is not yet added to pip (planned soon). Until then, download the file `expypeline.py` and add it
to your project. Just import it by

```python
import expypeline as ep
```

## Roadmap

This project is still in its infancy and many more features are planned.

### Stochastic runs
Any non-deterministic experiment should ideally be conducted in multiple runs and report statistic results. We would like to add this feature
to simply be able to call

```python
suite.run(stochastic_runs=10)
```

Reading experiment results could already come with precomputed statistics:

```python
# Non stochastically
model_score = experiment_result["test_score"]
# > 0.953

# Stochastically
model_score = experiment_result["test_score"]
avg = model_score.avg
# > 0.957
stddev = model_score.stddev
# > 0.012
max = model_score.max
# > 0.982
min = model_score.min
# > 0.937
# etc...
```

The goal is to enable this without adapting any experiment code.

### Parallel runs

Currently, all experiments, pipelines, and steps are executed sequentially. Especially large experiments, hwoever,
often run on clusters allowing for high parallelizing. We would love to leverage this ability and bring it to ExPypeline.

## Contributing

ExPypline is still waiting for many features to be implemented! Feel free to contact `frederic.mrozinski@tuta.com` for more information.

## License

ExPypeline is MIT licensed. So go play with it, use it, copy it, sell it, have fun with it!