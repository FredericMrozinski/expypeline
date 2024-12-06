# ExPypeline
A Python framework to modularize, log, and run scientific experiments.

## Key features:
- [x] Modularize experiments to "steps" such that multiple experiments can share common code blocks.
- [x] Easily define pipelines by chaining experiment steps.
- [x] Run experiments robustly through advanced error handling such that experiment steps keep running where possible.
- [x] Clear logging giving insights into the current position in the pipeline.
- [x] Experiment values are automatically saved among other details such as step runtimes. 
- [x] Conveniently share information across modules with experiment states.
- [x] Experiment outputs are cleanly organized on your hard-drive.

## Missing features
- [ ] Multi-threaded pipeline. Experiments are currently run sequentially.