# Observability

NanoNet makes neural-network internals observable through model inspection,
execution tracing, autograd graph inspection, and evidence-based diagnostics.

| API | Purpose |
| --- | --- |
| `model.inspect()` | Model structure, parameters, shapes, and statistics |
| `model.trace(x)` | Actual module execution order for an input |
| `tensor.graph()` | Autograd operation / dependency graph |
| `model.diagnose(x)` | Numerical and optimization warning checks |

All four APIs share the same output convention:

```python
report = model.inspect(verbose=False)   # structured report, no print
print(report)                           # formatted text via __str__
data = report.to_dict()                 # JSON-compatible metadata
```

## Model Inspection

```python
report = model.inspect()
report = model.inspect(x)               # includes runtime shapes / activations
```

Answers: *what does this model contain?*

## Execution Tracing

```python
trace = model.trace(x)
for step in trace.steps:
    print(step.module_name, step.outputs[0].shape)
```

Answers: *what actually executed for this input?*

`trace.output` retains the forward result so autograd can continue. Timing
includes instrumentation overhead and is for debugging, not benchmarking.

## Computation Graphs

```python
prediction = model(x)
loss = criterion(prediction, target)
graph = loss.graph()
```

Answers: *which differentiable tensor operations produced this tensor?*

Graph IDs (`T0`, `P0`, `OP0`) are local to each `graph()` call and need not
match `trace()` IDs.

## Diagnostics

```python
report = model.diagnose()               # parameters + existing gradients
report = model.diagnose(x)              # also analyzes activations
```

Answers: *is anything about this model state suspicious or unstable?*

`diagnose()` never calls `backward()`. NaN/Inf checks are definitive;
vanishing gradients, dead ReLU, and saturation are conservative heuristics.
Thresholds live in `DiagnosticThresholds`.

## Integrated Workflow

```python
model.inspect(x)

trace = model.trace(x)

prediction = model(x)
loss = criterion(prediction, target)
loss.graph()

loss.backward()
model.diagnose(x)
```

See `examples/observability_workflow.py`.
