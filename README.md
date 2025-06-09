# Graph Flow Library

A Python library for creating and executing graph-based workflows with context sharing and subflows.

## Features

- Create nodes with pre-execution, execution, and post-execution phases
- Connect nodes using operator overloading (>>)
- Share context between nodes in a flow
- Support for subflows (flows that can be used as nodes)
- Flow control with conditional branching and early termination
- Async execution model

## Core Components

### Node

The base class for all nodes in the graph. Each node has three execution phases:
- `pre()`: Pre-execution setup
- `exec()`: Main execution logic
- `post()`: Post-execution cleanup and output determination

```python
from src.node import Node

class MyNode(Node):
    async def pre(self, context):
        # Setup phase
        pass
        
    async def exec(self, context):
        # Main execution
        context['my_result'] = 42
        
    async def post(self, context):
        # Return string determines next node
        return "success"
```

### Flow

A special type of node that can contain and manage other nodes. Flows can be used as nodes within other flows.

```python
from src.flow import Flow

# Create a flow with a root node
flow = Flow("my_flow", root_node=first_node)

# Add nodes to the flow
flow.add_node(first_node)
flow.add_node(second_node)

# Connect nodes using >> operator
first_node >> (second_node, "success")

# Start the flow with initial context
output, final_context = await flow.start({"initial": "data"})
```

## Node Connection

Nodes can be connected using the overloaded >> operator:

```python
# Connect nodes with output conditions
node1 >> (node2, "success")
node1 >> (node3, "error")

# Chain multiple connections
node1 >> (node2, "success") >> (node3, "complete")
```

## Flow Control

### Early Termination

You can terminate a flow early by returning a string prefixed with "terminate:":

```python
async def post(self, context):
    if some_condition:
        return "terminate:success"  # Flow ends here
    return "continue"  # Flow continues to next node
```

### Flow Output

The `start()` method returns both the final output and context:

```python
output, final_context = await flow.start(initial_context)
print(f"Flow ended with: {output}")
print(f"Final state: {final_context}")
```

## Example

See [examples/calculator_flow.py](examples/calculator_flow.py) for a complete example showing:
- Node creation and connection using >>
- Context sharing between nodes
- Subflow usage with early termination
- Conditional branching based on node outputs

```python
# Create and connect nodes
input_node = InputNode("input")
calc_flow = CalculationFlow("calculator")
output_node = OutputNode("output")

# Connect using >> operator
input_node >> (calc_flow, "calculate")
calc_flow >> (output_node, "success")

# Execute the flow
output, context = await main_flow.start()
```

## Installation

1. Clone the repository
2. Install dependencies (if any)
3. Import and use in your project

## Requirements

- Python 3.7+
- asyncio support
