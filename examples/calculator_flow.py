from typing import Dict, Any

from graphlib.node import Node
from graphlib.flow import Flow

class InputNode(Node):
    """Node that sets up initial numbers in the context."""
    async def pre(self, context: Dict[str, Any]) -> None:
        context['numbers'] = [5, 3]

    async def exec(self, context: Dict[str, Any]) -> None:
        pass

class AddNode(Node):
    """Node that adds two numbers."""
    async def pre(self, context: Dict[str, Any]) -> None:
        self.nums = context['numbers']

    async def exec(self, context: Dict[str, Any]) -> None:
        context['result'] = self.nums[0] + self.nums[1]

class OutputNode(Node):
    """Node that prints the final result."""
    async def pre(self, context: Dict[str, Any]) -> None:
        pass

    async def exec(self, context: Dict[str, Any]) -> None:
        print(f"Final result: {context['result']}")

async def main():
    # Create nodes
    input_node = InputNode("input")
    add_node = AddNode("add")
    output_node = OutputNode("output")

    # Create calculation subflow
    calc_flow = Flow("calculator", root_node=input_node)

    # Connect nodes using >> operator
    input_node >> add_node >> output_node

    # Start the flow and get final output
    output, final_context = await calc_flow.start({})
    print(f"Flow completed with output: {output}")
    print(f"Final context: {final_context}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
