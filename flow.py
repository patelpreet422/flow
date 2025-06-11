from typing import Any, Dict, Optional, NamedTuple, Union
from abc import ABC, abstractmethod

from typing import Any, Dict, Optional, Tuple

class EdgeBuilder:
    def __init__(self, node: 'Node', output: str):
        self.node = node
        self.output = output

    def __rshift__(self, other: 'Node') -> 'Node':
        self.node.add_edge(other, self.output)
        return other
    
class Node(ABC):
    """
    Abstract base class for all nodes in the graph.
    Each node has pre-execution, execution, and post-execution phases.
    """
    def __init__(self, name: str):
        self.name = name
        self.edges: Dict[str, 'Node'] = {}  # Maps output strings to next nodes
        
    def __sub__(self, output: str) -> 'EdgeBuilder':
        return EdgeBuilder(self, output)

    def __rshift__(self, other: 'Node') -> 'Node':
        self.add_edge(other, "default")
        return other

    def add_edge(self, next_node: 'Node', output: str) -> None:
        """
        Add an edge from this node to another node based on output string.
        
        Args:
            next_node: The destination node
            output: The string output that triggers this edge. Defaults to "default"
        """
        self.edges[output] = next_node

    @abstractmethod
    async def pre(self, context: Dict[str, Any]) -> None:
        """
        Pre-execution phase - runs before exec.
        
        Args:
            context: Shared context dictionary for the flow
        """
        pass
    
    @abstractmethod
    async def exec(self, context: Dict[str, Any]) -> None:
        """
        Main execution phase.
        
        Args:
            context: Shared context dictionary for the flow
        """
        pass
    
    async def post(self, context: Dict[str, Any]) -> str:
        """
        Post-execution phase - runs after exec.
        Returns the output string that determines the next node.
        Default implementation returns "default".
        
        Args:
            context: Shared context dictionary for the flow
            
        Returns:
            str: Output string determining the next node to execute
        """
        return "default"

    def get_next_node(self, output: str) -> Optional['Node']:
        """
        Get the next node based on the output string.
        
        Args:
            output: Output string from post execution
            
        Returns:
            Optional[Node]: The next node to execute or None if no matching edge
        """
        return self.edges.get(output)


class Flow(Node):
    """
    A Flow is a special type of Node that can contain other nodes.
    It manages execution flow between nodes and maintains shared context.
    """
    def __init__(self, name: str, root_node: Optional[Node] = None):
        super().__init__(name)
        self.root_node = root_node
        self.context: Dict[str, Any] = {}
        self._last_output: str = ""

    async def start(self, initial_context: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Start the flow execution from the root node.
        
        Args:
            initial_context: Optional initial context to start the flow with
            
        Returns:
            Tuple[str, Dict[str, Any]]: (last node output, final context)
            
        Raises:
            ValueError: If no root node is defined
        """
        if initial_context is not None:
            self.context.update(initial_context)
            
        if not self.root_node:
            raise ValueError("Cannot start flow: no root node defined")
            
        current_node = self.root_node
        while current_node:
            # Execute the node's lifecycle methods
            await current_node.pre(self.context)
            await current_node.exec(self.context)
            output = await current_node.post(self.context)
            self._last_output = output
            
            # Check for flow termination
            if output.startswith("terminate:"):
                # Strip the terminate prefix and keep the actual output
                self._last_output = output[10:]
                break
                
            # Get the next node based on the output
            current_node = current_node.get_next_node(output)
            
            # If no matching edge found, end the flow
            if not current_node:
                break
                
        return self._last_output, self.context
    
    async def pre(self, context: Dict[str, Any]) -> None:
        """
        Pre-execution phase for the flow.
        Inherits shared context from parent flow.
        
        Args:
            context: Parent flow's context
        """
        self.context.update(context)
        
    async def exec(self, context: Dict[str, Any]) -> None:
        """
        Execute the flow as a subflow within another flow.
        
        Args:
            context: Parent flow's context
        """
        await self.start(context)
        
    async def post(self, context: Dict[str, Any]) -> str:
        """
        Post-execution phase for the flow.
        Updates parent context with any changes from this flow.
        Returns the last output from the final executed node.
        
        Args:
            context: Parent flow's context
            
        Returns:
            str: The last output from the final executed node
        """
        context.update(self.context)
        return self._last_output
