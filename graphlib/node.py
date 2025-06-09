from typing import Any, Dict, Optional, NamedTuple, Union
from abc import ABC, abstractmethod

class Transition(NamedTuple):
    """
    Represents a transition between nodes with an output string.
    Supports destructuring like a normal tuple.
    """
    output: str
    node: 'Node'

class Node(ABC):
    """
    Abstract base class for all nodes in the graph.
    Each node has pre-execution, execution, and post-execution phases.
    """
    def __init__(self, name: str):
        self.name = name
        self.edges: Dict[str, 'Node'] = {}  # Maps output strings to next nodes
        
    def __rshift__(self, other: Union['Node', Transition]) -> 'Node':
        """
        Overloaded >> operator for connecting nodes.
        Usage:
            node1 >> node2  # Uses "default" as output string
            node1 >> Transition("output_string", node2)  # Uses custom output string
        
        Args:
            other: Either a Node (uses "default" output) or Transition(output_string, node)
            
        Returns:
            Node: The destination node (enables chaining)
        """
        if isinstance(other, Transition):
            output, next_node = other
        else:
            output, next_node = "default", other
        self.add_edge(next_node, output)
        return next_node
        
    def add_edge(self, next_node: 'Node', output: str = "default") -> None:
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
