from typing import Any, Dict, Optional, List, Tuple

from .node import Node

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
