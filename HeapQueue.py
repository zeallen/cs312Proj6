# This class stores a heap and pointer array both of size n (at worst), so
# space complexity is O(2n) = O(n)
class heap_queue():

    def __init__(self):
        self.pointer_array = []
        self.heap = []
        self.node_id = -1

    def make_queue(self, nodes):
        self.pointer_array = [] # Arranged so that node_id is the index, and the node's location in heap is the value
        self.heap = []
        for node in nodes:
            self.insert(node)

    def size(self):
        return len(self.heap)

    # The only thing that isn't constant time in insert() is bubble_up, so since that is O(logn),
    # insert is too
    def insert(self, node):
        self.node_id += 1
        node.node_id = self.node_id
        self.pointer_array.append(node.node_id)  
        self.heap.append(node)
        self.bubble_up(len(self.heap)-1)

    # Delete min is similar to insert() in that it is O(logn) because of bubble_down()
    def delete_min(self):
        last_idx = len(self.heap) -1 
        top_node = self.heap[0]
        self.pointer_array[self.heap[last_idx].node_id] = 0
        #self.pointer_array[top_node.node_id] = -1  # Not necessary, will probably delete

        self.heap[0] = self.heap[last_idx]
        self.heap.pop(last_idx)

        self.bubble_down()
        return top_node

    # Also similar to insert() in that it is O(logn) because of bubble_up()
    def decrease_key(self, node):
        idx = self.pointer_array[node.node_id]
        self.bubble_up(idx)

    # Total time complexity is O(logn)
    # Space complexity is constant
    def bubble_down(self):
        parent_idx = 0
        # This while loop runs at most log(n) times since it
        # is based on the depth of the heap 
        while (2 * parent_idx + 1) <= len(self.heap) - 1:
            child_idx1 = 2 * parent_idx + 1
            child_idx2 = child_idx1 + 1

            child_node = None
            child_idx = None

            parent_node = self.heap[parent_idx]
            child_node1 = self.heap[child_idx1]
            try:
                child_node2 = self.heap[child_idx2]
                if child_node2.priority_val < child_node1.priority_val:
                    child_node = child_node2
                    child_idx = child_idx2
                else:
                    raise ValueError
            except Exception:
                child_node = child_node1
                child_idx = child_idx1

            if self.heap[parent_idx].priority_val <= self.heap[child_idx].priority_val:
                return
            self.swap_nodes(parent_idx, child_idx, child_node, parent_node)
            parent_idx = child_idx

    # Similar to bubble_down(), it runs in O(logn) time and the space complexity is
    # O(1)
    def bubble_up(self, child_idx):
        while child_idx != 0:
            parent_idx = child_idx // 2
            parent_node = self.heap[parent_idx]
            child_node = self.heap[child_idx]

            if self.heap[parent_idx].priority_val <= self.heap[child_idx].priority_val:
                return

            self.swap_nodes(parent_idx, child_idx, child_node, parent_node)
            child_idx = parent_idx

    # Helper function to swap two nodes and update the heap and pointer arrays
    # It is constant time
    def swap_nodes(self, parent_idx, child_idx, child_node, parent_node):
        tmp = self.heap[parent_idx]
        self.heap[parent_idx] = self.heap[child_idx]
        self.heap[child_idx] = tmp

        self.pointer_array[child_node.node_id] = parent_idx
        self.pointer_array[parent_node.node_id] = child_idx
       
    # O(1) since is just a comparison
    def is_empty(self):
        return self.heap == []