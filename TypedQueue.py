from collections import deque


class TypedQueue:
    def __init__(self):
        self.queue = deque()

    def push(self, item_type, item):
        self.queue.append((item_type, item))

    def pop(self, count: int = 1, item_type=None):
        popped_items = []
        temp_queue = deque()

        if count < 1:
            raise Exception("Queue count must be greater than 0")

        if item_type is None:
            item_type, item = self.queue.popleft()

            # Iterate through the queue to find and pop the specified items
            while self.queue and len(popped_items) < count - 1:
                current_type, item = self.queue.popleft()
                if current_type == item_type:
                    popped_items.append(item)
                else:
                    temp_queue.append((current_type, item))
        else:
            # Iterate through the queue to find and pop the specified items
            while self.queue and len(popped_items) < count:
                current_type, item = self.queue.popleft()
                if current_type == item_type:
                    popped_items.append(item)
                else:
                    temp_queue.append((current_type, item))

        # Put the remaining items back into the queue
        self.queue = temp_queue

        return popped_items


# Example usage
queue = TypedQueue()
queue.push("Type3", "Type3_1")
queue.push("Type2", "Type2_1")
queue.push("Type1", "Type1_1")
queue.push("Type2", "Type2_2")

print("Initial queue:", [(item_type, item) for item_type, item in queue.queue])
# Output: Initial queue: [('Type3', 'Type3_1'), ('Type2', 'Type2_1'), ('Type1', 'Type1_1'), ('Type2', 'Type2_2')]

popped = queue.pop(2, "Type2")
print("Popped items:", popped)
# Output: Popped items: ['Type2_1', 'Type2_2']

print("Queue after popping:", [(item_type, item) for item_type, item in queue.queue])
# Output: Queue after popping: [('Type3', 'Type3_1'), ('Type1', 'Type1_1')]
