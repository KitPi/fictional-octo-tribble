from collections import deque


class TypedQueue:
    def __init__(self):
        self.queue = deque()

    def push(self, item_type, item):
        self.queue.append((item_type, item))

    def pop(self, count: int = 1, item_type=None):
        popped_items = []
        temp_queue = deque()
        # size = len(self.queue)

        if count < 1:
            raise ValueError("Queue count must be greater than 0")

        if self.queue and item_type is None:
            item_type, item = self.queue.popleft()
            popped_items.append({"type": item_type, "item": item})

        # Iterate through the queue to find and pop the specified items
        # iter = 0
        while self.queue and len(popped_items) < count:
            current_type, item = self.queue.popleft()
            if current_type == item_type:
                popped_items.append({"type": item_type, "item": item})
            else:
                temp_queue.append((current_type, item))
            # iter += 1

        # Put the remaining items back into the queue
        while temp_queue:
            x, y = temp_queue.pop()
            self.queue.appendleft((x, y))

        # self.queue = temp_queue

        return popped_items


# +++++ ====== +++++
import unittest
from collections import deque


class TestTypedQueue(unittest.TestCase):
    def setUp(self):
        self.queue = TypedQueue()

    def test_push_and_pop(self):
        self.queue.push(0, "Item1")
        self.queue.push(1, "Item2")
        self.queue.push(0, "Item3")
        popped = self.queue.pop(2, 0)
        self.assertEqual(popped, ["Item1", "Item3"])
        self.assertEqual(len(self.queue.queue), 1)  # Only "Item2" should remain

    def test_pop_all(self):
        self.queue.push(0, "Item1")
        self.queue.push(1, "Item2")
        self.queue.push(0, "Item3")
        popped = self.queue.pop(3, 0)
        self.assertEqual(popped, ["Item1", "Item3"])
        self.assertEqual(len(self.queue.queue), 1)  # Only "Item2" should remain

    def test_pop_nonexistent_type(self):
        self.queue.push(0, "Item1")
        self.queue.push(0, "Item2")
        popped = self.queue.pop(1, 1)
        self.assertEqual(popped, [])
        self.assertEqual(len(self.queue.queue), 2)  # Both items should remain

    def test_pop_more_than_available(self):
        self.queue.push(0, "Item1")  # Only one item of type 0
        popped = self.queue.pop(2, 0)
        self.assertEqual(popped, ["Item1"])
        self.assertEqual(len(self.queue.queue), 0)  # Queue should be empty

    def test_pop_preserves_order(self):
        self.queue.push(0, "Item1")
        self.queue.push(1, "Item2")
        self.queue.push(0, "Item3")
        self.queue.push(1, "Item4")
        popped = self.queue.pop(2, 0)
        self.assertEqual(popped, ["Item1", "Item3"])
        self.assertEqual(
            len(self.queue.queue), 2
        )  # "Item2" and "Item4" should remain in order

    def test_pop_with_none_type(self):
        self.queue.push(0, "Item1")
        self.queue.push(1, "Item2")
        self.queue.push(0, "Item3")
        popped = self.queue.pop(2)
        self.assertEqual(popped, ["Item1", "Item3"])
        self.assertEqual(len(self.queue.queue), 1)  # Only "Item2" should remain

    def test_pop_empty_queue(self):
        popped = self.queue.pop(1)
        self.assertEqual(popped, [])
        self.assertEqual(len(self.queue.queue), 0)  # Queue should remain empty

    def test_invalid_count(self):
        with self.assertRaises(ValueError):
            self.queue.pop(0)


if __name__ == "__main__":
    unittest.main()
