# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
Lock-Free SPSC Queue - Cython implementation with C-level performance.

Uses memory barriers and atomic-like operations for sub-10μs latency.
This implementation releases the GIL during spin-wait operations.

Performance target: <5μs put/get operations.
"""

from cpython.mem cimport PyMem_Malloc, PyMem_Free
from cpython.ref cimport PyObject, Py_INCREF, Py_DECREF
from libc.stdint cimport uint64_t, int64_t
from libc.string cimport memset

cdef extern from "stdatomic.h" nogil:
    # C11 atomic operations
    ctypedef enum memory_order:
        memory_order_relaxed
        memory_order_consume
        memory_order_acquire
        memory_order_release
        memory_order_acq_rel
        memory_order_seq_cst

    void atomic_thread_fence(memory_order order)


cdef class FastSPSCQueue:
    """
    Ultra-fast Single-Producer Single-Consumer lock-free queue.

    Uses cache-line padding to avoid false sharing and memory barriers
    for correct ordering across CPU cores.

    Warning:
        ONLY safe with exactly ONE producer thread and ONE consumer thread.
        Multiple producers or consumers will cause data corruption.

    Example:
        >>> queue = FastSPSCQueue(1024)
        >>> queue.put(my_message)
        >>> msg = queue.get()
    """

    cdef:
        # Separate cache lines for head and tail to avoid false sharing
        # Typical cache line is 64 bytes
        uint64_t _head_padded[8]  # head + padding
        uint64_t _tail_padded[8]  # tail + padding

        uint64_t _capacity
        uint64_t _mask
        PyObject** _buffer
        bint _shutdown

    def __cinit__(self, uint64_t capacity=4096):
        """Initialize the queue with given capacity (rounded to power of 2)."""
        # Round up to power of 2
        cdef uint64_t cap = 1
        while cap < capacity:
            cap <<= 1

        self._capacity = cap
        self._mask = cap - 1

        # Allocate buffer for PyObject pointers
        self._buffer = <PyObject**>PyMem_Malloc(cap * sizeof(PyObject*))
        if self._buffer == NULL:
            raise MemoryError("Failed to allocate queue buffer")

        # Zero initialize
        memset(self._buffer, 0, cap * sizeof(PyObject*))

        # Initialize head/tail (in padded arrays, index 0 is the actual value)
        self._head_padded[0] = 0
        self._tail_padded[0] = 0
        self._shutdown = False

    def __dealloc__(self):
        """Free buffer memory and release references."""
        cdef uint64_t i
        cdef PyObject* obj

        if self._buffer != NULL:
            # Release all held references
            for i in range(self._capacity):
                obj = self._buffer[i]
                if obj != NULL:
                    Py_DECREF(<object>obj)
            PyMem_Free(self._buffer)
            self._buffer = NULL

    @property
    def capacity(self):
        """Get buffer capacity."""
        return self._capacity

    @property
    def size(self):
        """Get current size (approximate)."""
        return (self._tail_padded[0] - self._head_padded[0]) & self._mask

    @property
    def empty(self):
        """Check if empty."""
        return self._head_padded[0] == self._tail_padded[0]

    @property
    def full(self):
        """Check if full."""
        return ((self._tail_padded[0] + 1) & self._mask) == self._head_padded[0]

    cpdef bint put(self, object item) except False:
        """
        Put an item into the queue (non-blocking).

        Args:
            item: Python object to enqueue.

        Returns:
            True if successful, False if full.
        """
        cdef uint64_t tail = self._tail_padded[0]
        cdef uint64_t next_tail = (tail + 1) & self._mask
        cdef uint64_t head = self._head_padded[0]

        # Check if full
        if next_tail == head:
            return False

        # Store item (increment reference)
        Py_INCREF(item)
        self._buffer[tail] = <PyObject*>item

        # Release barrier - ensures item is visible before tail update
        with nogil:
            atomic_thread_fence(memory_order_release)

        # Update tail
        self._tail_padded[0] = next_tail
        return True

    cpdef object get(self):
        """
        Get an item from the queue (non-blocking).

        Returns:
            Item or None if empty.
        """
        cdef uint64_t head = self._head_padded[0]
        cdef uint64_t tail = self._tail_padded[0]
        cdef PyObject* obj
        cdef object result

        # Check if empty
        if head == tail:
            return None

        # Acquire barrier - ensures we see the stored item
        with nogil:
            atomic_thread_fence(memory_order_acquire)

        # Load item
        obj = self._buffer[head]
        if obj == NULL:
            return None

        # Clear slot
        self._buffer[head] = NULL

        # Convert to Python object (steals reference)
        result = <object>obj
        Py_DECREF(result)  # Balance the INCREF in put()

        # Update head
        self._head_padded[0] = (head + 1) & self._mask
        return result

    cpdef bint put_wait(self, object item, double timeout_sec=1.0) except False:
        """
        Put with spin-wait (releases GIL during spin).

        Args:
            item: Item to enqueue.
            timeout_sec: Maximum wait time in seconds.

        Returns:
            True if successful, False if timeout.
        """
        cdef uint64_t tail, next_tail, head
        cdef double start_time, elapsed
        cdef int spins = 0
        cdef int max_spins = 1000

        # Import time module for timing
        import time
        start_time = time.perf_counter()

        while True:
            tail = self._tail_padded[0]
            next_tail = (tail + 1) & self._mask
            head = self._head_padded[0]

            if next_tail != head:
                # Space available
                Py_INCREF(item)
                self._buffer[tail] = <PyObject*>item

                with nogil:
                    atomic_thread_fence(memory_order_release)

                self._tail_padded[0] = next_tail
                return True

            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout_sec:
                return False

            if self._shutdown:
                return False

            # Brief spin with GIL released
            spins += 1
            if spins >= max_spins:
                spins = 0
                # Yield to other threads
                with nogil:
                    pass  # This releases GIL briefly

    cpdef object get_wait(self, double timeout_sec=1.0):
        """
        Get with spin-wait (releases GIL during spin).

        Args:
            timeout_sec: Maximum wait time in seconds.

        Returns:
            Item or None if timeout.
        """
        cdef uint64_t head, tail
        cdef PyObject* obj
        cdef object result
        cdef double start_time, elapsed
        cdef int spins = 0
        cdef int max_spins = 1000

        import time
        start_time = time.perf_counter()

        while True:
            head = self._head_padded[0]
            tail = self._tail_padded[0]

            if head != tail:
                # Item available
                with nogil:
                    atomic_thread_fence(memory_order_acquire)

                obj = self._buffer[head]
                if obj != NULL:
                    self._buffer[head] = NULL
                    result = <object>obj
                    Py_DECREF(result)
                    self._head_padded[0] = (head + 1) & self._mask
                    return result

            # Check timeout
            elapsed = time.perf_counter() - start_time
            if elapsed >= timeout_sec:
                return None

            if self._shutdown:
                return None

            # Brief spin
            spins += 1
            if spins >= max_spins:
                spins = 0
                with nogil:
                    pass

    cpdef void clear(self):
        """Clear all items from the queue."""
        cdef PyObject* obj
        cdef uint64_t head = self._head_padded[0]
        cdef uint64_t tail = self._tail_padded[0]

        while head != tail:
            obj = self._buffer[head]
            if obj != NULL:
                Py_DECREF(<object>obj)
                self._buffer[head] = NULL
            head = (head + 1) & self._mask

        self._head_padded[0] = head

    cpdef void shutdown(self):
        """Signal shutdown to unblock waiters."""
        self._shutdown = True

    def __len__(self):
        """Get current size."""
        return (self._tail_padded[0] - self._head_padded[0]) & self._mask

    def __repr__(self):
        return f"FastSPSCQueue(size={len(self)}/{self._capacity})"


# Convenience function to check if Cython extension is available
def is_cython_available():
    """Check if Cython extension is loaded."""
    return True
