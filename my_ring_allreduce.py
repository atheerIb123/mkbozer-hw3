import numpy as np
import mpi4py
mpi4py.rc(initialize=False, finalize=False)
from mpi4py import MPI

def ringallreduce(send, recv, comm, op):
    """ ring all reduce implementation
    You need to use the algorithm shown in the lecture.

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    rank = comm.Get_rank()
    size = comm.Get_size()
    np.copyto(recv, send)
    recv_chunks = np.array_split(recv, size)
    current_segment = rank
    prev_segment = None

    left_rank = rank - 1
    right_rank = rank + 1

    if rank <= 0:
        left_rank = size - 1
    
    if rank >= size - 1:
        right_rank = 0 

    for _ in range(size):
        if current_segment != 0: 
            prev_segment = current_segment - 1
        else:
            prev_segment = size - 1

        temp_buffer = np.empty_like(recv_chunks[prev_segment])
        request_recv = comm.Irecv(temp_buffer, source=left_rank, tag=1)
        process_arr = recv_chunks[current_segment]
        request_send = comm.Isend(process_arr, dest=right_rank, tag=1)
        request_recv.Wait()
        request_send.Wait()

        recv_chunks[prev_segment] = op(recv_chunks[prev_segment], temp_buffer)
        current_segment = (current_segment - 1) % size
    
    current_segment = right_rank

    for _ in range(size):
        if current_segment != 0: 
            prev_segment = current_segment - 1
        else:
            prev_segment = size - 1
        
        request_recv = comm.Irecv(recv_chunks[prev_segment], source=left_rank, tag=1)
        request_send = comm.Isend(recv_chunks[current_segment], dest=right_rank, tag=1)
        request_send.Wait()
        request_recv.Wait()
        current_segment = (current_segment - 1) % size
    
    np.copyto(recv, np.concatenate(recv_chunks))