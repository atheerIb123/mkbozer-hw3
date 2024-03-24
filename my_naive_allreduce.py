import numpy as np
from mpi4py import MPI
from time import time
    

def allreduce(send, recv, comm, op):
    """ Naive all reduce implementation

    Parameters
    ----------
    send : numpy array
        the array of the current process
    recv : numpy array
        an array to store the result of the reduction. Of same shape as send
    comm : MPI.Comm
    op : associative commutative binary operator
    """
    
    size = comm.Get_size()
    rank = comm.Get_rank()

    for pid in range(size):
        if pid != rank:
            comm.Send(send, dest=pid, tag=5)
        else:
            np.copyto(recv, send)

            for _ in range(size - 1):
                curr_recv = np.empty_like(recv)
                comm.Recv(curr_recv, source=MPI.ANY_SOURCE, tag=5)            
                for index in range(np.size(recv)):
                     recv[index] = op(recv[index], curr_recv[index])