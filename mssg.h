/*
 * Copyright (c) 2016 UChicago Argonne, LLC
 * Copyright (c) 2018 Carnegie Mellon University.
 *
 * See COPYRIGHT in top-level directory.
 */

#pragma once

#include <mercury.h>
#include <mpi.h>

/*
 * MPI-based init for mssg (extracted from an old version of ANL ssg)
 */

#ifdef __cplusplus
extern "C" {
#endif

struct mssg; /* forward decl */
typedef struct mssg mssg_t;

/**
 * mssg_init_mpi: init mssg using MPI!
 *
 * @param hgcl the mercury class to use
 * @param comm the MPI communicator to use
 * @return pointer to our structure
 */
mssg_t* mssg_init_mpi(hg_class_t* hgcl, MPI_Comm comm);

/**
 * mssg_finalize: free mssg state
 *
 * @param s the mssg to finalize and free
 */
void mssg_finalize(mssg_t* s);

/**
 * mssg_get_rank: get my rank
 *
 * @param s our mssg
 * @return our rank
 */
int mssg_get_rank(const mssg_t* s);

/**
 * mssg_get_count: get the number of participants (e.g. world size)
 *
 * @param s our mssg
 * @return world size
 */
int mssg_get_count(const mssg_t* s);

/**
 * mssg_get_addr: get addres for group member of given rank
 *
 * @param s our mssg
 * @return mercury address
 */
hg_addr_t mssg_get_addr(const mssg_t* s, int rank);

/**
 * mssg_get_addr_str: get string addr for given rank
 *
 * @param s our mssg
 * @return mercury address string
 */
const char* mssg_get_addr_str(const mssg_t* s, int rank);

/**
 * mssg_lookup: lookup all the addresses of interest in one big go
 *
 * @param s our mssg
 * @parma hgctx mercury context to use (XXX: init caches class, but not ctx)
 * @return success or error code
 */
hg_return_t mssg_lookup(mssg_t* s, hg_context_t* hgctx);

#ifdef __cplusplus
}
#endif
