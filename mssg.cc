/*
 * Copyright (c) 2016 UChicago Argonne, LLC
 * Copyright (c) 2018 Carnegie Mellon University.
 *
 * See COPYRIGHT in top-level directory.
 */
#include <mssg.h>

#include <assert.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * mssg: the state structure we return to the caller.  the internal
 * structure for mssg_t is private to this file...
 */
struct mssg {
  hg_class_t* hgcl;  /* our mercury class, e.g. to free hd_addr_t */
  char** addr_strs;  /* addr strings ptr array, points into backing_buf */
  hg_addr_t* addrs;  /* array of mercury address pointers */
  void* backing_buf; /* addr_strs[] backing buffer */
  int num_addrs;     /* address array size */
  int backing_bufsz; /* the buffer size */
  int rank;          /* my rank in the comm we init'd with */
};

/*
 * mssg_state: tracks state for looking up hg address strings
 */
typedef struct mssg_state {
  hg_return_t ret;
  hg_addr_t addr;
  int* count;
} mssg_state_t;

/*
 * helper functions
 */
namespace {
/* abort with an error message. */
#define ABORT(msg) msg_abort(__FILE__, __LINE__, msg)
#define OUT_OF_MEMORY() ABORT("out-of-memory, cannot malloc")
void msg_abort(const char* f, int d, const char* msg) {
  fprintf(stderr, "=== FATAL === ");
  fprintf(stderr, "%s (%s:%d)", msg, f, d);
  fprintf(stderr, "\n");
  abort();
}

/*
 * setup_addr_str_list: create an array of pointers into a buffer
 * filled with null terminated strings.  use strlen() to obtain string
 * size as we walk through it.
 */
char** setup_addr_str_list(int num_addrs, char* buf) {
  char** result = (char**)malloc(num_addrs * sizeof(char*));
  if (result == NULL) {
    OUT_OF_MEMORY();
  } else {
    result[0] = buf;
  }
  for (int i = 1; i < num_addrs; i++) {
    char* a = result[i - 1];
    result[i] = a + strlen(a) + 1;
  }
  return result;
}

/*
 * mssg_cb: address lookup callback function
 */
hg_return_t mssg_cb(const struct hg_cb_info* info) {
  mssg_state_t* const out = (mssg_state_t*)info->arg;
  *out->count += 1;
  out->ret = info->ret;
  if (out->ret != HG_SUCCESS)
    out->addr = HG_ADDR_NULL;
  else
    out->addr = info->info.lookup.addr;

  return HG_SUCCESS;
}
}  // namespace

/*
 * API functions
 */
extern "C" {

/*
 * mssg_init_mpi: init mssg via MPI.  must call mssg_lookup() separately
 * to perform lookups of the address strings.  the recv must be
 * MPI_COMM_NULL if the caller rank is only a sender.
 */
mssg_t* mssg_init_mpi(hg_class_t* hgcl, MPI_Comm comm, MPI_Comm recv) {
  hg_addr_t self_hg_addr = HG_ADDR_NULL;
  char* self_hg_str = NULL;
  hg_size_t self_hg_size = 0;
  hg_addr_t* hg_addrs = NULL;
  hg_return_t hg_ret;

  char* buf = NULL; /* back space for all address strings */
  int* bufszs = NULL;
  int* displs = NULL;
  int total_bufsz = 0;
  int comm_size = 0;
  int comm_rank = 0;

  int my_addr_size = 0;
  char* my_addr = NULL;
  char** addr_strs = NULL;
  mssg_t* s = NULL;

  /* get my own hg_addr_t address, and my address string */
  if (hgcl != NULL && recv != MPI_COMM_NULL) {
    hg_ret = HG_Addr_self(hgcl, &self_hg_addr);
    if (hg_ret == HG_SUCCESS)
      hg_ret = HG_Addr_to_string(hgcl, NULL, &self_hg_size, self_hg_addr);
    if (hg_ret != HG_SUCCESS) {
      ABORT("fail to HG_Addr_self or HG_Addr_to_string");
    }

    /* this includes the null terminator at the end */
    self_hg_str = (char*)malloc(self_hg_size);
    if (self_hg_str == NULL) OUT_OF_MEMORY();
    hg_ret = HG_Addr_to_string(hgcl, self_hg_str, &self_hg_size, self_hg_addr);
    if (hg_ret != HG_SUCCESS) {
      ABORT("fail to HG_Addr_to_string");
    }
  }

  if (self_hg_str != NULL) {
    my_addr_size = (int)self_hg_size;
    my_addr = self_hg_str;
  } else {
    my_addr_size = 1;
    my_addr = (char*)"";
  }

  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  /* communicate the size for each address string */
  bufszs = (int*)malloc(comm_size * sizeof(int));
  if (bufszs == NULL) OUT_OF_MEMORY();
  bufszs[comm_rank] = my_addr_size;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, bufszs, 1, MPI_INT, comm);

  /* build the displacement list needed by MPI for all gather operations */
  displs = (int*)malloc((comm_size + 1) * sizeof(int));
  if (displs == NULL) {
    OUT_OF_MEMORY();
  } else {
    displs[0] = 0;
  }
  for (int i = 1; i < comm_size + 1; i++) {
    displs[i] = displs[i - 1] + bufszs[i - 1];
  }

  /* gather all address strings */
  total_bufsz = displs[comm_size]; /* total string length */
  buf = (char*)malloc(total_bufsz);
  if (buf == NULL) OUT_OF_MEMORY();
  MPI_Allgatherv(my_addr, my_addr_size, MPI_BYTE, buf, bufszs, displs, MPI_BYTE,
                 comm);

  /* setup the address list */
  addr_strs = setup_addr_str_list(comm_size, buf);
  hg_addrs = (hg_addr_t*)malloc(comm_size * sizeof(hg_addr_t));
  if (hg_addrs == NULL) {
    OUT_OF_MEMORY();
  } else {
    for (int i = 0; i < comm_size; i++) {
      hg_addrs[i] = HG_ADDR_NULL;
    }
  }

  /*
   * note it sets the locals to NULL, causing the free()'s in fini
   * to not do anything...
   */
  s = (mssg_t*)malloc(sizeof(mssg_t));
  if (s == NULL) OUT_OF_MEMORY();
  s->hgcl = NULL; /* to be set by mssg_lookup() */
  hg_addrs[comm_rank] = self_hg_addr;
  self_hg_addr = HG_ADDR_NULL;
  s->addrs = hg_addrs;
  hg_addrs = NULL;
  s->addr_strs = addr_strs;
  addr_strs = NULL;
  s->backing_bufsz = total_bufsz;
  s->backing_buf = buf;
  buf = NULL;
  s->num_addrs = comm_size;
  s->rank = comm_rank;

fini:
  if (self_hg_addr != HG_ADDR_NULL) HG_Addr_free(hgcl, self_hg_addr);
  free(self_hg_str);
  free(bufszs);
  free(displs);
  free(hg_addrs);
  free(addr_strs);
  free(buf);
  return s;
}

/*
 * mssg_lookup: lookup all addr strings to get hg_addr_t.  this may
 * establish connections to everything we lookup...
 */
hg_return_t mssg_lookup(mssg_t* s, hg_context_t* hgctx) {
  unsigned int hg_count = 0;
  mssg_state_t* outs = NULL;
  hg_return_t hg_ret;
  int posted = 0;
  int done = 0;

  s->hgcl = HG_Context_get_class(hgctx);
  outs = (mssg_state_t*)malloc(s->num_addrs * sizeof(mssg_state_t));
  if (outs == NULL) {
    OUT_OF_MEMORY();
  } else {
    for (int i = 1; i < s->num_addrs; i++) {
      const int r = (s->rank + i) % s->num_addrs;
      outs[r].ret = HG_SUCCESS;
      outs[r].addr = HG_ADDR_NULL;
      outs[r].count = &done;
      if (s->addr_strs[r][0] != 0) {
        hg_ret = HG_Addr_lookup(hgctx, &mssg_cb, &outs[r], s->addr_strs[r],
                                HG_OP_ID_IGNORE);
        if (hg_ret != HG_SUCCESS) {
          ABORT("fail to HG_Addr_lookup");
        } else {
          posted++;
        }
      }
    }
  }

  while (done < posted) {
    do {
      hg_ret = HG_Trigger(hgctx, 0, 1, &hg_count);
    } while (hg_ret == HG_SUCCESS && hg_count > 0);
    if (hg_ret != HG_SUCCESS && hg_ret != HG_TIMEOUT) {
      ABORT("fail to HG_Trigger");
    }

    hg_ret = HG_Progress(hgctx, 100);
    if (hg_ret != HG_SUCCESS && hg_ret != HG_TIMEOUT) {
      ABORT("fail to HG_Progress");
    }
  }

  for (int i = 1; i < s->num_addrs; i++) {
    const int r = (s->rank + i) % s->num_addrs;
    if (outs[r].ret != HG_SUCCESS) {
      ABORT("HG cb failed");
    } else {
      s->addrs[r] = outs[r].addr;
    }
  }

  free(outs);

  return HG_SUCCESS;
}

/*
 * mssg_get_rank: gets our rank
 */
int mssg_get_rank(const mssg_t* s) {
  if (s != NULL)
    return s->rank;
  else
    return -1;
}

/*
 * mssg_get_count: gets our count (world size)
 */
int mssg_get_count(const mssg_t* s) {
  if (s != NULL)
    return s->num_addrs;
  else
    return -1;
}

/*
 * mssg_get_addr: get a rank's address
 */
hg_addr_t mssg_get_addr(const mssg_t* s, int rank) {
  if (s != NULL && rank >= 0 && rank < s->num_addrs)
    return s->addrs[rank];
  else
    return HG_ADDR_NULL;
}

/*
 * mssg_get_addr_str: get a rank's address string
 */
const char* mssg_get_addr_str(const mssg_t* s, int rank) {
  if (s != NULL && rank >= 0 && rank < s->num_addrs)
    return s->addr_strs[rank];
  else
    return NULL;
}

/*
 * mssg_finalize: we're done, free everything!
 */
void mssg_finalize(mssg_t* s) {
  if (!s) return;

  for (int i = 0; i < s->num_addrs; i++) {
    if (s->addrs[i] != HG_ADDR_NULL) HG_Addr_free(s->hgcl, s->addrs[i]);
  }
  free(s->backing_buf);
  free(s->addr_strs);
  free(s->addrs);
  free(s);
}
}
