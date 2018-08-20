/*
 * Copyright (c) 2016 UChicago Argonne, LLC
 * Copyright (c) 2017-2018, Carnegie Mellon University.
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
  int buf_size;      /* the buffer size */
  int rank;          /* my rank in the comm we init'd with */
};

/*
 * mssg_lookup_out: tracks state for looking up hg addr strings
 */
typedef struct mssg_lookup_out {
  hg_return_t hret;
  hg_addr_t addr;
  int* cb_count;
} mssg_lookup_out_t;

/*
 * helper functions
 */

/*
 * setup_addr_str_list: create an array of pointers into a buffer
 * filled with null terminated strings (using strlen() to get string
 * size as we walk through it).
 */
static char** setup_addr_str_list(int num_addrs, char* buf) {
  char** ret = (char**)malloc(num_addrs * sizeof(*ret));
  if (ret == NULL) return NULL;

  ret[0] = buf;
  for (int i = 1; i < num_addrs; i++) {
    char* a = ret[i - 1];
    ret[i] = a + strlen(a) + 1;
  }
  return ret;
}

/*
 * mssg_lookup_cb: adress lookup callback function
 */
static hg_return_t mssg_lookup_cb(const struct hg_cb_info* info) {
  mssg_lookup_out_t* out = (mssg_lookup_out_t*)info->arg;
  *out->cb_count += 1;
  out->hret = info->ret;
  if (out->hret != HG_SUCCESS)
    out->addr = HG_ADDR_NULL;
  else
    out->addr = info->info.lookup.addr;
  return HG_SUCCESS;
}

/*
 * API functions
 */

/*
 * mssg_init_mpi: init mssg via MPI.  this doesn't actually do the
 * lookups of the address strings (see mssg_lookup()).
 */
mssg_t* mssg_init_mpi(hg_class_t* hgcl, MPI_Comm comm) {
  hg_addr_t self_addr = HG_ADDR_NULL;
  char* self_addr_str = NULL;
  hg_size_t self_addr_size = 0;
  int self_addr_size_int = 0;  // for mpi-friendly conversion

  // collective helpers
  char* buf = NULL;
  int* sizes = NULL;
  int* sizes_psum = NULL;
  int comm_size = 0;
  int comm_rank = 0;

  // hg addresses
  hg_addr_t* addrs = NULL;

  // return data
  char** addr_strs = NULL;
  mssg_t* s = NULL;

  // misc return codes
  hg_return_t hret;

  /* get my hg_addr_t address, and my address string */
  hret = HG_Addr_self(hgcl, &self_addr);
  if (hret != HG_SUCCESS) goto fini;
  hret = HG_Addr_to_string(hgcl, NULL, &self_addr_size, self_addr);
  if (self_addr == NULL) goto fini;
  self_addr_str = (char*)malloc(self_addr_size);
  if (self_addr_str == NULL) goto fini;
  hret = HG_Addr_to_string(hgcl, self_addr_str, &self_addr_size, self_addr);
  if (hret != HG_SUCCESS) goto fini;
  self_addr_size_int = (int)self_addr_size;  // null char included in call

  // gather the buffer sizes: we xchg self_addr_size among ranks via sizes[]
  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);
  sizes = (int*)malloc(comm_size * sizeof(*sizes));
  if (sizes == NULL) goto fini;
  sizes[comm_rank] = self_addr_size_int;
  MPI_Allgather(MPI_IN_PLACE, 0, MPI_BYTE, sizes, 1, MPI_INT, comm);

  // compute a exclusive prefix sum of the data sizes,
  // including the total at the end
  sizes_psum = (int*)malloc((comm_size + 1) * sizeof(*sizes_psum));
  if (sizes_psum == NULL) goto fini;
  sizes_psum[0] = 0;
  for (int i = 1; i < comm_size + 1; i++)
    sizes_psum[i] = sizes_psum[i - 1] + sizes[i - 1];

  // allgather the addresses
  buf = (char*)malloc(sizes_psum[comm_size]);
  if (buf == NULL) goto fini;
  MPI_Allgatherv(self_addr_str, self_addr_size_int, MPI_BYTE, buf, sizes,
                 sizes_psum, MPI_BYTE, comm);

  // set the addresses
  addr_strs = setup_addr_str_list(comm_size, buf);
  if (addr_strs == NULL) goto fini;

  // init peer addresses
  addrs = (hg_addr_t*)malloc(comm_size * sizeof(*addrs));
  if (addrs == NULL) goto fini;
  for (int i = 0; i < comm_size; i++) addrs[i] = HG_ADDR_NULL;
  addrs[comm_rank] = self_addr;

  // set up the output
  /*
   * note it sets the locals to NULL, causing the free()'s in fini
   * to not do anything...
   */
  s = (struct mssg*)malloc(sizeof(*s));
  if (s == NULL) goto fini;
  s->hgcl = NULL;  // set in mssg_lookup
  s->addr_strs = addr_strs;
  addr_strs = NULL;
  s->addrs = addrs;
  addrs = NULL;
  s->backing_buf = buf;
  buf = NULL;
  s->num_addrs = comm_size;
  s->buf_size = sizes_psum[comm_size];
  s->rank = comm_rank;
  self_addr = HG_ADDR_NULL;  // don't free this on success

fini:
  if (self_addr != HG_ADDR_NULL) HG_Addr_free(hgcl, self_addr);
  free(buf);
  free(sizes);
  free(addr_strs);
  free(addrs);
  free(self_addr_str);
  free(sizes_psum);
  return s;
}

/*
 * mssg_lookup: lookup all addr strings to get hg_addr_t.  this may
 * establish connections to everything we lookup...
 */
hg_return_t mssg_lookup(mssg_t* s, hg_context_t* hgctx) {
  // set of outputs
  mssg_lookup_out_t* out = NULL; /* an array */
  int cb_count = 0;
  // "effective" rank for the lookup loop
  int eff_rank = 0;

  // set the hg class up front - need for destructing addrs
  /* XXX: harmless, but already done by init */
  s->hgcl = HG_Context_get_class(hgctx);
  if (s->hgcl == NULL) return HG_INVALID_PARAM;

  eff_rank = s->rank;
  cb_count++;

  // init addr metadata
  out = (mssg_lookup_out_t*)malloc(s->num_addrs * sizeof(*out));
  if (out == NULL) return HG_NOMEM_ERROR;
  // FIXME: lookups don't have a cancellation path, so in an intermediate
  // error we can't free the memory, lest we cause a segfault

  // rank is set, perform lookup.  we start i at 1 to skip looking up
  // our own address (we've already got that).
  hg_return_t hret;
  for (int i = 1; i < s->num_addrs; i++) {
    int r = (eff_rank + i) % s->num_addrs;
    out[r].cb_count = &cb_count;
    hret = HG_Addr_lookup(hgctx, &mssg_lookup_cb, &out[r], s->addr_strs[r],
                          HG_OP_ID_IGNORE);
    if (hret != HG_SUCCESS) return hret;
  }

  // lookups posted, enter the progress loop until finished
  do {
    unsigned int count = 0;
    do {
      hret = HG_Trigger(hgctx, 0, 1, &count);
    } while (hret == HG_SUCCESS && count > 0);
    if (hret != HG_SUCCESS && hret != HG_TIMEOUT) return hret;

    hret = HG_Progress(hgctx, 100);
  } while (cb_count < s->num_addrs &&
           (hret == HG_SUCCESS || hret == HG_TIMEOUT));

  if (hret != HG_SUCCESS && hret != HG_TIMEOUT) return hret;

  for (int i = 0; i < s->num_addrs; i++) {
    if (i != s->rank) {
      if (out[i].hret != HG_SUCCESS)
        return out[i].hret;
      else
        s->addrs[i] = out[i].addr;
    }
  }

  free(out);

  return HG_SUCCESS;
}

/*
 * mssg_get_rank: gets our rank
 */
int mssg_get_rank(const mssg_t* s) { return s->rank; }

/*
 * mssg_get_count: gets our count (world size)
 */
int mssg_get_count(const mssg_t* s) { return s->num_addrs; }

/*
 * mssg_get_addr: get a rank's address
 */
hg_addr_t mssg_get_addr(const mssg_t* s, int rank) {
  if (rank >= 0 && rank < s->num_addrs)
    return s->addrs[rank];
  else
    return HG_ADDR_NULL;
}

/*
 * mssg_get_addr_str: get a rank's address string
 */
const char* mssg_get_addr_str(const mssg_t* s, int rank) {
  if (rank >= 0 && rank < s->num_addrs)
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
