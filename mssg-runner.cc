/*
 * Copyright (c) 2018 Carnegie Mellon University.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT
 * HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * mssg-runner.cc a simple program for checking mssg functions.
 */
#include "mssg.h"

#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/*
 * helper/utility functions, included inline here so we are self-contained
 * in one single source file...
 */
static char* argv0; /* argv[0], program name */
static int myrank = 0;

/*
 * vcomplain/complain about something.  if ret is non-zero we exit(ret)
 * after complaining.  if r0only is set, we only print if myrank == 0.
 */
static void vcomplain(int ret, int r0only, const char* format, va_list ap) {
  if (!r0only || myrank == 0) {
    fprintf(stderr, "%s: ", argv0);
    vfprintf(stderr, format, ap);
    fprintf(stderr, "\n");
  }
  if (ret) {
    MPI_Finalize();
    exit(ret);
  }
}

static void complain(int ret, int r0only, const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  vcomplain(ret, r0only, format, ap);
  va_end(ap);
}

/*
 * abort with a fatal message
 */
#define FATAL(msg) fatal(__FILE__, __LINE__, msg)
static void fatal(const char* f, int d, const char* msg) {
  fprintf(stderr, "=== ABORT === ");
  fprintf(stderr, "%s (%s:%d)", msg, f, d);
  fprintf(stderr, "\n");
  abort();
}

/*
 * default values
 */
#define DEF_PROTO "bmi+tcp"  /* default mercury protocol to use */
#define DEF_ADDR "127.0.0.1" /* default address to use */
#define DEF_RECVRADIX 0      /* everyone is both a sender and a receiver */
#define DEF_PORT 50000       /* default port number for rank 0 */
#define DEF_TIMEOUT 120      /* alarm timeout */

/*
 * gs: shared global data (e.g. from the command line)
 */
static struct gs {
  int size;          /* world size (from MPI) */
  const char* proto; /* mercury protocol specifier */
  const char* addr;  /* hostname, ip, or the net interface to use */
  int rr;            /* receiver radix */
  int port;          /* port number for rank 0 */
  int timeout;       /* alarm timeout */
} g;

/*
 * alarm signal handler
 */
static void sigalarm(int foo) {
  fprintf(stderr, "SIGALRM detected (%d)\n", myrank);
  fprintf(stderr, "Alarm clock\n");
  MPI_Finalize();
  exit(1);
}

/*
 * usage
 */
static void usage(const char* msg) {
  /* only have rank 0 print usage error message */
  if (myrank) goto skip_prints;

  if (msg) fprintf(stderr, "%s: %s\n", argv0, msg);
  fprintf(stderr, "usage: %s [options]\n", argv0);
  fprintf(stderr, "\noptions:\n");
  fprintf(stderr, "\t-p proto  hg proto\n");
  fprintf(stderr, "\t-n addr   hostname, ip, or net interface\n");
  fprintf(stderr, "\t-b base   base port number\n");
  fprintf(stderr, "\t-r radix  receiver radix [0-8]\n");
  fprintf(stderr, "\t-t sec    timeout (alarm), in seconds\n");

skip_prints:
  MPI_Finalize();
  exit(1);
}

/*
 * per-rank program state (diffs from rank to rank)
 */
static struct ps {
  char hg_str[256];
  hg_context_t* hg_ctx;
  hg_class_t* hg_clz;
  unsigned int hg_recvmask;
  int is_receiver;
  MPI_Comm recv;
} p;

/*
 * forward prototype decls.
 */
static void doit();

/*
 * main program.
 */
int main(int argc, char* argv[]) {
  int ch;

  argv0 = argv[0];

  /* mpich says we should call this early as possible */
  if (MPI_Init(&argc, &argv) != MPI_SUCCESS) {
    FATAL("!MPI_Init");
  }

  /* we want lines!! */
  setlinebuf(stdout);

  /* setup default to zero/null, except as noted below */
  memset(&g, 0, sizeof(g));
  if (MPI_Comm_rank(MPI_COMM_WORLD, &myrank) != MPI_SUCCESS)
    FATAL("!MPI_Comm_rank");
  if (MPI_Comm_size(MPI_COMM_WORLD, &g.size) != MPI_SUCCESS)
    FATAL("!MPI_Comm_size");

  g.proto = DEF_PROTO;
  g.addr = DEF_ADDR;
  g.rr = DEF_RECVRADIX;
  g.port = DEF_PORT;
  g.timeout = DEF_TIMEOUT;

  while ((ch = getopt(argc, argv, "b:n:p:r:t:")) != -1) {
    switch (ch) {
      case 'b':
        g.port = atoi(optarg);
        break;
      case 'n':
        g.addr = optarg;
        break;
      case 'p':
        g.proto = optarg;
        break;
      case 'r':
        g.rr = atoi(optarg);
        if (g.rr < 0 || g.rr > 8) usage("bad receiver radix");
        break;
      case 't':
        g.timeout = atoi(optarg);
        if (g.timeout < 0) usage("bad timeout");
        break;
      default:
        usage(NULL);
    }
  }

  if (myrank == 0) {
    printf("== Program options:\n");
    printf(" > MPI_rank   = %d\n", myrank);
    printf(" > MPI_size   = %d\n", g.size);
    printf(" > HG_proto   = %s\n", g.proto);
    printf(" > HG_addr    = %s\n", g.addr);
    printf(" > HG_port    = %d\n", g.port);
    printf(" > recv_mask  = %d (radix=%d)\n", 32 - g.rr, g.rr);
    printf(" > timeout    = %d secs\n", g.timeout);
    printf("\n");
  }

  signal(SIGALRM, sigalarm);
  alarm(g.timeout);

  memset(&p, 0, sizeof(p));
  p.recv = MPI_COMM_NULL;
  p.hg_recvmask = ~static_cast<unsigned int>(0);
  p.hg_recvmask <<= g.rr;
  p.is_receiver = myrank == (myrank & p.hg_recvmask);

  /* establish the addr str according to the proto requested at the cmd */
  if (strcmp(g.proto, "na+sm") == 0) {
    /*
     * na+sm address = PID followed by a user-specified
     * unique SM ID number.
     */
    snprintf(p.hg_str, sizeof(p.hg_str), "na+sm://%d:0", getpid());
  } else if (strcmp(g.proto, "bmi+tcp") == 0) {
    /*
     * as if mercury v1.0.0, when we omit hostname/ip or port
     * for server-side initialization, mercury will try to fill
     * them in for us before passing the address to bmi.
     * To determine the name of the host, mercury uses the
     * POSIX gethostname() function. To decide a port, mercury
     * retries from port 22222 to port 22222+128 until it
     * finds a free port. This port detection process is not
     * avoided when we set the port to 0.
     *
     * earlier versions of mercury does allow port 0 to be
     * passed to bmi, which in turn will honor it and will pass
     * it to the OS so the OS can pick up a free port for us.
     * Yet, unfortunately, if we later try to retrieve the
     * final address, bmi will return port 0 back to us rather
     * than the actual port picked up by the OS.
     *
     * conclusion: always pass full address information when
     * using bmi+tcp.
     */
    snprintf(p.hg_str, sizeof(p.hg_str), "bmi+tcp://%s:%d", g.addr,
             g.port + myrank);
  } else if (strcmp(g.proto, "ofi+gni") == 0) {
    /*
     * as if libfabric v1.7.0, only hostname, ip, or the
     * interface name of the NIC is used by the driver to select
     * an underlying adaptor. Port number is not used and
     * will be ignored if specified.
     *
     * as if mercury v1.0.0, when we omit hostname, ip, or the
     * interface name of the NIC, mercury will use "ipogif0".
     */
    snprintf(p.hg_str, sizeof(p.hg_str), "ofi+gni://%s", g.addr);
  } else if (strcmp(g.proto, "ofi+psm2") == 0) {
    /*
     * as if libfabric v1.7.0, any hostname, ip, NIC interface name,
     * or port number specified will be ignored.
     */
    snprintf(p.hg_str, sizeof(p.hg_str), "ofi+psm2");
  } else {
    /* default: assume an ip:port format */
    snprintf(p.hg_str, sizeof(p.hg_str), "%s://%s:%d", g.proto, g.addr,
             g.port + myrank);
  }

  p.hg_clz = HG_Init(p.hg_str, p.is_receiver);
  if (!p.hg_clz) complain(EXIT_FAILURE, 0, "!HG_Init");
  p.hg_ctx = HG_Context_create(p.hg_clz);
  if (!p.hg_ctx) complain(EXIT_FAILURE, 0, "!HG_Context_create");

  MPI_Comm_split(MPI_COMM_WORLD, p.is_receiver ? 1 : MPI_UNDEFINED, myrank,
                 &p.recv);
  if (p.is_receiver) {
    if (p.recv == MPI_COMM_NULL) {
      complain(EXIT_FAILURE, 0, "!MPI_Comm_split");
    }
  }

  doit();

  if (p.recv != MPI_COMM_NULL) MPI_Comm_free(&p.recv);
  HG_Context_destroy(p.hg_ctx);
  HG_Finalize(p.hg_clz);
  MPI_Finalize();

  return 0;
}

static void doit() {
  mssg_t* m;

  m = mssg_init_mpi(p.hg_clz, MPI_COMM_WORLD, p.is_receiver);
  if (!m) complain(EXIT_FAILURE, 0, "!mssg_init_mpi");
  mssg_lookup(m, p.hg_ctx);

  if (myrank == 0) printf("== Results:\n\n");
  for (int r = 0; r < g.size; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (myrank == r) {
      printf("[rank-%d]\n", myrank);
      printf("is_receiver? %d\n", p.is_receiver);
      printf("mssg_backing_bufsize: %d bytes\n", mssg_backing_bufsize(m));
      for (int i = 0; i < g.size; i++) {
        const char* const str = mssg_get_addr_str(m, i);
        if (myrank == i && strcmp(p.hg_str, str) != 0) {
          printf("%d=%s (init=%s)\n", i, str, p.hg_str);
        } else {
          printf("%d=%s\n", i, str);
        }
      }
      printf("\n");
    }
  }

  mssg_finalize(m);
}
