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
 * mssg-runner.cc a simple program for driving mssg routines.
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
#define DEF_RECVRADIX 0      /* everyone is both a sender and a receiver */
#define DEF_PROTO "bmi+tcp"  /* default mercury protocol to use */
#define DEF_HOST "127.0.0.1" /* default address to use */
#define DEF_PORT "50000"     /* default port number for rank 0 */
#define DEF_TIMEOUT 120      /* alarm timeout */

/*
 * gs: shared global data (e.g. from the command line)
 */
static struct gs {
  int size;          /* world size (from MPI) */
  const char* proto; /* mercury protocol specifier */
  const char* host;  /* hostname, ip, or the net interface to use */
  const char* port;  /* port number for rank 0 */
  int rr;            /* receiver radix */
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
  fprintf(stderr, "\t-P proto    hg proto\n");
  fprintf(stderr, "\t-H host     hostname, ip, or net interface\n");
  fprintf(stderr, "\t-p port     base port number\n");
  fprintf(stderr, "\t-r radix    receiver radix [0-8]\n");
  fprintf(stderr, "\t-t sec      timeout (alarm), in seconds\n");

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
  g.host = DEF_HOST;
  g.rr = DEF_RECVRADIX;
  g.port = DEF_PORT;
  g.timeout = DEF_TIMEOUT;

  while ((ch = getopt(argc, argv, "p:H:P:r:t:h")) != -1) {
    switch (ch) {
      case 'p':
        g.port = optarg;
        break;
      case 'H':
        g.host = optarg;
        break;
      case 'P':
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
    printf(" > HG_host    = %s\n", g.host);
    printf(" > HG_port    = %s\n", g.port);
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
     * as if mercury v1.0.0, when we omit the hostname/ip or the port
     * for server-side initialization, mercury will try to fill
     * them in for us before passing the address to bmi.
     * To determine the hostname to bind, mercury uses the
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
    snprintf(p.hg_str, sizeof(p.hg_str), "bmi+tcp://%s:%d", g.host,
             atoi(g.port) + myrank);
  } else if (strcmp(g.proto, "ofi+gni") == 0) {
    /*
     * as if mercury v1.0.0, mercury will use the hostname, ip addr,
     * or the nic's interface name one specifies to determine the ip addr
     * to be passed to ofi. Port number is not used and
     * will be ignored if specified. When hostname/ip/iface is
     * omitted, mercury will use ipogif0.
     */
    snprintf(p.hg_str, sizeof(p.hg_str), "ofi+gni://%s", g.host);
  } else if (strcmp(g.proto, "ofi+psm2") == 0) {
    /*
     * as if mercury v1.0.0, mercury will not pass any addr
     * information to ofi for psm2. Any hostname/ip/iface or
     * port number specified will be ignored.
     */
    snprintf(p.hg_str, sizeof(p.hg_str), "ofi+psm2");
  } else {
    /*
     * in general, we assume an "<hostname/ip/iface>:<port>" format.
     * We will omit the port or the hostname/ip/iface if it is
     * specified as "x". note also that while the following code allows
     * specifying only the port and not the hostname/ip/iface,
     * mercury may not accept it for a specific proto.
     *
     * Both ofi+tcp and ofi+verbs allow the port or both the port
     * and the hostname/ip/iface to be omitted, in which case no
     * addr information will be passed to ofi. If hostname/ip/iface
     * is specified, it will be converted to iface and ip before given
     * to ofi. Additionally, if port is specified, it will be sent
     * to ofi too.
     */
    if (strcmp(g.host, "x") != 0 && strcmp(g.port, "x") != 0) {
      snprintf(p.hg_str, sizeof(p.hg_str), "%s://%s:%d", g.proto, g.host,
               atoi(g.port) + myrank);
    } else if (strcmp(g.port, "x") != 0) {
      snprintf(p.hg_str, sizeof(p.hg_str), "%s://:%d", g.proto,
               atoi(g.port) + myrank);
    } else if (strcmp(g.host, "x") != 0) {
      snprintf(p.hg_str, sizeof(p.hg_str), "%s://%s", g.proto, g.host);
    } else {
      snprintf(p.hg_str, sizeof(p.hg_str), "%s", g.proto);
    }
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
