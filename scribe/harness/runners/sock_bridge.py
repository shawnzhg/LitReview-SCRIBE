#!/usr/bin/env python3
"""Bridges a TCP port and a unix socket so a loopback-only namespace can reach the host-side proxy,
and lists the namespace's TCP listeners. Usage: python sock_bridge.py --unix-listen <sock>
--tcp-connect <host:port> | --tcp-listen <host:port> --unix-connect <sock>."""

from __future__ import annotations

import argparse
import errno
import os
import re
import signal
import socket
import subprocess
import sys
import threading

BUFSZ = 65536
BACKLOG_DEFAULT = 128
_STOP = threading.Event()


def log(msg: str) -> None:
    sys.stderr.write("[sock_bridge] %s\n" % msg)
    sys.stderr.flush()


def _pump(src: socket.socket, dst: socket.socket) -> None:
    try:
        while True:
            b = src.recv(BUFSZ)
            if not b:
                break
            dst.sendall(b)
    except OSError:
        pass
    finally:
        try:
            dst.shutdown(socket.SHUT_WR)
        except OSError:
            pass


def _handle(client: socket.socket, connect) -> None:
    up = None
    try:
        up = connect()
    except OSError as e:
        log("upstream connect failed: %s" % e)
        try:
            client.close()
        except OSError:
            pass
        return
    t = threading.Thread(target=_pump, args=(client, up), daemon=True)
    t.start()
    _pump(up, client)
    t.join(timeout=30)
    for s in (client, up):
        try:
            s.close()
        except OSError:
            pass


def serve(listener: socket.socket, connect, name: str) -> int:
    listener.settimeout(1.0)
    log("%s: ready" % name)
    n = 0
    while not _STOP.is_set():
        try:
            client, _ = listener.accept()
        except socket.timeout:
            continue
        except OSError as e:
            if e.errno in (errno.EINTR, errno.EBADF):
                continue
            raise
        n += 1
        client.settimeout(None)
        threading.Thread(target=_handle, args=(client, connect), daemon=True).start()
    log("%s: stopping after %d connection(s)" % (name, n))
    try:
        listener.close()
    except OSError:
        pass
    return 0


def parse_hostport(s: str):
    if ":" not in s:
        return "127.0.0.1", int(s)
    h, p = s.rsplit(":", 1)
    return (h or "127.0.0.1"), int(p)


SUN_PATH_MAX = 107


def check_sun_path(sock_path: str) -> None:
    b = os.fsencode(sock_path)
    if len(b) > SUN_PATH_MAX:
        raise SystemExit(
            "[sock_bridge] FATAL: unix socket path is %d bytes; the kernel limit is %d.\n"
            "  %s\n"
            "  Put the lane socket in a short node-local directory (e.g. /tmp/lwc/<job>/<i>.sock)\n"
            "  and record the path in the task directory instead of placing the file there."
            % (len(b), SUN_PATH_MAX, sock_path))


def unix_listen_tcp_connect(sock_path: str, hostport: str, backlog: int) -> int:
    check_sun_path(sock_path)
    host, port = parse_hostport(hostport)
    d = os.path.dirname(os.path.abspath(sock_path))
    os.makedirs(d, exist_ok=True)
    if os.path.exists(sock_path):
        os.unlink(sock_path)
    ls = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    ls.bind(sock_path)
    os.chmod(sock_path, 0o600)
    ls.listen(backlog)

    def connect():
        c = socket.create_connection((host, port), timeout=30)
        c.settimeout(None)
        return c

    try:
        return serve(ls, connect, "unix(%s) -> tcp(%s:%d)" % (sock_path, host, port))
    finally:
        try:
            os.unlink(sock_path)
        except OSError:
            pass


def tcp_listen_unix_connect(hostport: str, sock_path: str, backlog: int) -> int:
    check_sun_path(sock_path)
    host, port = parse_hostport(hostport)
    ls = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind((host, port))
    ls.listen(backlog)

    def connect():
        c = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        c.settimeout(30)
        c.connect(sock_path)
        c.settimeout(None)
        return c

    return serve(ls, connect, "tcp(%s:%d) -> unix(%s)" % (host, port, sock_path))


def _listeners_via_ss():
    try:
        r = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if r.returncode != 0:
        return None
    ports = set()
    for line in r.stdout.splitlines():
        f = line.split()
        if len(f) < 4:
            continue
        local = f[3]
        m = re.search(r":(\d+)$", local)
        if m:
            ports.add(int(m.group(1)))
    return ports


def _listeners_via_proc():
    ports = set()
    for path in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(path) as f:
                next(f, None)
                for line in f:
                    fld = line.split()
                    if len(fld) < 4 or fld[3] != "0A":
                        continue
                    ports.add(int(fld[1].rsplit(":", 1)[1], 16))
        except FileNotFoundError:
            continue
        except Exception:
            continue
    return ports


def listener_ports():
    p = _listeners_via_ss()
    src = "ss"
    if p is None:
        p = _listeners_via_proc()
        src = "/proc/net/tcp"
    return p, src


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unix-listen", help="host side: unix socket to bind")
    ap.add_argument("--tcp-connect", help="host side: HOST:PORT to dial per connection")
    ap.add_argument("--tcp-listen", help="ns side: HOST:PORT to bind")
    ap.add_argument("--unix-connect", help="ns side: unix socket to dial per connection")
    a = ap.parse_args()

    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: _STOP.set())

    if a.unix_listen and a.tcp_connect:
        return unix_listen_tcp_connect(a.unix_listen, a.tcp_connect, BACKLOG_DEFAULT)
    if a.tcp_listen and a.unix_connect:
        return tcp_listen_unix_connect(a.tcp_listen, a.unix_connect, BACKLOG_DEFAULT)
    ap.error("choose --unix-listen/--tcp-connect or --tcp-listen/--unix-connect")


if __name__ == "__main__":
    sys.exit(main())
