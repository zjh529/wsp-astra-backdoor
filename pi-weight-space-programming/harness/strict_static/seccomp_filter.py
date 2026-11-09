"""Install a fail-closed seccomp filter for the untrusted analysis worker.

The filter blocks process execution, network sockets, mount changes, and
ptrace. It does not try to recognize "forward-shaped" arithmetic. It runs
inside the worker after trusted imports and before checkpoint parsing.
"""
from __future__ import annotations

import ctypes
import errno
import struct

# x86_64 syscall numbers. The probe refuses to run if the machine is not x86_64
# or if installing the filter fails.
_BLOCKED_X86_64 = (
    59,  # execve
    322,  # execveat
    41,  # socket
    53,  # socketpair
    42,  # connect
    49,  # bind
    50,  # listen
    43,  # accept
    288,  # accept4
    101,  # ptrace
    310,  # process_vm_readv
    311,  # process_vm_writev
    165,  # mount
    166,  # umount2
    155,  # pivot_root
    272,  # unshare
    308,  # setns
    246,  # kexec_load
    320,  # kexec_file_load
    175,  # init_module
    313,  # finit_module
    176,  # delete_module
    303,  # name_to_handle_at
    304,  # open_by_handle_at
    321,  # bpf
    298,  # perf_event_open
    323,  # userfaultfd
    169,  # reboot
    167,  # swapon
    168,  # swapoff
    51,  # getsockname
    52,  # getpeername
    44,  # sendto
    45,  # recvfrom
    46,  # sendmsg
    47,  # recvmsg
    290,  # sendmmsg
    299,  # recvmmsg
)

PR_SET_NO_NEW_PRIVS = 38
PR_SET_SECCOMP = 22
SECCOMP_MODE_FILTER = 2
SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
AUDIT_ARCH_X86_64 = 0xC000003E

BPF_LD = 0x00
BPF_W = 0x00
BPF_ABS = 0x20
BPF_JMP = 0x05
BPF_JEQ = 0x10
BPF_K = 0x00
BPF_RET = 0x06


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint16),
        ("jt", ctypes.c_uint8),
        ("jf", ctypes.c_uint8),
        ("k", ctypes.c_uint32),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [
        ("len", ctypes.c_ushort),
        ("filter", ctypes.POINTER(_SockFilter)),
    ]


def _stmt(code: int, k: int) -> _SockFilter:
    return _SockFilter(code, 0, 0, k & 0xFFFFFFFF)


def _jump(code: int, k: int, jt: int, jf: int) -> _SockFilter:
    return _SockFilter(code, jt, jf, k & 0xFFFFFFFF)


def build_filter() -> ctypes.Array:
    """Allow every syscall except the blocked set. Wrong arch kills the process."""
    ins = [
        _stmt(BPF_LD | BPF_W | BPF_ABS, 4),
        _jump(BPF_JMP | BPF_JEQ | BPF_K, AUDIT_ARCH_X86_64, 1, 0),
        _stmt(BPF_RET | BPF_K, SECCOMP_RET_KILL_PROCESS),
        _stmt(BPF_LD | BPF_W | BPF_ABS, 0),
    ]
    for nr in _BLOCKED_X86_64:
        ins.append(_jump(BPF_JMP | BPF_JEQ | BPF_K, nr, 0, 1))
        ins.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ERRNO | errno.EPERM))
    ins.append(_stmt(BPF_RET | BPF_K, SECCOMP_RET_ALLOW))
    arr = (_SockFilter * len(ins))(*ins)
    return arr


def install_seccomp() -> None:
    """Install the filter. Raise OSError if the kernel rejects it."""
    import platform

    if platform.machine() not in ("x86_64", "amd64"):
        raise OSError("strict-static seccomp filter is implemented for x86_64 only")
    filters = build_filter()
    prog = _SockFprog(len(filters), filters)
    # Keep the array alive for the duration of the prctl call and beyond.
    prog._keepalive = filters  # type: ignore[attr-defined]
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl.argtypes = [
        ctypes.c_int,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
    ]
    libc.prctl.restype = ctypes.c_int
    if libc.prctl(PR_SET_NO_NEW_PRIVS, 1, 0, 0, 0) != 0:
        err = ctypes.get_errno()
        raise OSError(err, "PR_SET_NO_NEW_PRIVS failed: " + __import__("os").strerror(err))
    rc = libc.prctl(PR_SET_SECCOMP, ctypes.c_ulong(SECCOMP_MODE_FILTER), ctypes.byref(prog), 0, 0)
    if rc != 0:
        err = ctypes.get_errno()
        raise OSError(err, "PR_SET_SECCOMP failed: " + __import__("os").strerror(err))


def filter_summary() -> dict:
    return {
        "arch": "x86_64",
        "blocked_syscalls": list(_BLOCKED_X86_64),
        "on_block": "EPERM",
        "on_wrong_arch": "kill",
    }


# Keep struct imported so a future BPF dump helper can use it without a new import
# after seccomp is installed. The worker imports this module before the filter.
_ = struct
