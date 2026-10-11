# -*- coding: utf-8 -*-
"""
Work around a macOS linker bug: some ld versions (seen with ld-27037 / Xcode 27) leave the LC_SYMTAB
string table only 4-byte aligned, and dyld then refuses the extension with
"mis-aligned LINKEDIT string pool". This script moves the string table to an 8-byte boundary and
re-signs the file ad hoc. Release wheels built on GitHub runners are not affected.

usage: python scripts/macos_fix_linkedit.py path/to/_geatpy_core.abi3.so
"""
import struct
import subprocess
import sys


def realign(path):
    subprocess.run(['codesign', '--remove-signature', path], check=True)
    with open(path, 'rb') as fh:
        data = bytearray(fh.read())
    magic, _, _, _, ncmds = struct.unpack_from('<IiiII', data, 0)
    if magic != 0xFEEDFACF:
        raise SystemExit('%s is not a 64-bit Mach-O file' % path)
    off, symtab, linkedit = 32, None, None
    for _ in range(ncmds):
        cmd, size = struct.unpack_from('<II', data, off)
        if cmd == 0x2:  # LC_SYMTAB
            symtab = off
        elif cmd == 0x19 and data[off + 8:off + 24].rstrip(b'\0') == b'__LINKEDIT':
            linkedit = off
        off += size
    stroff = struct.unpack_from('<I', data, symtab + 16)[0]
    pad = (-stroff) % 8
    if pad:
        data[stroff:stroff] = b'\0' * pad
        struct.pack_into('<I', data, symtab + 16, stroff + pad)
        vmaddr, vmsize, fileoff, filesize = struct.unpack_from('<QQQQ', data, linkedit + 24)
        filesize += pad
        vmsize = max(vmsize, (filesize + 0x3FFF) & ~0x3FFF)
        struct.pack_into('<QQQQ', data, linkedit + 24, vmaddr, vmsize, fileoff, filesize)
        with open(path, 'wb') as fh:
            fh.write(data)
    subprocess.run(['codesign', '-s', '-', '-f', path], check=True, capture_output=True)
    return pad


if __name__ == '__main__':
    for p in sys.argv[1:]:
        moved = realign(p)
        print('%s: %s' % (p, 'string table moved by %d bytes' % moved if moved else 'already aligned'))
