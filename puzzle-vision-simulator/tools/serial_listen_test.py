#!/usr/bin/env python3
"""Listen for the STM32 USART1 power-on diagnostic frame."""

from __future__ import annotations

import argparse
import time


DIAGNOSTIC_FRAME = bytes.fromhex("AA 03 BD BE 55")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serial", required=True)
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=12.0)
    args = parser.parse_args()

    import serial

    with serial.Serial(args.serial, args.baudrate, timeout=0.05) as port:
        port.reset_input_buffer()
        print("Listening. Reset or power-cycle the STM32 now...")
        received = bytearray()
        deadline = time.monotonic() + args.timeout
        while time.monotonic() < deadline:
            chunk = port.read(max(1, port.in_waiting))
            if chunk:
                received.extend(chunk)
                print(f"RX {len(chunk)} B: {chunk.hex(' ').upper()}")
                if DIAGNOSTIC_FRAME in received:
                    print("PASS: STM32 USART1 PA9 transmit path is working.")
                    return 0
        print(f"FAIL: no diagnostic frame, total RX {len(received)} B.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
