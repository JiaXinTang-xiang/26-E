#!/usr/bin/env python3
"""Test the STM32-to-PC serial return path without moving any motor."""

from __future__ import annotations

import argparse
import time

from puzzle_device.calibration.gantry_protocol import build_serial_health_check_frame


EXPECTED_REJECT = bytes.fromhex("AA 03 B2 B1 55")
LOOPBACK_FRAME = bytes.fromhex("A5 5A 12 34 56 78")


def read_until(port, expected: bytes, timeout: float) -> bytes:
    received = bytearray()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        chunk = port.read(max(1, port.in_waiting))
        if chunk:
            received.extend(chunk)
            if expected in received:
                break
    return bytes(received)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send an invalid command and wait for the STM32 B2 response."
    )
    parser.add_argument("--serial", required=True, help="CH340 port, for example COM30")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--timeout", type=float, default=2.0)
    parser.add_argument(
        "--loopback", action="store_true",
        help="test CH340 alone after disconnecting STM32 and shorting TXD to RXD",
    )
    args = parser.parse_args()

    try:
        import serial
    except ImportError as exc:
        raise SystemExit("pyserial is required: python -m pip install pyserial") from exc

    if args.loopback:
        frame = LOOPBACK_FRAME
        expected = LOOPBACK_FRAME
    else:
        # Corrupt only the checksum. The controller must reject this frame
        # before any coordinates are used, so it cannot start a motor action.
        frame = build_serial_health_check_frame()
        expected = EXPECTED_REJECT

    with serial.Serial(
        args.serial, args.baudrate, timeout=0.05, write_timeout=1.0
    ) as port:
        port.reset_input_buffer()
        port.reset_output_buffer()
        port.write(frame)
        port.flush()
        print(f"TX {len(frame)} B: {frame.hex(' ').upper()}")

        received = read_until(port, expected, args.timeout)
        if expected in received:
            print(f"RX {len(received)} B: {received.hex(' ').upper()}")
            if args.loopback:
                print("PASS: CH340 TXD/RXD loopback is working.")
            else:
                print("PASS: STM32 TX -> CH340 RX -> PC return path is working.")
            return 0

        if received:
            print(f"RX {len(received)} B: {received.hex(' ').upper()}")
            print("FAIL: bytes arrived, but the expected frame was not found.")
        else:
            print("RX 0 B")
            if args.loopback:
                print("FAIL: CH340 RXD loopback received nothing.")
            else:
                print("FAIL: no return bytes from the selected STM32 TX output.")
                print("USART1: check PA9/TX1; USART2 test: check PA2/TX2; always share GND.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
