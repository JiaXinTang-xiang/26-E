"""Compatibility adapter for the 17-byte gantry command protocol."""

from __future__ import annotations

from pathlib import Path
import struct
from typing import Iterable


FRAME_HEADER = 0xAA
FRAME_LENGTH = 0x0F
FRAME_FOOTER = 0x55
CMD_PICK_AND_PLACE = 0xA1
CMD_PICK_AND_PLACE_DUAL_ANGLE = 0xA2
CMD_PICK_AND_PLACE_DUAL_ANGLE_CONTINUE = 0xA3
SERVO_COMMAND_MARKER = 0x5A5A

STATUS_FRAME_LENGTH = 0x03
STATUS_COMMAND_ACCEPTED = 0xB0
STATUS_ACTION_COMPLETE = 0xB1
STATUS_COMMAND_REJECTED = 0xB2
STATUS_ACTION_FAILED = 0xB3
STATUS_ACTION_CONTINUE_READY = 0xB4
STATUS_NAMES = {
    STATUS_COMMAND_ACCEPTED: "命令已接收",
    STATUS_ACTION_COMPLETE: "动作已完成并回零",
    STATUS_COMMAND_REJECTED: "命令校验失败或参数越界",
    STATUS_ACTION_FAILED: "正常运动计数停止，动作已中止",
    STATUS_ACTION_CONTINUE_READY: "当前块完成，Z轴安全，XY保持当前位置",
}


def build_serial_health_check_frame() -> bytes:
    """Build a deliberately rejected frame that cannot start motor motion."""
    frame = bytearray(build_pick_and_place_frame(0, 0, 0, 0))
    frame[-2] ^= 0xFF
    return bytes(frame)


def _port_search_text(port_info) -> str:
    return " ".join(
        str(getattr(port_info, name, "") or "")
        for name in ("device", "description", "manufacturer", "hwid", "product")
    ).upper()


def select_ch340_port(
    ports: Iterable, preferred_port: str | None = None
) -> str | None:
    """Select a CH340/USB-serial device, preferring the configured COM port."""
    keywords = ("CH340", "CH341", "USB-SERIAL", "USB SERIAL", "QINHENG", "WCH")
    values = list(ports)
    preferred = (preferred_port or "").strip().upper()
    if preferred:
        for port_info in values:
            if (
                str(getattr(port_info, "device", "")).upper() == preferred
                and any(keyword in _port_search_text(port_info) for keyword in keywords)
            ):
                return str(port_info.device)
    matches = [
        str(port_info.device)
        for port_info in values
        if any(keyword in _port_search_text(port_info) for keyword in keywords)
    ]
    return matches[0] if len(matches) == 1 else None


def discover_ch340_port(preferred_port: str | None = None) -> str | None:
    try:
        from serial.tools import list_ports
    except ImportError:
        return None
    # When the user explicitly named a port that exists on the filesystem
    # but wasn't enumerated by list_ports (common on Jetson with custom udev
    # rules naming devices like /dev/ttyCH341USB0), trust the user's choice.
    if preferred_port:
        preferred_path = Path(preferred_port)
        if preferred_path.exists():
            return str(preferred_path)
    return select_ch340_port(list_ports.comports(), preferred_port)


def build_pick_and_place_frame(
    source_x: int, source_y: int, destination_x: int, destination_y: int,
    source_z: int = 0, destination_z: int = 0,
    *, rotation_angle_deg: int | None = None,
) -> bytes:
    """Build the 17-byte STM32 pick-and-place frame in big-endian order.

    The two legacy Z fields remain zero for old commands. New rotation-aware
    commands put a marker in source Z and the absolute 0-270 degree servo angle
    in destination Z, preserving compatibility with calibration tools.
    """
    if rotation_angle_deg is not None:
        if source_z != 0 or destination_z != 0:
            raise ValueError("rotation_angle_deg cannot be combined with explicit Z fields")
        if not 0 <= rotation_angle_deg <= 270:
            raise ValueError("servo angle must be between 0 and 270 degrees")
        source_z = SERVO_COMMAND_MARKER
        destination_z = rotation_angle_deg
    values = (source_x, source_y, source_z, destination_x, destination_y, destination_z)
    if any(not 0 <= value <= 0xFFFF for value in values):
        raise ValueError("all pulse coordinates must fit an unsigned 16-bit value")
    payload = struct.pack(
        ">BHHHHHH", CMD_PICK_AND_PLACE, source_x, source_y, source_z,
        destination_x, destination_y, destination_z,
    )
    checksum = 0
    for value in bytes([FRAME_LENGTH]) + payload:
        checksum ^= value
    return bytes([FRAME_HEADER, FRAME_LENGTH]) + payload + bytes([checksum, FRAME_FOOTER])


def build_dual_angle_pick_and_place_frame(
    source_x: int, source_y: int, destination_x: int, destination_y: int,
    *, pick_angle_deg: int, place_angle_deg: int, return_home: bool = True,
) -> bytes:
    """Build an A2/A3 frame carrying independent pick and place servo angles.

    A2 returns every axis home after the placement. A3 is used only for an
    intermediate automatic task: it raises Z and returns the servo home while
    preserving the current XY position for the next absolute-coordinate task.
    """
    values = (
        source_x, source_y, pick_angle_deg,
        destination_x, destination_y, place_angle_deg,
    )
    if any(not 0 <= value <= 0xFFFF for value in values):
        raise ValueError("all pulse coordinates must fit an unsigned 16-bit value")
    if not 0 <= pick_angle_deg <= 270 or not 0 <= place_angle_deg <= 270:
        raise ValueError("servo angles must be between 0 and 270 degrees")
    command = (
        CMD_PICK_AND_PLACE_DUAL_ANGLE
        if return_home
        else CMD_PICK_AND_PLACE_DUAL_ANGLE_CONTINUE
    )
    payload = struct.pack(
        ">BHHHHHH", command,
        source_x, source_y, pick_angle_deg,
        destination_x, destination_y, place_angle_deg,
    )
    checksum = 0
    for value in bytes([FRAME_LENGTH]) + payload:
        checksum ^= value
    return bytes([FRAME_HEADER, FRAME_LENGTH]) + payload + bytes([checksum, FRAME_FOOTER])


class GantryStatusParser:
    """Extract compact controller status frames from echoed serial traffic."""

    def __init__(self) -> None:
        self._buffer = bytearray()

    def reset(self) -> None:
        self._buffer.clear()

    def feed(self, data: bytes) -> list[int]:
        self._buffer.extend(data)
        statuses: list[int] = []
        while len(self._buffer) >= 5:
            try:
                start = self._buffer.index(FRAME_HEADER)
            except ValueError:
                self._buffer.clear()
                break
            if start:
                del self._buffer[:start]
            if len(self._buffer) < 5:
                break
            if self._buffer[1] != STATUS_FRAME_LENGTH:
                del self._buffer[0]
                continue
            status = self._buffer[2]
            checksum = STATUS_FRAME_LENGTH ^ status
            if self._buffer[3] == checksum and self._buffer[4] == FRAME_FOOTER:
                statuses.append(status)
                del self._buffer[:5]
            else:
                del self._buffer[0]
        if len(self._buffer) > 64:
            del self._buffer[:-4]
        return statuses


class OptionalSerialPort:
    """Serial sender that remains usable in dry-run mode without pyserial."""

    def __init__(self, port: str | None = None, baudrate: int = 115200):
        self.port = port
        self.baudrate = baudrate
        self._serial = None

    @property
    def connected(self) -> bool:
        return self._serial is not None and bool(getattr(self._serial, "is_open", True))

    def connect(self) -> None:
        if not self.port:
            return
        self.close()
        try:
            import serial
        except ImportError as exc:
            raise RuntimeError("install pyserial to use a physical serial port") from exc
        self._serial = serial.Serial(
            self.port, self.baudrate, timeout=0, write_timeout=1.0,
        )
        self._serial.reset_input_buffer()
        self._serial.reset_output_buffer()

    def send(self, frame: bytes) -> None:
        if not self.connected:
            raise RuntimeError("serial port is not connected")
        written = self._serial.write(frame)
        self._serial.flush()
        if written != len(frame):
            raise RuntimeError(f"serial write incomplete: {written}/{len(frame)} bytes")

    def discard_input(self) -> None:
        """Discard status bytes left over from an earlier command."""
        if self.connected:
            self._serial.reset_input_buffer()

    def read_available(self) -> bytes:
        """Return pending controller bytes without blocking the GUI."""
        if not self.connected:
            return b""
        try:
            waiting = self._serial.in_waiting
        except (OSError, PermissionError):
            # On Windows/CH340, ClearCommError (used by ``in_waiting``) can
            # briefly fail while the USB driver is changing state.  A zero
            # timeout read does not require that status query and can still
            # retrieve a pending B0/B1 frame.
            return self._serial.read(64)
        return self._serial.read(waiting) if waiting else b""

    def close(self) -> None:
        if self._serial is not None:
            try:
                self._serial.close()
            finally:
                self._serial = None
