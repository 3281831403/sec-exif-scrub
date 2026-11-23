"""EXIF 清洗练习的唯一实现文件。

数据格式
--------
图像使用可构造的 SEGX 段容器，测试不依赖真实图片：

* 文件头固定 8 字节：魔数 b"SEGX"，版本 1 字节且必须为 1，字节序 1 字节
  （b"M" 表示大端，b"I" 表示小端），保留 2 字节。
* 段头固定 6 字节：标记 uint16、载荷长度 uint32；两个整数都按文件头声明的
  字节序编码。长度只计算载荷，不含 6 字节段头。
* 段载荷紧随段头。段序列由标记 0xFFD9、长度 0 的终止段结束；终止段不进入
  iter_segments 的结果，终止段之后不允许再出现字节。
* 段偏移 offset 是段头首字节的绝对位置，payload_offset 等于 offset 加 6。

EXIF 载荷以 b"Exif\\x00\\x00" 开头，其后是 TIFF 子集：

* TIFF 头为字节序标记 b"II" 或 b"MM"、魔数 42、IFD0 偏移 8。
* 每个 IFD 为 uint16 条目数、若干 12 字节条目、uint32 下一 IFD 偏移。
* 条目为 tag uint16、type uint16、count uint32、value_or_offset uint32。
* 支持的 type 为 1 BYTE、2 ASCII、3 SHORT、4 LONG、5 RATIONAL，元素大小
  依次为 1、1、2、4、8 字节。
* 数据总长不超过 4 字节时内联在 value_or_offset 中，否则该字段是相对 TIFF
  起点的 uint32 偏移；所有 TIFF 偏移都相对 TIFF 起点，不相对载荷起点。
* IFD0 的 0x8825 指向 GPS IFD，IFD0 的 next 指向 IFD1（缩略图 IFD）。
* 使用到的标签名：Make、Model、Orientation、ImageDescription、Software、
  GPSInfo、GPSLatitudeRef、GPSLatitude、GPSLongitudeRef、GPSLongitude、
  Compression、JPEGInterchangeFormat、JPEGInterchangeFormatLength。
"""

from __future__ import annotations

import struct
from fractions import Fraction


MAGIC = b"SEGX"
VERSION = 1
TRAILER_MARKER = 0xFFD9
EXIF_PREFIX = b"Exif\x00\x00"

EXIF_SEGMENT_MARKER = 0xE1

TYPE_BYTE = 1
TYPE_ASCII = 2
TYPE_SHORT = 3
TYPE_LONG = 4
TYPE_RATIONAL = 5

TYPE_SIZE = {
    TYPE_BYTE: 1,
    TYPE_ASCII: 1,
    TYPE_SHORT: 2,
    TYPE_LONG: 4,
    TYPE_RATIONAL: 8,
}

GPS_INFO = 0x8825
MAKE = 0x010F
MODEL = 0x0110
ORIENTATION = 0x0112
IMAGE_DESCRIPTION = 0x010E
SOFTWARE = 0x0131
COMPRESSION = 0x0103
THUMB_OFFSET = 0x0201
THUMB_LENGTH = 0x0202
GPS_LAT_REF = 0x0001
GPS_LAT = 0x0002
GPS_LON_REF = 0x0003
GPS_LON = 0x0004

TAG_NAMES = {
    MAKE: "Make",
    MODEL: "Model",
    ORIENTATION: "Orientation",
    IMAGE_DESCRIPTION: "ImageDescription",
    SOFTWARE: "Software",
    GPS_INFO: "GPSInfo",
    GPS_LAT_REF: "GPSLatitudeRef",
    GPS_LAT: "GPSLatitude",
    GPS_LON_REF: "GPSLongitudeRef",
    GPS_LON: "GPSLongitude",
    COMPRESSION: "Compression",
    THUMB_OFFSET: "JPEGInterchangeFormat",
    THUMB_LENGTH: "JPEGInterchangeFormatLength",
}

TAG_IDS = {name: tag for tag, name in TAG_NAMES.items()}

# 指针类标签始终以单个 LONG 内联值保存。
_POINTER_TAGS = (GPS_INFO, THUMB_OFFSET, THUMB_LENGTH)


def _align2(value: int) -> int:
    return (value + 1) & ~1


def _order(endian: str) -> str:
    if endian == "M":
        return ">"
    if endian == "I":
        return "<"
    raise ValueError("unknown byte order")


# ---------------------------------------------------------------------------
# SEGX 容器
# ---------------------------------------------------------------------------


def iter_segments(data: bytes) -> list[dict]:
    """按 SEGX 容器格式解析全部图像段。"""

    if not isinstance(data, (bytes, bytearray)):
        raise ValueError("image data must be bytes")
    data = bytes(data)
    if len(data) < 8:
        raise ValueError("truncated SEGX header")
    if data[:4] != MAGIC:
        raise ValueError("bad SEGX magic")
    if data[4] != VERSION:
        raise ValueError("unsupported SEGX version")
    try:
        endian = data[5:6].decode("ascii")
        bo = _order(endian)
    except (UnicodeDecodeError, ValueError):
        raise ValueError("unknown byte order") from None

    segments: list[dict] = []
    position = 8
    total = len(data)
    while True:
        if position == total:
            raise ValueError("missing SEGX trailer")
        if position + 6 > total:
            raise ValueError("truncated segment header")
        marker, length = struct.unpack_from(bo + "HI", data, position)
        if marker == TRAILER_MARKER:
            if length != 0:
                raise ValueError("trailer segment has non-zero length")
            position += 6
            if position != total:
                raise ValueError("data present after SEGX trailer")
            break
        payload_start = position + 6
        payload_end = payload_start + length
        if payload_end > total:
            raise ValueError("truncated segment payload")
        segments.append(
            {
                "marker": marker,
                "length": length,
                "offset": position,
                "payload_offset": payload_start,
                "payload": data[payload_start:payload_end],
            }
        )
        position = payload_end
    return segments


def find_exif(segments: list[dict]) -> list[int]:
    """返回所有 EXIF 元数据段在 segments 中的下标，支持零个或多个元数据段。"""

    indexes: list[int] = []
    for index, segment in enumerate(segments):
        if not isinstance(segment, dict):
            raise ValueError("segment must be a mapping")
        if "marker" not in segment or "payload" not in segment:
            raise ValueError("segment missing marker or payload")
        payload = segment["payload"]
        if segment["marker"] == EXIF_SEGMENT_MARKER and payload.startswith(EXIF_PREFIX):
            indexes.append(index)
    return indexes


# ---------------------------------------------------------------------------
# TIFF 读取
# ---------------------------------------------------------------------------


def _u16(data: bytes, offset: int, bo: str) -> int:
    return struct.unpack_from(bo + "H", data, offset)[0]


def _u32(data: bytes, offset: int, bo: str) -> int:
    return struct.unpack_from(bo + "I", data, offset)[0]


def _read_ifd(tiff: bytes, offset: int, bo: str) -> tuple[list[dict], int]:
    size = len(tiff)
    if offset < 8 or offset + 2 > size:
        raise ValueError("IFD offset out of bounds")
    count = _u16(tiff, offset, bo)
    table_end = offset + 2 + 12 * count
    if table_end + 4 > size:
        raise ValueError("IFD table runs past TIFF end")
    entries: list[dict] = []
    for index in range(count):
        entry_offset = offset + 2 + 12 * index
        tag, value_type, item_count = struct.unpack_from(bo + "HHI", tiff, entry_offset)
        element_size = TYPE_SIZE.get(value_type)
        if element_size is None:
            raise ValueError("unknown TIFF type")
        total = element_size * item_count
        if total <= 4:
            raw = tiff[entry_offset + 8 : entry_offset + 8 + total]
        else:
            value_offset = _u32(tiff, entry_offset + 8, bo)
            if value_offset < 8 or value_offset + total > size:
                raise ValueError("value offset out of bounds")
            raw = tiff[value_offset : value_offset + total]
        entries.append(
            {
                "tag": tag,
                "type": value_type,
                "count": item_count,
                "raw": raw,
            }
        )
    next_offset = _u32(tiff, table_end, bo)
    if next_offset and (next_offset < 8 or next_offset + 2 > size):
        raise ValueError("next IFD offset out of bounds")
    return entries, next_offset


def _find_entry(entries: list[dict] | None, tag: int) -> dict | None:
    if entries is None:
        return None
    for entry in entries:
        if entry["tag"] == tag:
            return entry
    return None


def _entry_int(entry: dict, bo: str) -> int:
    if entry["type"] == TYPE_LONG and entry["count"] >= 1:
        return struct.unpack_from(bo + "I", entry["raw"], 0)[0]
    if entry["type"] == TYPE_SHORT and entry["count"] >= 1:
        return struct.unpack_from(bo + "H", entry["raw"], 0)[0]
    if entry["type"] == TYPE_BYTE and entry["count"] >= 1:
        return entry["raw"][0]
    raise ValueError("entry is not an integer value")


def _decode_entry(entry: dict, bo: str):
    value_type = entry["type"]
    count = entry["count"]
    raw = entry["raw"]
    if value_type == TYPE_ASCII:
        return raw.split(b"\x00", 1)[0].decode("ascii")
    if value_type == TYPE_BYTE:
        return tuple(raw[:count])
    if value_type == TYPE_SHORT:
        return struct.unpack(bo + "H" * count, raw[: 2 * count])
    if value_type == TYPE_LONG:
        return struct.unpack(bo + "I" * count, raw[: 4 * count])
    if value_type == TYPE_RATIONAL:
        numbers = struct.unpack(bo + "I" * (count * 2), raw[: 8 * count])
        return tuple(zip(numbers[0::2], numbers[1::2]))
    raise ValueError("unsupported TIFF type")


def _parse_exif(payload: bytes) -> dict:
    if not isinstance(payload, (bytes, bytearray)):
        raise ValueError("EXIF payload must be bytes")
    payload = bytes(payload)
    if not payload.startswith(EXIF_PREFIX):
        raise ValueError("missing EXIF prefix")
    tiff = payload[len(EXIF_PREFIX) :]
    if len(tiff) < 8:
        raise ValueError("truncated TIFF header")
    if tiff[:2] == b"MM":
        endian = "M"
    elif tiff[:2] == b"II":
        endian = "I"
    else:
        raise ValueError("bad TIFF byte order")
    bo = _order(endian)
    if _u16(tiff, 2, bo) != 42:
        raise ValueError("bad TIFF magic")
    ifd0_offset = _u32(tiff, 4, bo)
    visited = {ifd0_offset}
    ifd0, next0 = _read_ifd(tiff, ifd0_offset, bo)

    gps = None
    pointer = _find_entry(ifd0, GPS_INFO)
    if pointer is not None:
        gps_offset = _entry_int(pointer, bo)
        if gps_offset in visited:
            raise ValueError("IFD loop detected")
        visited.add(gps_offset)
        gps, gps_next = _read_ifd(tiff, gps_offset, bo)
        if gps_next != 0:
            raise ValueError("GPS IFD must terminate")

    ifd1 = None
    if next0:
        if next0 in visited:
            raise ValueError("IFD loop detected")
        visited.add(next0)
        ifd1, next1 = _read_ifd(tiff, next0, bo)
        if next1 != 0:
            raise ValueError("IFD1 must terminate")

    thumb = None
    if ifd1 is not None:
        offset_entry = _find_entry(ifd1, THUMB_OFFSET)
        length_entry = _find_entry(ifd1, THUMB_LENGTH)
        if offset_entry is not None and length_entry is not None:
            thumb_offset = _entry_int(offset_entry, bo)
            thumb_length = _entry_int(length_entry, bo)
            if thumb_length:
                if (
                    thumb_offset < 8
                    or thumb_offset + thumb_length > len(tiff)
                ):
                    raise ValueError("thumbnail out of bounds")
                thumb = tiff[thumb_offset : thumb_offset + thumb_length]

    return {
        "tiff": tiff,
        "endian": endian,
        "bo": bo,
        "ifd0": ifd0,
        "gps": gps,
        "ifd1": ifd1,
        "thumb": thumb,
    }


# ---------------------------------------------------------------------------
# TIFF 重建
# ---------------------------------------------------------------------------


def _ifd_bytes(entries: list[dict], next_offset: int, bo: str) -> bytes:
    out = bytearray(struct.pack(bo + "H", len(entries)))
    for entry in entries:
        out += struct.pack(
            bo + "HHI", entry["tag"], entry["type"], entry["count"]
        )
        if len(entry["raw"]) > 4:
            out += struct.pack(bo + "I", entry["_offset"])
        else:
            out += entry["raw"].ljust(4, b"\x00")[:4]
    out += struct.pack(bo + "I", next_offset)
    return bytes(out)


def _build(
    ifd0: list[dict],
    gps: list[dict] | None,
    ifd1: list[dict] | None,
    thumb: bytes | None,
    bo: str,
) -> bytes:
    """按稳定顺序重建 TIFF 数据区并重写全部偏移。"""

    include_gps = gps is not None and _find_entry(ifd0, GPS_INFO) is not None
    if not include_gps:
        gps = None

    has_thumb = (
        ifd1 is not None
        and thumb is not None
        and _find_entry(ifd1, THUMB_OFFSET) is not None
        and _find_entry(ifd1, THUMB_LENGTH) is not None
    )
    if not has_thumb:
        thumb = None

    # 指针标签统一为单个 LONG，稍后写入真实值。
    for entries in (ifd0, gps, ifd1):
        if not entries:
            continue
        for entry in entries:
            if entry["tag"] in _POINTER_TAGS:
                entry["type"] = TYPE_LONG
                entry["count"] = 1
                entry["raw"] = b"\x00\x00\x00\x00"

    ifd0_size = 2 + 12 * len(ifd0) + 4
    gps_size = 2 + 12 * len(gps) + 4 if gps is not None else 0
    ifd1_size = 2 + 12 * len(ifd1) + 4 if ifd1 is not None else 0

    ifd0_offset = 8
    gps_offset = ifd0_offset + ifd0_size if gps is not None else 0
    ifd1_offset = ifd0_offset + ifd0_size + gps_size if ifd1 is not None else 0
    cursor = ifd0_offset + ifd0_size + gps_size + ifd1_size

    blocks: list[dict] = []
    for entries in (ifd0, gps, ifd1):
        if not entries:
            continue
        for entry in entries:
            if len(entry["raw"]) > 4:
                cursor = _align2(cursor)
                entry["_offset"] = cursor
                blocks.append(entry)
                cursor += len(entry["raw"])

    thumb_offset = 0
    if thumb is not None:
        thumb_offset = _align2(cursor)
        cursor += len(thumb)

    for entry in ifd0:
        if entry["tag"] == GPS_INFO:
            entry["raw"] = struct.pack(bo + "I", gps_offset)

    if ifd1 is not None:
        for entry in ifd1:
            if entry["tag"] == THUMB_OFFSET:
                entry["raw"] = struct.pack(bo + "I", thumb_offset)
            elif entry["tag"] == THUMB_LENGTH:
                entry["raw"] = struct.pack(
                    bo + "I", len(thumb) if thumb is not None else 0
                )

    out = bytearray(b"MM" if bo == ">" else b"II")
    out += struct.pack(bo + "H", 42)
    out += struct.pack(bo + "I", ifd0_offset)
    out += _ifd_bytes(ifd0, ifd1_offset if ifd1 is not None else 0, bo)
    if gps is not None:
        out += _ifd_bytes(gps, 0, bo)
    if ifd1 is not None:
        out += _ifd_bytes(ifd1, 0, bo)

    for entry in blocks:
        if len(out) > entry["_offset"]:
            raise ValueError("overlapping TIFF data blocks")
        out += b"\x00" * (entry["_offset"] - len(out))
        out += entry["raw"]

    if thumb is not None:
        if len(out) > thumb_offset:
            raise ValueError("overlapping thumbnail data")
        out += b"\x00" * (thumb_offset - len(out))
        out += thumb

    if len(out) != cursor:
        raise ValueError("TIFF layout length mismatch")
    return EXIF_PREFIX + bytes(out)


def _filter_entries(
    entries: list[dict], drop_ids: set[int]
) -> tuple[list[dict], bool]:
    kept: list[dict] = []
    removed = False
    for entry in entries:
        if entry["tag"] in drop_ids:
            removed = True
        else:
            kept.append(entry)
    return kept, removed


# ---------------------------------------------------------------------------
# 对外接口
# ---------------------------------------------------------------------------


def scrub_fields(payload: bytes, drop: list[str]) -> bytes:
    """从 EXIF 载荷中按标签名删除字段，并返回结构自洽的新载荷。"""

    if not drop:
        return payload
    drop_ids = {
        TAG_IDS[name]
        for name in drop
        if isinstance(name, str) and name in TAG_IDS
    }
    if not drop_ids:
        return payload

    model = _parse_exif(payload)
    ifd0, removed = _filter_entries(model["ifd0"], drop_ids)

    gps = model["gps"]
    if gps is not None:
        gps, gps_removed = _filter_entries(gps, drop_ids)
        removed = removed or gps_removed
        if not gps:
            # GPS IFD 空了，必须同时移除 IFD0 的 GPSInfo 指针。
            ifd0, _ = _filter_entries(ifd0, {GPS_INFO})
            gps = None
            removed = True
        elif _find_entry(ifd0, GPS_INFO) is None:
            gps = None
            removed = True

    ifd1 = model["ifd1"]
    if ifd1 is not None:
        ifd1, ifd1_removed = _filter_entries(ifd1, drop_ids)
        removed = removed or ifd1_removed
        if not ifd1:
            ifd1 = None
            removed = True

    if not removed:
        return payload

    thumb = model["thumb"] if ifd1 is not None else None
    return _build(ifd0, gps, ifd1, thumb, model["bo"])


def drop_thumbnail(payload: bytes) -> bytes:
    """移除 IFD1 及其内嵌缩略图，并清理所有缩略图相关偏移。"""

    model = _parse_exif(payload)
    if model["ifd1"] is None or model["thumb"] is None:
        return payload

    drop_ids = {THUMB_OFFSET, THUMB_LENGTH}
    ifd0, _ = _filter_entries(model["ifd0"], drop_ids)
    gps = None
    if model["gps"] is not None:
        gps, _ = _filter_entries(model["gps"], drop_ids)
        if not gps:
            ifd0, _ = _filter_entries(ifd0, {GPS_INFO})
            gps = None

    return _build(ifd0, gps, None, None, model["bo"])


def rewrite_offsets(payload: bytes) -> bytes:
    """重建 EXIF 载荷的偏移表，删除没有引用的尾部字节。"""

    model = _parse_exif(payload)
    return _build(
        model["ifd0"],
        model["gps"],
        model["ifd1"],
        model["thumb"],
        model["bo"],
    )


def _format_six(value: Fraction) -> str:
    sign = "-" if value < 0 else ""
    magnitude = -value if value < 0 else value
    scaled = magnitude * 1_000_000
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder >= scaled.denominator:
        quotient += 1
    whole, fraction = divmod(quotient, 1_000_000)
    if quotient == 0:
        sign = ""
    return f"{sign}{whole}.{fraction:06d}"


def _dms_to_fraction(values) -> Fraction:
    total = Fraction(0)
    for power, (numerator, denominator) in enumerate(values[:3]):
        if denominator == 0:
            raise ValueError("zero rational denominator")
        total += Fraction(numerator, denominator) / (60**power)
    return total


def _gps_coordinate(
    gps: list[dict], value_tag: int, ref_tag: int, bo: str
) -> Fraction | None:
    entry = _find_entry(gps, value_tag)
    if entry is None:
        return None
    values = _decode_entry(entry, bo)
    coordinate = _dms_to_fraction(values)
    ref_entry = _find_entry(gps, ref_tag)
    if ref_entry is not None and ref_entry["type"] == TYPE_ASCII:
        reference = _decode_entry(ref_entry, bo).strip().upper()
        if reference in ("S", "W"):
            coordinate = -coordinate
    return coordinate


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def verify_clean(data: bytes, forbidden: dict) -> list[str]:
    """校验清洗后的图像是否仍含禁止内容，返回按固定顺序排列的发现列表。"""

    if not isinstance(forbidden, dict):
        raise ValueError("forbidden must be a mapping")
    tag_names = forbidden.get("tags") or []
    coordinate_literals = forbidden.get("coordinates") or []
    device_values = forbidden.get("devices") or []
    tag_set = set(tag_names)
    coordinate_set = set(coordinate_literals)
    device_set = set(device_values)

    segments = iter_segments(data)
    tag_findings: list[str] = []
    coordinate_findings: list[str] = []
    device_findings: list[str] = []

    for index in find_exif(segments):
        model = _parse_exif(segments[index]["payload"])
        for entries in (model["ifd0"], model["gps"], model["ifd1"]):
            if entries is None:
                continue
            for entry in entries:
                name = TAG_NAMES.get(entry["tag"])
                if name is not None and name in tag_set:
                    tag_findings.append("tag:" + name)
                if entry["type"] == TYPE_ASCII:
                    value = _decode_entry(entry, model["bo"])
                    if value in device_set:
                        device_findings.append("device:" + value)
        if model["gps"] is not None:
            latitude = _gps_coordinate(
                model["gps"], GPS_LAT, GPS_LAT_REF, model["bo"]
            )
            longitude = _gps_coordinate(
                model["gps"], GPS_LON, GPS_LON_REF, model["bo"]
            )
            if latitude is not None and longitude is not None:
                literal = f"{_format_six(latitude)},{_format_six(longitude)}"
                if literal in coordinate_set:
                    coordinate_findings.append("coordinate:" + literal)

    return (
        _dedupe(tag_findings)
        + _dedupe(coordinate_findings)
        + _dedupe(device_findings)
    )


def validate_policy(policy: dict) -> list[str]:
    """检查清洗策略，返回问题代码列表；合法策略返回空列表。"""

    if not isinstance(policy, dict):
        return ["policy_not_mapping"]

    issues: list[str] = []

    drop = policy.get("drop", [])
    if "drop" in policy and not isinstance(drop, list):
        issues.append("drop_not_list")

    forbid = policy.get("forbid", {})
    if "forbid" in policy and not isinstance(forbid, dict):
        issues.append("forbid_not_dict")

    tags = forbid.get("tags", []) if isinstance(forbid, dict) else []
    if not (
        isinstance(tags, list)
        and "GPSLatitude" in tags
        and "GPSLongitude" in tags
    ):
        issues.append("coordinates_not_forbidden")

    if policy.get("drop_thumbnail") is not True:
        issues.append("thumbnail_not_removed")

    if policy.get("verify_residue") is not True:
        issues.append("residue_not_verified")

    return issues


def _rebuild_image(image: bytes, segments: list[dict], payloads: list[bytes]) -> bytes:
    endian = image[5:6].decode("ascii")
    bo = _order(endian)
    out = bytearray(image[:8])
    for segment, payload in zip(segments, payloads):
        out += struct.pack(bo + "HI", segment["marker"], len(payload))
        out += payload
    out += struct.pack(bo + "HI", TRAILER_MARKER, 0)
    return bytes(out)


def _scrub_one(
    image: bytes,
    drop: list[str],
    forbid: dict,
    remove_thumb: bool,
    do_verify: bool,
) -> dict:
    segments = iter_segments(image)
    indexes = find_exif(segments)
    if not indexes:
        return {"data": image, "ok": True, "errors": [], "residues": []}

    exif_indexes = set(indexes)
    payloads: list[bytes] = []
    for index, segment in enumerate(segments):
        payload = segment["payload"]
        if index in exif_indexes:
            payload = scrub_fields(payload, drop)
            if remove_thumb:
                payload = drop_thumbnail(payload)
            payload = rewrite_offsets(payload)
        payloads.append(payload)

    rebuilt = _rebuild_image(image, segments, payloads)

    expected_total = 8 + sum(6 + len(payload) for payload in payloads) + 6
    mismatch = len(rebuilt) != expected_total
    try:
        check = iter_segments(rebuilt)
    except ValueError:
        mismatch = True
        check = []
    if len(check) != len(segments):
        mismatch = True
    else:
        for got, want in zip(check, segments):
            if got["marker"] != want["marker"]:
                mismatch = True
                break
        if not mismatch:
            for got, payload in zip(check, payloads):
                if got["length"] != len(payload) or got["payload"] != payload:
                    mismatch = True
                    break

    if mismatch:
        return {
            "data": image,
            "ok": False,
            "errors": ["output_length_mismatch"],
            "residues": [],
        }

    residues = verify_clean(rebuilt, forbid) if do_verify else []
    if residues:
        return {
            "data": rebuilt,
            "ok": False,
            "errors": [],
            "residues": residues,
        }
    return {"data": rebuilt, "ok": True, "errors": [], "residues": []}


def scrub_batch(images: dict[str, bytes], policy: dict) -> dict:
    """批量清洗图像，每张图独立解析、独立重建，返回同键名结果字典。"""

    problems = validate_policy(policy)
    if problems:
        raise ValueError("invalid policy: " + ", ".join(problems))

    drop = policy.get("drop") or []
    forbid = policy.get("forbid") or {}
    remove_thumb = policy.get("drop_thumbnail") is True
    do_verify = policy.get("verify_residue") is True

    results: dict[str, dict] = {}
    for name, image in images.items():
        try:
            results[name] = _scrub_one(
                image, drop, forbid, remove_thumb, do_verify
            )
        except (ValueError, struct.error) as exc:
            results[name] = {
                "data": image,
                "ok": False,
                "errors": [str(exc)],
                "residues": [],
            }
    return results
