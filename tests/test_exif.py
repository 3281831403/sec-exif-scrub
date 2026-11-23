import struct

from scrub.exif import (
    drop_thumbnail,
    find_exif,
    iter_segments,
    rewrite_offsets,
    scrub_batch,
    scrub_fields,
    validate_policy,
    verify_clean,
)


MAGIC = b"SEGX"
VERSION = 1
APP0 = 0xE0
APP1 = 0xE1
APP2 = 0xE2
TRAILER = 0xFFD9
EXIF_PREFIX = b"Exif\x00\x00"

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
COMPRESSION = 0x0103
THUMB_OFFSET = 0x0201
THUMB_LENGTH = 0x0202
GPS_LAT_REF = 0x0001
GPS_LAT = 0x0002
GPS_LON_REF = 0x0003
GPS_LON = 0x0004

THUMBNAIL = b"\xff\xd8THUMBNAIL-UNIQUE-9f3a\xff\xd9"
COORD_LITERAL = "39.904200,116.407400"


def _order(endian):
    if endian == "M":
        return ">"
    if endian == "I":
        return "<"
    raise ValueError("endian must be M or I")


def make_segment(marker, payload, endian="M"):
    return struct.pack(_order(endian) + "HI", marker, len(payload)) + payload


def make_image(segments, endian="M"):
    header = MAGIC + bytes([VERSION]) + endian.encode("ascii") + b"\x00\x00"
    trailer = struct.pack(_order(endian) + "HI", TRAILER, 0)
    return header + b"".join(segments) + trailer


def _encode_value(value_type, values, endian):
    bo = _order(endian)
    if value_type == TYPE_BYTE:
        if isinstance(values, int):
            return bytes([values])
        return bytes(values)
    if value_type == TYPE_ASCII:
        if isinstance(values, str):
            return values.encode("ascii") + b"\x00"
        raw = bytes(values)
        return raw if raw.endswith(b"\x00") else raw + b"\x00"
    if value_type == TYPE_SHORT:
        return struct.pack(bo + "H" * len(values), *values)
    if value_type == TYPE_LONG:
        return struct.pack(bo + "I" * len(values), *values)
    if value_type == TYPE_RATIONAL:
        raw = bytearray()
        for numerator, denominator in values:
            raw += struct.pack(bo + "II", numerator, denominator)
        return bytes(raw)
    raise ValueError("unsupported TIFF type")


def _record(tag, value_type, values, endian):
    raw = _encode_value(value_type, values, endian)
    if value_type in (TYPE_BYTE, TYPE_ASCII):
        count = len(raw)
    else:
        count = len(values)
    return {
        "tag": tag,
        "type": value_type,
        "count": count,
        "raw": raw,
        "offset": None,
    }


def _long_record(tag, value, endian):
    return _record(tag, TYPE_LONG, (value,), endian)


def _ifd_size(records):
    return 2 + 12 * len(records) + 4


def _ifd_bytes(records, next_offset, endian):
    bo = _order(endian)
    out = bytearray(struct.pack(bo + "H", len(records)))
    for record in records:
        out += struct.pack(
            bo + "HHI",
            record["tag"],
            record["type"],
            record["count"],
        )
        if record["offset"] is None:
            out += record["raw"].ljust(4, b"\x00")[:4]
        else:
            out += struct.pack(bo + "I", record["offset"])
    out += struct.pack(bo + "I", next_offset)
    return bytes(out)


def _align2(value):
    return (value + 1) & ~1


def build_exif(ifd0=(), gps=None, ifd1=None, thumb=b"", endian="M"):
    """Build a synthetic EXIF payload with zero or one thumbnail."""
    bo = _order(endian)
    ifd0_records = [_record(tag, typ, values, endian) for tag, typ, values in ifd0]
    gps_records = (
        [_record(tag, typ, values, endian) for tag, typ, values in gps]
        if gps is not None
        else None
    )
    make_ifd1 = ifd1 is not None or bool(thumb)
    ifd1_records = (
        [_record(tag, typ, values, endian) for tag, typ, values in (ifd1 or [])]
        if make_ifd1
        else None
    )

    if gps_records is not None and not any(r["tag"] == GPS_INFO for r in ifd0_records):
        ifd0_records.append(_long_record(GPS_INFO, 0, endian))
    if ifd1_records is not None and thumb:
        ifd1_records = [
            r for r in ifd1_records if r["tag"] not in (THUMB_OFFSET, THUMB_LENGTH)
        ]
        ifd1_records.append(_long_record(THUMB_OFFSET, 0, endian))
        ifd1_records.append(_long_record(THUMB_LENGTH, len(thumb), endian))

    cursor = 8
    ifd0_offset = cursor
    cursor += _ifd_size(ifd0_records)
    gps_offset = 0
    if gps_records is not None:
        gps_offset = cursor
        cursor += _ifd_size(gps_records)
    ifd1_offset = 0
    if ifd1_records is not None:
        ifd1_offset = cursor
        cursor += _ifd_size(ifd1_records)
    data_start = cursor

    all_records = list(ifd0_records)
    if gps_records is not None:
        all_records += gps_records
    if ifd1_records is not None:
        all_records += ifd1_records
    for record in all_records:
        if len(record["raw"]) > 4:
            data_start = _align2(data_start)
            record["offset"] = data_start
            data_start += len(record["raw"])

    thumb_offset = _align2(data_start) if thumb else 0
    for record in ifd0_records:
        if record["tag"] == GPS_INFO:
            record["raw"] = struct.pack(bo + "I", gps_offset)
    if ifd1_records is not None:
        for record in ifd1_records:
            if record["tag"] == THUMB_OFFSET:
                record["raw"] = struct.pack(bo + "I", thumb_offset)

    tiff = bytearray(b"MM" if endian == "M" else b"II")
    tiff += struct.pack(bo + "H", 42)
    tiff += struct.pack(bo + "I", 8)
    tiff += _ifd_bytes(
        ifd0_records,
        ifd1_offset if ifd1_records is not None else 0,
        endian,
    )
    if gps_records is not None:
        tiff += _ifd_bytes(gps_records, 0, endian)
    if ifd1_records is not None:
        tiff += _ifd_bytes(ifd1_records, 0, endian)

    data = bytearray()
    for record in all_records:
        if record["offset"] is not None:
            while len(tiff) + len(data) < record["offset"]:
                data.append(0)
            if len(tiff) + len(data) != record["offset"]:
                raise AssertionError("test builder layout error")
            data += record["raw"]
    if thumb:
        while len(tiff) + len(data) < thumb_offset:
            data.append(0)
        data += thumb
    return EXIF_PREFIX + bytes(tiff) + bytes(data)


def sample_exif(
    endian="M",
    with_gps=True,
    thumb=b"",
    description="Canonical holiday photo",
):
    ifd0 = [
        (MAKE, TYPE_ASCII, "ACME Cam"),
        (MODEL, TYPE_ASCII, "ACME X1"),
        (ORIENTATION, TYPE_SHORT, (1,)),
        (IMAGE_DESCRIPTION, TYPE_ASCII, description),
    ]
    gps = None
    if with_gps:
        gps = [
            (GPS_LAT_REF, TYPE_ASCII, "N"),
            (GPS_LAT, TYPE_RATIONAL, ((39, 1), (54, 1), (1512, 100))),
            (GPS_LON_REF, TYPE_ASCII, "E"),
            (GPS_LON, TYPE_RATIONAL, ((116, 1), (24, 1), (2664, 100))),
        ]
    ifd1 = [(COMPRESSION, TYPE_SHORT, (6,))] if thumb else None
    return build_exif(ifd0=ifd0, gps=gps, ifd1=ifd1, thumb=thumb, endian=endian)


def clean_policy(drop=None):
    tags = [
        "Make",
        "Model",
        "GPSLatitude",
        "GPSLongitude",
        "GPSLatitudeRef",
        "GPSLongitudeRef",
    ]
    return {
        "drop": list(tags if drop is None else drop),
        "forbid": {"tags": list(tags), "coordinates": [], "devices": []},
        "drop_thumbnail": True,
        "verify_residue": True,
    }


def _u16(data, offset, endian):
    return struct.unpack_from(_order(endian) + "H", data, offset)[0]


def _u32(data, offset, endian):
    return struct.unpack_from(_order(endian) + "I", data, offset)[0]


def _decode_raw(value_type, raw, count, endian):
    bo = _order(endian)
    if value_type == TYPE_BYTE:
        return tuple(raw)
    if value_type == TYPE_ASCII:
        return raw.split(b"\x00", 1)[0].decode("ascii")
    if value_type == TYPE_SHORT:
        return struct.unpack(bo + "H" * count, raw)
    if value_type == TYPE_LONG:
        return struct.unpack(bo + "I" * count, raw)
    if value_type == TYPE_RATIONAL:
        numbers = struct.unpack(bo + "I" * (count * 2), raw)
        return tuple(zip(numbers[0::2], numbers[1::2]))
    raise AssertionError("unsupported TIFF type")


def _read_ifd(tiff, offset, endian):
    if not (8 <= offset <= len(tiff) - 2):
        raise AssertionError("IFD offset out of bounds")
    count = _u16(tiff, offset, endian)
    table_end = offset + 2 + 12 * count + 4
    if table_end > len(tiff):
        raise AssertionError("IFD table runs past TIFF end")
    entries = {}
    for index in range(count):
        entry_offset = offset + 2 + 12 * index
        tag = _u16(tiff, entry_offset, endian)
        value_type = _u16(tiff, entry_offset + 2, endian)
        item_count = _u32(tiff, entry_offset + 4, endian)
        if value_type not in TYPE_SIZE:
            raise AssertionError("unknown TIFF type")
        total = TYPE_SIZE[value_type] * item_count
        if total <= 4:
            value_offset = None
            raw = tiff[entry_offset + 8 : entry_offset + 8 + total]
        else:
            value_offset = _u32(tiff, entry_offset + 8, endian)
            if value_offset < 8 or value_offset + total > len(tiff):
                raise AssertionError("value offset out of bounds")
            raw = tiff[value_offset : value_offset + total]
        entries[tag] = {
            "tag": tag,
            "type": value_type,
            "count": item_count,
            "values": _decode_raw(value_type, raw, item_count, endian),
            "raw": raw,
            "value_offset": value_offset,
            "entry_offset": entry_offset,
        }
    next_offset = _u32(tiff, table_end - 4, endian)
    if next_offset and not (8 <= next_offset <= len(tiff) - 2):
        raise AssertionError("next IFD offset out of bounds")
    return entries, next_offset, (offset, table_end)


def parse_exif(payload):
    if not payload.startswith(EXIF_PREFIX):
        raise AssertionError("missing EXIF prefix")
    tiff = payload[len(EXIF_PREFIX) :]
    if tiff[:2] == b"MM":
        endian = "M"
    elif tiff[:2] == b"II":
        endian = "I"
    else:
        raise AssertionError("bad TIFF byte order")
    if _u16(tiff, 2, endian) != 42:
        raise AssertionError("bad TIFF magic")
    ifd0_offset = _u32(tiff, 4, endian)
    ifd0, next0, ifd0_range = _read_ifd(tiff, ifd0_offset, endian)
    gps = None
    gps_range = None
    if GPS_INFO in ifd0:
        gps_offset = ifd0[GPS_INFO]["values"][0]
        gps, gps_next, gps_range = _read_ifd(tiff, gps_offset, endian)
        if gps_next != 0:
            raise AssertionError("GPS IFD must terminate")
    ifd1 = None
    ifd1_range = None
    if next0:
        ifd1, next1, ifd1_range = _read_ifd(tiff, next0, endian)
        if next1 != 0:
            raise AssertionError("IFD1 must terminate")
    return {
        "tiff": tiff,
        "endian": endian,
        "ifd0": ifd0,
        "gps": gps,
        "ifd1": ifd1,
        "ifd0_next": next0,
        "ranges": [r for r in (ifd0_range, gps_range, ifd1_range) if r],
    }


def assert_layout(payload):
    """Parse and require every out-of-line value to be in bounds and non-overlapping."""
    parsed = parse_exif(payload)
    datablocks = []
    for ifd in (parsed["ifd0"], parsed["gps"], parsed["ifd1"]):
        if ifd is None:
            continue
        for entry in ifd.values():
            if entry["value_offset"] is not None:
                start = entry["value_offset"]
                datablocks.append((start, start + len(entry["raw"]), entry["tag"]))
    for start, end, _ in datablocks:
        for range_start, range_end in parsed["ranges"]:
            if start < range_end and range_start < end:
                raise AssertionError("data block overlaps an IFD table")
    ordered = sorted(datablocks)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise AssertionError("data blocks overlap")
    thumb = None
    if parsed["ifd1"] and THUMB_OFFSET in parsed["ifd1"]:
        thumb_offset = parsed["ifd1"][THUMB_OFFSET]["values"][0]
        thumb_length = parsed["ifd1"][THUMB_LENGTH]["values"][0]
        if thumb_offset + thumb_length > len(parsed["tiff"]):
            raise AssertionError("thumbnail out of bounds")
        thumb = parsed["tiff"][thumb_offset : thumb_offset + thumb_length]
    return parsed, thumb


def only_exif_payload(data):
    segments = iter_segments(data)
    indexes = find_exif(segments)
    if len(indexes) != 1:
        raise AssertionError("expected exactly one EXIF segment")
    return segments[indexes[0]]["payload"]


def test_collects_all_segments_not_only_first():
    first = sample_exif(with_gps=False, description="first")
    second = sample_exif(with_gps=False, description="second")
    data = make_image(
        [
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP1, first),
            make_segment(APP2, b"ICC_PROFILE\x00"),
            make_segment(APP1, second),
        ]
    )

    segments = iter_segments(data)

    assert [item["marker"] for item in segments] == [APP0, APP1, APP2, APP1]
    assert [item["payload"] for item in segments] == [
        b"JFIF\x00",
        first,
        b"ICC_PROFILE\x00",
        second,
    ]
    assert segments[0]["offset"] == 8
    assert segments[1]["offset"] == 8 + 6 + len(b"JFIF\x00")
    assert segments[1]["payload_offset"] == segments[1]["offset"] + 6
    assert segments[1]["length"] == len(first)


def test_uses_declared_little_endian_for_segment_lengths():
    long_description = "L" * 300
    payload = sample_exif(
        endian="I", with_gps=False, description=long_description
    )
    data = make_image([make_segment(APP1, payload, endian="I")], endian="I")

    segments = iter_segments(data)

    assert len(segments) == 1
    assert segments[0]["marker"] == APP1
    assert segments[0]["length"] == len(payload)
    assert segments[0]["length"] > 255
    assert segments[0]["payload"] == payload
    assert segments[0]["payload_offset"] == 14


def test_find_exif_ignores_leading_non_exif_segments():
    first = sample_exif(with_gps=False, description="one")
    second = sample_exif(with_gps=False, description="two")
    data = make_image(
        [
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP2, EXIF_PREFIX + b"decoy"),
            make_segment(APP1, first),
            make_segment(APP1, second),
        ]
    )

    segments = iter_segments(data)

    assert find_exif(segments) == [2, 3]
    assert find_exif([segments[0], segments[1]]) == []


def test_scrub_fields_keeps_offsets_consistent():
    payload = sample_exif(with_gps=True)

    cleaned = scrub_fields(payload, ["Make"])

    parsed, _ = assert_layout(cleaned)
    assert MAKE not in parsed["ifd0"]
    assert parsed["ifd0"][MODEL]["values"] == "ACME X1"
    assert parsed["ifd0"][ORIENTATION]["values"] == (1,)
    assert parsed["ifd0"][IMAGE_DESCRIPTION]["values"] == "Canonical holiday photo"
    assert parsed["gps"] is not None
    assert parsed["gps"][GPS_LAT]["values"] == ((39, 1), (54, 1), (1512, 100))
    assert b"ACME Cam" not in cleaned


def test_drop_thumbnail_clears_pointer_and_payload():
    cleaned = drop_thumbnail(sample_exif(thumb=THUMBNAIL, with_gps=True))

    parsed, thumb = assert_layout(cleaned)
    assert parsed["ifd0_next"] == 0
    assert parsed["ifd1"] is None
    assert parsed["gps"] is not None
    assert THUMB_OFFSET not in parsed["ifd0"]
    assert THUMB_LENGTH not in parsed["ifd0"]
    assert thumb is None
    assert THUMBNAIL not in cleaned


def test_verify_clean_detects_coordinate_values():
    without_gps = sample_exif(with_gps=False, description="no location")
    with_gps = sample_exif(with_gps=True, description="located")
    data = make_image(
        [
            make_segment(APP1, without_gps),
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP1, with_gps),
        ]
    )

    findings = verify_clean(
        data,
        {
            "tags": ["GPSLatitude"],
            "coordinates": [COORD_LITERAL],
            "devices": [],
        },
    )

    assert findings == ["tag:GPSLatitude", "coordinate:" + COORD_LITERAL]


def test_verify_clean_does_not_match_substrings():
    data = make_image([make_segment(APP1, sample_exif(with_gps=False))])

    assert (
        verify_clean(
            data,
            {"tags": [], "coordinates": [], "devices": ["Canon"]},
        )
        == []
    )
    assert verify_clean(
        data,
        {"tags": [], "coordinates": [], "devices": ["ACME Cam"]},
    ) == ["device:ACME Cam"]


def test_scrub_batch_does_not_reuse_segments_between_images():
    image_a = make_image(
        [
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP1, sample_exif(description="A", with_gps=True)),
        ]
    )
    image_b = make_image(
        [
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP2, b"ICC_PROFILE\x00"),
            make_segment(APP1, sample_exif(description="B", with_gps=False)),
        ]
    )

    together = scrub_batch({"a": image_a, "b": image_b}, clean_policy())
    alone = scrub_batch({"b": image_b}, clean_policy())

    assert together["b"] == alone["b"]
    assert together["b"]["ok"] is True
    cleaned_segments = iter_segments(together["b"]["data"])
    assert [item["marker"] for item in cleaned_segments] == [APP0, APP2, APP1]


def test_validate_policy_requires_thumbnail_handling():
    assert validate_policy(clean_policy()) == []

    policy = clean_policy()
    policy["drop_thumbnail"] = False
    assert validate_policy(policy) == ["thumbnail_not_removed"]

    assert validate_policy({}) == [
        "coordinates_not_forbidden",
        "thumbnail_not_removed",
        "residue_not_verified",
    ]


def test_scrub_batch_reports_truncated_input():
    good = make_image([make_segment(APP1, sample_exif(with_gps=True))])
    broken = good[:-10]

    results = scrub_batch({"bad": broken, "good": good}, clean_policy())

    assert results["bad"]["ok"] is False
    assert any("truncated" in error.lower() for error in results["bad"]["errors"])
    assert results["good"]["ok"] is True
    assert results["good"]["data"].startswith(MAGIC)
    rebuilt = results["good"]["data"]
    rebuilt_segments = iter_segments(rebuilt)
    assert len(rebuilt) == 8 + sum(
        6 + len(segment["payload"]) for segment in rebuilt_segments
    ) + 6


def test_image_without_exif_is_returned_unchanged():
    plain = make_image(
        [
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP2, b"ICC_PROFILE\x00"),
        ]
    )

    result = scrub_batch({"plain": plain}, clean_policy())["plain"]

    assert result["ok"] is True
    assert result["data"] == plain
    assert result["errors"] == []
    assert result["residues"] == []


def test_single_exif_segment_is_cleared():
    data = make_image([make_segment(APP1, sample_exif(with_gps=True))])
    policy = clean_policy()

    result = scrub_batch({"one": data}, policy)["one"]

    assert result["ok"] is True
    parsed, _ = assert_layout(only_exif_payload(result["data"]))
    assert MAKE not in parsed["ifd0"]
    assert MODEL not in parsed["ifd0"]
    assert GPS_INFO not in parsed["ifd0"]
    assert parsed["gps"] is None
    assert verify_clean(result["data"], policy["forbid"]) == []


def test_thumbnail_is_removed_from_clean_output():
    data = make_image(
        [make_segment(APP1, sample_exif(thumb=THUMBNAIL, with_gps=False))]
    )
    policy = {
        "drop": [],
        "forbid": {
            "tags": ["GPSLatitude", "GPSLongitude"],
            "coordinates": [],
            "devices": [],
        },
        "drop_thumbnail": True,
        "verify_residue": True,
    }

    result = scrub_batch({"photo": data}, policy)["photo"]

    assert result["ok"] is True
    assert THUMBNAIL not in result["data"]
    parsed, thumb = assert_layout(only_exif_payload(result["data"]))
    assert parsed["ifd1"] is None
    assert thumb is None


def test_rewrite_offsets_is_self_consistent_and_idempotent():
    payload = sample_exif(thumb=THUMBNAIL, with_gps=True)
    padded = payload + b"UNREFERENCED-TRAILING-BYTES"

    once = rewrite_offsets(padded)
    twice = rewrite_offsets(once)

    assert once == twice
    assert b"UNREFERENCED-TRAILING-BYTES" not in once
    assert len(once) < len(padded)
    parsed, thumb = assert_layout(once)
    assert thumb == THUMBNAIL
    assert parsed["ifd0"][MAKE]["values"] == "ACME Cam"
    assert parsed["gps"][GPS_LON]["values"] == ((116, 1), (24, 1), (2664, 100))


def test_two_batches_match_individual_runs():
    image_a = make_image([make_segment(APP1, sample_exif(with_gps=True))])
    image_b = make_image(
        [
            make_segment(APP0, b"JFIF\x00"),
            make_segment(APP1, sample_exif(thumb=THUMBNAIL, with_gps=False)),
        ]
    )
    policy = clean_policy()

    forward = scrub_batch({"a": image_a, "b": image_b}, policy)
    reverse = scrub_batch({"b": image_b, "a": image_a}, policy)
    solo_a = scrub_batch({"a": image_a}, policy)
    solo_b = scrub_batch({"b": image_b}, policy)

    assert forward["a"] == reverse["a"] == solo_a["a"]
    assert forward["b"] == reverse["b"] == solo_b["b"]
    assert forward["a"]["ok"] is True
    assert forward["b"]["ok"] is True
