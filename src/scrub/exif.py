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

import hashlib
import re
import struct


MAGIC = b"SEGX"
VERSION = 1
TRAILER_MARKER = 0xFFD9
EXIF_PREFIX = b"Exif\x00\x00"


def iter_segments(data: bytes) -> list[dict]:
    """按 SEGX 容器格式解析全部图像段。

    文件头为 8 字节，段头为 2 字节标记加 4 字节长度。标记和长度必须按文件头
    第 6 字节声明的字节序解释：b"M" 为大端，b"I" 为小端；不能固定按大端读取。
    长度字段只表示载荷字节数，不包含 6 字节段头。段偏移以文件起点为 0。

    返回列表中的每个字典恰好包含：
        marker: int，段标记；
        length: int，载荷长度；
        offset: int，段头首字节的绝对偏移；
        payload_offset: int，载荷首字节的绝对偏移，等于 offset 加 6；
        payload: bytes，段载荷本身。
    终止段 0xFFD9 不返回，紧跟在终止段之后的任何字节都视为损坏。

    遇到以下情况抛出 ValueError：输入短于文件头、魔数或版本错误、字节序标志
    未知、段头被截断、长度越过文件末尾、终止段缺失或长度非零、终止段后仍有
    数据、同一文件出现多个终止段。空段列表是合法的。

    该函数必须返回所有段，不能只返回第一段；后续 find_exif 依赖完整列表。
    """

    raise NotImplementedError


def find_exif(segments: list[dict]) -> list[int]:
    """返回所有 EXIF 元数据段在 segments 中的下标，支持零个或多个元数据段。

    判定条件必须同时满足：marker 等于 0xE1，且 payload 以 b"Exif\\x00\\x00"
    开头。不能假设 EXIF 位于固定下标或固定文件偏移。返回下标按 segments 的
    原始顺序排列；没有元数据时返回空列表。

    参数 segments 必须是 iter_segments 的返回值。若某个字典缺少 marker 或
    payload 字段，抛出 ValueError。
    """

    raise NotImplementedError


def scrub_fields(payload: bytes, drop: list[str]) -> bytes:
    """从 EXIF 载荷中按标签名删除字段，并返回结构自洽的新载荷。

    payload 必须以 b"Exif\\x00\\x00" 开头并包含合法 TIFF 结构。drop 中的名称
    区分大小写，只接受模块说明列出的标签名；未知名称忽略。IFD0、GPS IFD 与
    IFD1 中同名标签都要删除。删除 GPS 字段后如果 GPS IFD 为空，必须同时移除
    IFD0 的 GPSInfo 指针；删除字段后如果 IFD1 为空，则移除 IFD1。

    返回值必须继续以 b"Exif\\x00\\x00" 开头，且删除条目后要重建偏移表：
    所有保留下来的 ASCII、SHORT、LONG、RATIONAL 数据偏移都要指向新位置，
    不能留下指向被删条目的悬空指针，也不能把旧偏移原样复制。返回载荷必须
    可以被独立解析，不需要调用方再补救。

    drop 为空时原样返回 payload。输入损坏或偏移越界时抛出 ValueError。
    """

    raise NotImplementedError


def drop_thumbnail(payload: bytes) -> bytes:
    """移除 IFD1 及其内嵌缩略图，并清理所有缩略图相关偏移。

    必须完成四件事：从 IFD0 断开 next IFD 指针；删除 IFD1 整张表；删除
    JPEGInterchangeFormat 与 JPEGInterchangeFormatLength 标签；从 TIFF 数据区
    删除缩略图字节。随后重建偏移表，任何残留偏移都不能再指向旧缩略图位置。

    返回载荷仍以 b"Exif\\x00\\x00" 开头，其余 IFD0 和 GPS 字段保持不变且可
    解析。没有缩略图或没有 IFD1 时原样返回 payload。输入损坏时抛出 ValueError。
    """

    raise NotImplementedError


def rewrite_offsets(payload: bytes) -> bytes:
    """重建 EXIF 载荷的偏移表，删除没有引用的尾部字节。

    按稳定顺序重新布局：TIFF 头、IFD0、GPS IFD、IFD1、各条目的外部数据、
    缩略图。所有 out-of-line 数据偏移和 next IFD 指针都相对 TIFF 起点重新
    计算；数据区按 2 字节对齐；不再被任何条目引用的字节不进入结果。

    函数必须幂等：rewrite_offsets(rewrite_offsets(payload)) 与只调用一次完全
    相同。已有缩略图要继续可读，缩略图偏移和长度必须同步更新。输入损坏、
    偏移越界或 IFD 循环时抛出 ValueError。

    payload 必须以 b"Exif\\x00\\x00" 开头。返回载荷保持该前缀。
    """

    raise NotImplementedError


def verify_clean(data: bytes, forbidden: dict) -> list[str]:
    """校验清洗后的图像是否仍含禁止内容，返回按固定顺序排列的发现列表。

    forbidden 是字典，允许三个键，缺省视为空列表：
        tags: 标签名字符串列表；
        coordinates: 坐标字面量列表，每项形如 纬度,经度；
        devices: 设备字符串列表。

    检查规则：
    * tags 中的标签名只要出现在任一 EXIF 段的任一 IFD 中，就产生
      tag:标签名。
    * coordinates 必须从 GPSLatitude 与 GPSLongitude 的度分秒有理数精确
      换算，容差 0。坐标字面量固定保留 6 位小数，例如 39.904200,116.407400；
      只有换算值与字面量逐位相同时才产生 coordinate:纬度,经度。只检查标签名
      是否存在、不比较实际坐标值的实现是不合格的。
    * devices 与 ASCII 字段的完整值比较，必须整值相等，不能做子串匹配。
      例如字段值 Canonical photo 不包含设备值 Canon。命中时产生
      device:字段值。

    结果先去重，再按 tags、coordinates、devices 的输入顺序排列；同一类内按
    图像中的段顺序和 IFD 顺序排列。没有 EXIF 或没有命中时返回空列表。
    图像损坏或 EXIF 载荷损坏时抛出 ValueError。
    """

    raise NotImplementedError


def validate_policy(policy: dict) -> list[str]:
    """检查清洗策略，返回问题代码列表；合法策略返回空列表。

    policy 至少包含：
        drop: 要删除的标签名列表；
        forbid: 传给 verify_clean 的禁止内容字典；
        drop_thumbnail: 是否移除缩略图；
        verify_residue: 是否在输出上执行残留校验。

    必须报告以下问题，并按此固定顺序返回：
        coordinates_not_forbidden
            forbid.tags 没有同时包含 GPSLatitude 与 GPSLongitude。
        thumbnail_not_removed
            drop_thumbnail 不为 True。
        residue_not_verified
            verify_residue 不为 True。

    policy 不是字典时返回 policy_not_mapping。字段类型明显错误时分别返回
    drop_not_list、forbid_not_dict。不能因为策略缺少键而抛出 KeyError。
    """

    raise NotImplementedError


def scrub_batch(images: dict[str, bytes], policy: dict) -> dict:
    """批量清洗图像，每张图独立解析、独立重建，返回同键名结果字典。

    处理顺序：先调用 validate_policy；有问题立即抛出 ValueError，不处理任何
    图像。然后对每张图重新调用 iter_segments，不能复用上一张图的段列表。
    对 find_exif 找到的每个 EXIF 段依次执行 scrub_fields、按策略执行
    drop_thumbnail、rewrite_offsets，再按原字节序重建整个 SEGX 容器；非 EXIF
    段逐字节保留。没有 EXIF 的图像必须原样返回。

    每个结果字典包含：
        data: bytes，清洗后的图像；失败时为原始输入；
        ok: bool，是否成功且没有残留；
        errors: list[str]，失败原因，成功时为空；
        residues: list[str]，verify_clean 的发现，未执行校验时为空。

    重建后必须重新解析输出并核对段数、各段载荷长度和文件总长度；长度不一致
    时返回 ok 为 False、errors 含 output_length_mismatch，不能把截断文件当作
    成功。单张图的 ValueError 或 struct.error 只能让该图失败，错误写入该图的
    errors，不能中止同批其他图。若残留校验发现内容，ok 为 False，residues
    保留发现，data 仍返回已清洗版本。
    """

    raise NotImplementedError
