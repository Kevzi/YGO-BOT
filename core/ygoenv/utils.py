import struct

class BinaryReader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def read_byte(self) -> int:
        if self.offset >= len(self.data):
            raise EOFError("Fin de buffer atteinte")
        val = self.data[self.offset]
        self.offset += 1
        return val

    def read_uint8(self) -> int:
        return self.read_byte()
        
    def read_int8(self) -> int:
        val = self.read_byte()
        return val - 256 if val > 127 else val

    def read_uint16(self) -> int:
        if self.offset + 2 > len(self.data):
            raise EOFError("Fin de buffer atteinte")
        val = struct.unpack_from("<H", self.data, self.offset)[0]
        self.offset += 2
        return val

    def read_uint32(self) -> int:
        if self.offset + 4 > len(self.data):
            raise EOFError("Fin de buffer atteinte")
        val = struct.unpack_from("<I", self.data, self.offset)[0]
        self.offset += 4
        return val
        
    def read_int32(self) -> int:
        if self.offset + 4 > len(self.data):
            raise EOFError("Fin de buffer atteinte")
        val = struct.unpack_from("<i", self.data, self.offset)[0]
        self.offset += 4
        return val

    def read_uint64(self) -> int:
        if self.offset + 8 > len(self.data):
            raise EOFError("Fin de buffer atteinte")
        val = struct.unpack_from("<Q", self.data, self.offset)[0]
        self.offset += 8
        return val

    def read_bytes(self, length: int) -> bytes:
        if self.offset + length > len(self.data):
            raise EOFError("Fin de buffer atteinte")
        val = self.data[self.offset:self.offset+length]
        self.offset += length
        return val

    def eof(self) -> bool:
        return self.offset >= len(self.data)
