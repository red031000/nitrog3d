from .utils import error, read32, np_fixed_to_float, fixed_to_float, to_rgb, vec10_to_vec
from enum import IntEnum
import numpy as np

def parse_dl(data, size, report_func):
    offset = 0
    display_list = []
    while offset < size:
        commandData = read32(data, offset)
        offset += 4
        commands, offset = parse_dl_command(data, offset, commandData, report_func)
        display_list.append(commands)
    return display_list

def parse_dl_command(data, offset, commandData, report_func):
    commands = []
    for i in range(4):
        command = (commandData >> i * 8) & 0xFF

        if command == 0x00:
            commands.append(DLCommandNoop())
        elif command == 0x10:
            mode = MatrixMode(read32(data, offset))
            offset += 4
            commands.append(DLCommandMtxMode(mode))
        elif command == 0x11:
            commands.append(DLCommandPushMtx())
        elif command == 0x12:
            matrixId = read32(data, offset)
            offset += 4
            commands.append(DLCommandPopMtx(matrixId))
        elif command == 0x13:
            matrixId = read32(data, offset)
            offset += 4
            commands.append(DLCommandStoreMtx(matrixId))
        elif command == 0x14:
            matrixId = read32(data, offset)
            offset += 4
            commands.append(DLCommandRestoreMtx(matrixId))
        elif command == 0x15:
            commands.append(DLCommandIdentity())
        elif command == 0x16:
            matrix = np_fixed_to_float(np.frombuffer(data[offset:offset + 64], dtype=np.dtype('uint32').newbyteorder('<')).reshape((4, 4)))
            offset += 64
            commands.append(DLCommandLoadMtx44(matrix))
        elif command == 0x17:
            matrix = np_fixed_to_float(np.frombuffer(data[offset:offset + 48], dtype=np.dtype('uint32').newbyteorder('<')).reshape((4, 3)))
            offset += 48
            commands.append(DLCommandLoadMtx43(matrix))
        elif command == 0x18:
            matrix = np_fixed_to_float(np.frombuffer(data[offset:offset + 64], dtype=np.dtype('uint32').newbyteorder('<')).reshape((4, 4)))
            offset += 64
            commands.append(DLCommandMultMtx44(matrix))
        elif command == 0x19:
            matrix = np_fixed_to_float(np.frombuffer(data[offset:offset + 48], dtype=np.dtype('uint32').newbyteorder('<')).reshape((4, 3)))
            offset += 48
            commands.append(DLCommandMultMtx43(matrix))
        elif command == 0x1A:
            matrix = np_fixed_to_float(np.frombuffer(data[offset:offset + 36], dtype=np.dtype('uint32').newbyteorder('<')).reshape((3, 3)))
            offset += 36
            commands.append(DLCommandMultMtx33(matrix))
        elif command == 0x1B:
            x = fixed_to_float(read32(data, offset))
            y = fixed_to_float(read32(data, offset + 4))
            z = fixed_to_float(read32(data, offset + 8))
            matrix = np.identity(4)
            matrix[0, 0] = x
            matrix[1, 1] = y
            matrix[2, 2] = z
            offset += 12
            commands.append(DLCommandScale(matrix))
        elif command == 0x1C:
            x = fixed_to_float(read32(data, offset))
            y = fixed_to_float(read32(data, offset + 4))
            z = fixed_to_float(read32(data, offset + 8))
            matrix = np.identity(4)
            matrix[0, 3] = x
            matrix[1, 3] = y
            matrix[2, 3] = z
            offset += 12
            commands.append(DLCommandTranslate(matrix))
        elif command == 0x20:
            color = read32(data, offset)
            rgb = to_rgb(color)
            offset += 4
            commands.append(DLCommandColor(rgb))
        elif command == 0x21:
            normal = vec10_to_vec(read32(data, offset))
            offset += 4
            commands.append(DLCommandNormal(normal))
        elif command == 0x22:
            texcoord = read32(data, offset)
            s = fixed_to_float((texcoord & 0xFFFF) << 8) # unsure if this shift is correct
            t = fixed_to_float(((texcoord >> 16) & 0xFFFF) << 8)
            offset += 4
            commands.append(DLCommandTexcoord(s, t))
        #todo more commands
    return commands, offset

class MatrixMode(IntEnum):
    PROJECTION = 0
    POSITION = 1
    POSITION_VECTOR = 2
    TEXTURE = 3

class DLCommand:
    def __init__(self, commandId):
        self.commandId = commandId

class DLCommandNoop(DLCommand):
    def __init__(self):
        super().__init__(0x00)

class DLCommandMtxMode(DLCommand):
    def __init__(self, mode):
        super().__init__(0x10)
        self.mode = mode

class DLCommandPushMtx(DLCommand):
    def __init__(self):
        super().__init__(0x11)

class DLCommandPopMtx(DLCommand):
    def __init__(self, matrixId):
        super().__init__(0x12)
        self.matrixId = matrixId

class DLCommandStoreMtx(DLCommand):
    def __init__(self, matrixId):
        super().__init__(0x13)
        self.matrixId = matrixId

class DLCommandRestoreMtx(DLCommand):
    def __init__(self, matrixId):
        super().__init__(0x14)
        self.matrixId = matrixId

class DLCommandIdentity(DLCommand):
    def __init__(self):
        super().__init__(0x15)

class DLCommandLoadMtx44(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x16)
        self.matrix = matrix

class DLCommandLoadMtx43(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x17)
        self.matrix = matrix

class DLCommandMultMtx44(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x18)
        self.matrix = matrix

class DLCommandMultMtx43(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x19)
        self.matrix = matrix

class DLCommandMultMtx33(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x1A)
        self.matrix = matrix

class DLCommandScale(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x1B)
        self.matrix = matrix

class DLCommandTranslate(DLCommand):
    def __init__(self, matrix):
        super().__init__(0x1C)
        self.matrix = matrix

class DLCommandColor(DLCommand):
    def __init__(self, color):
        super().__init__(0x20)
        self.color = color

class DLCommandNormal(DLCommand):
    def __init__(self, normal):
        super().__init__(0x21)
        self.normal = normal

class DLCommandTexcoord(DLCommand):
    def __init__(self, s, t):
        super().__init__(0x22)
        self.s = s
        self.t = t