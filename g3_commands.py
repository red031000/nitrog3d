from .utils import error, read8, read32, np_fixed_to_float, fixed_to_float, to_rgb, vec10_to_vec, PolygonMode, CullMode, TexturePalette0Mode, TextureFlip, TextureRepeat, TextureTSize, TextureSSize, TextureConversionMode, TextureFormat
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
        elif command == 0x23:
            xy = read32(data, offset)
            z = fixed_to_float(read32(data, offset + 4))
            x = fixed_to_float(xy & 0xFFFF)
            y = fixed_to_float((xy >> 16) & 0xFFFF)
            vertex = np.array([x, y, z])
            offset += 8
            commands.append(DLCommandVtx(vertex))
        elif command == 0x24:
            vertex = vec10_to_vec(read32(data, offset))
            offset += 4
            commands.append(DLCommandVtx(vertex))
        elif command == 0x25:
            xy = read32(data, offset)
            x = fixed_to_float(xy & 0xFFFF)
            y = fixed_to_float((xy >> 16) & 0xFFFF)
            vertex = np.array([x, y])
            offset += 4
            commands.append(DLCommandVtxXY(vertex))
        elif command == 0x26:
            xz = read32(data, offset)
            x = fixed_to_float(xz & 0xFFFF)
            z = fixed_to_float((xz >> 16) & 0xFFFF)
            vertex = np.array([x, z])
            offset += 4
            commands.append(DLCommandVtxXZ(vertex))
        elif command == 0x27:
            yz = read32(data, offset)
            y = fixed_to_float(yz & 0xFFFF)
            z = fixed_to_float((yz >> 16) & 0xFFFF)
            vertex = np.array([y, z])
            offset += 4
            commands.append(DLCommandVtxYZ(vertex))
        elif command == 0x28:
            vertex = vec10_to_vec(read32(data, offset))
            offset += 4
            commands.append(DLCommandVtxDiff(vertex))
        elif command == 0x29:
            attributes = read32(data, offset)
            offset += 4
            polyAttr = DLCommandPolygonAttr()
            polyAttr.parse(attributes)
            commands.append(polyAttr)
        elif command == 0x2A:
            attributes = read32(data, offset)
            offset += 4
            texImageParam = DLCommandTexImageParam()
            texImageParam.parse(attributes)
            commands.append(texImageParam)
        elif command == 0x2B:
            address = read32(data, offset)
            offset += 4
            commands.append(DLCommandTexPlttBase(address))
        elif command == 0x30:
            attributes = read32(data, offset)
            offset += 4
            materialColourDiffAmb = DLCommandMaterialColourDiffAmb()
            materialColourDiffAmb.parse(attributes)
            commands.append(materialColourDiffAmb)
        elif command == 0x31:
            attributes = read32(data, offset)
            offset += 4
            materialColourSpecEmi = DLCommandMaterialColourSpecEmi()
            materialColourSpecEmi.parse(attributes)
            commands.append(materialColourSpecEmi)
        elif command == 0x32:
            attributes = read32(data, offset)
            offset += 4
            lightId = (attributes >> 30) & 0x3
            x = fixed_to_float(attributes & 0x3FFF)
            y = fixed_to_float((attributes >> 0xA) & 0x3FFF)
            z = fixed_to_float((attributes >> 0x14) & 0x3FFF)
            vertex = np.array([x, y, z])
            commands.append(DLCommandLightVector(lightId, vertex))
        elif command == 0x33:
            attributes = read32(data, offset)
            offset += 4
            lightId = (attributes >> 30) & 0x3
            rgb = to_rgb(attributes & 0x7FFF)
            commands.append(DLCommandLightColour(lightId, rgb))
        elif command == 0x34:
            shininessTable = []
            for j in range(32):
                shininess = read32(data, offset)
                offset += 4
                shininessTable.append(shininess & 0xFF)
                shininessTable.append((shininess >> 8) & 0xFF)
                shininessTable.append((shininess >> 16) & 0xFF)
                shininessTable.append((shininess >> 24) & 0xFF)
            commands.append(DLCommandShininess(shininessTable))
            
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

class DLCommandVtx(DLCommand):
    def __init__(self, vertex):
        super().__init__(0x23)
        self.vertex = vertex

class DLCommandVtx10(DLCommand):
    def __init__(self, vertex):
        super().__init__(0x24)
        self.vertex = vertex

class DLCommandVtxXY(DLCommand):
    def __init__(self, vertex):
        super().__init__(0x25)
        self.vertex = vertex

class DLCommandVtxXZ(DLCommand):
    def __init__(self, vertex):
        super().__init__(0x26)
        self.vertex = vertex

class DLCommandVtxYZ(DLCommand):
    def __init__(self, vertex):
        super().__init__(0x27)
        self.vertex = vertex

class DLCommandVtxDiff(DLCommand):
    def __init__(self, vertex):
        super().__init__(0x28)
        self.vertex = vertex

class DLCommandPolygonAttr(DLCommand):
    def __init__(self):
        super().__init__(0x29)
        self.lights = [False, False, False, False]
        self.polyMode = PolygonMode.MODULATE
        self.cullMode = CullMode.NONE
        self.polygonId = 0
        self.alpha = 0
        self.xluDepthUpdate = False
        self.farClipping = False
        self.display1Dot = False
        self.depthTest = False
        self.fog = False
        
    def parse(self, attributes):
        light = attributes & 0xF
        self.lights[0] = (light & 0x1) != 0
        self.lights[1] = (light & 0x2) != 0
        self.lights[2] = (light & 0x4) != 0
        self.lights[3] = (light & 0x8) != 0

        polyMode = (attributes >> 4) & 0x3
        self.polyMode = PolygonMode(polyMode)

        cullMode = (attributes >> 6) & 0x3
        self.cullMode = CullMode(cullMode)

        polygonId = (attributes >> 24) & 0x3F
        self.polygonId = polygonId

        alpha = (attributes >> 16) & 0x1F
        self.alpha = alpha

        self.xluDepthUpdate = (attributes >> 11) & 0x1 != 0

        self.farClipping = (attributes >> 12) & 0x1 != 0

        self.display1Dot = (attributes >> 13) & 0x1 != 0

        self.depthTest = (attributes >> 14) & 0x1 != 0

        self.fog = (attributes >> 15) & 0x1 != 0

class DLCommandTexImageParam(DLCommand):
    def __init__(self):
        super().__init__(0x2A)
        self.texturePalette0Mode = TexturePalette0Mode.USE
        self.textureFlip = TextureFlip.NONE
        self.textureRepeat = TextureRepeat.NONE
        self.textureTSize = TextureTSize.T8
        self.textureSSize = TextureSSize.S8
        self.textureConversionMode = TextureConversionMode.NONE
        self.textureFormat = TextureFormat.NONE
        self.textureAddress = 0
    
    def parse(self, attributes):
        texturePalette0Mode = (attributes >> 29) & 0x1
        self.texturePalette0Mode = TexturePalette0Mode(texturePalette0Mode)

        textureFlip = (attributes >> 18) & 0x3
        self.textureFlip = TextureFlip(textureFlip)

        textureRepeat = (attributes >> 16) & 0x3
        self.textureRepeat = TextureRepeat(textureRepeat)

        textureTSize = (attributes >> 23) & 0x7
        self.textureTSize = TextureTSize(textureTSize)

        textureSSize = (attributes >> 20) & 0x7
        self.textureSSize = TextureSSize(textureSSize)

        textureConversionMode = (attributes >> 30) & 0x3
        self.textureConversionMode = TextureConversionMode(textureConversionMode)

        textureFormat = (attributes >> 26) & 0x7
        self.textureFormat = TextureFormat(textureFormat)

        self.textureAddress = attributes & 0xFFFF

class DLCommandTexPlttBase(DLCommand):
    def __init__(self, address):
        super().__init__(0x2B)
        self.paletteAddress = address

class DLCommandMaterialColourDiffAmb(DLCommand):
    def __init__(self):
        super().__init__(0x30)
        self.diffuse = (0, 0, 0)
        self.ambient = (0, 0, 0)
        self.isVertexColour = False

    def parse(self, attributes):
        self.diffuse = to_rgb(attributes & 0x7FFF)
        self.isVertexColour = (attributes >> 15) & 0x1 != 0
        self.ambient = to_rgb((attributes >> 16) & 0x7FFF)

class DLCommandMaterialColourSpecEmi(DLCommand):
    def __init__(self):
        super().__init__(0x31)
        self.specular = (0, 0, 0)
        self.emission = (0, 0, 0)
        self.isShininess = False

    def parse(self, attributes):
        self.specular = to_rgb(attributes & 0x7FFF)
        self.isShininess = (attributes >> 15) & 0x1 != 0
        self.emission = to_rgb((attributes >> 16) & 0x7FFF)

class DLCommandLightVector(DLCommand):
    def __init__(self, lightId, vertex):
        super().__init__(0x32)
        self.lightId = lightId
        self.vertex = vertex

class DLCommandLightColour(DLCommand):
    def __init__(self, lightId, colour):
        super().__init__(0x33)
        self.lightId = lightId
        self.colour = colour

class DLCommandShininess(DLCommand):
    def __init__(self, shininessTable):
        super().__init__(0x34)
        self.shininessTable = shininessTable
