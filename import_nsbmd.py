from enum import IntEnum, IntFlag
from os.path import isfile
from .utils import read8, read16, read32, read_str, log, debug, parse_dictionary, fixed_to_float
import numpy as np

class ScalingRule(IntEnum):
    NORMAL = 0
    MAYA = 1
    SOFTIMAGE = 2

class TextureMatrixMode(IntEnum):
    MAYA = 0
    SOFTIMAGE_3D = 1
    _3DSMAX = 2
    SOFTIMAGE_XSI = 3

class NSBMDOptions():
    def __init__(self):
        self.scalingRule = ScalingRule.NORMAL
        self.textureMatrixMode = TextureMatrixMode.MAYA
        self.jointNumber = 0
        self.materialNumber = 0
        self.shapeNumber = 0
        self.firstUnusedMatrixStackId = 0
        self.positionScale = 0
        self.inversePositionScale = 0
        self.vertexNumber = 0
        self.polygonNumber = 0
        self.triangleNumber = 0
        self.quadNumber = 0
        self.boxX = 0
        self.boxY = 0
        self.boxZ = 0
        self.boxWidth = 0
        self.boxHeight = 0
        self.boxDepth = 0
        self.boxPositionScale = 0
        self.boxInversePositionScale = 0

class NodePivotData(IntEnum):
    MASK = 0xF0
    SHIFT = 4

class NodeFlags(IntFlag):
    TRANSLATION_ZERO = 0x0001
    ROTATION_ZERO = 0x0002
    SCALE_ONE = 0x0004
    ROTATION_COMPRESSED = 0x0008
    PIVOT_MINUS = 0x0100
    PIVOT_REVERSED_C = 0x0200
    PIVOT_REVERSED_D = 0x0400


class NSBMDNode():
    pivot_table = [
        [
            [(1, 1), (1, 2), (2, 1), (2, 2)],
            [(1, 0), (1, 2), (2, 0), (2, 2)],
            [(1, 0), (1, 1), (2, 0), (2, 1)]
        ],
        [
            [(0, 1), (0, 2), (2, 1), (2, 2)],
            [(0, 0), (0, 2), (2, 0), (2, 2)],
            [(0, 0), (0, 1), (2, 0), (2, 1)]
        ],
        [
            [(0, 1), (0, 2), (1, 1), (1, 2)],
            [(0, 0), (0, 2), (1, 0), (1, 2)],
            [(0, 0), (0, 1), (1, 0), (1, 1)]
        ]
    ]

    def __init__(self, name):
        self.name = name
        self.translation = np.array([0, 0, 0])
        self.rotation = np.identity(3, dtype=np.float32)
        self.scale = np.array([1, 1, 1], dtype=np.float32)
        self.inverseScale = np.array([1, 1, 1], dtype=np.float32)
    
    def parse_data(self, flags, report_func, data):
        offset = 4
        if (flags & NodeFlags.TRANSLATION_ZERO) != 0:
            log('Translation zero', report_func)
            self.translation = np.array([0, 0, 0], dtype=np.float32)
        else:
            self.translation = np.array([fixed_to_float(read32(data, offset)), fixed_to_float(read32(data, offset + 4)), fixed_to_float(read32(data, offset + 8))], dtype=np.float32)
            log('Translation: ' + str(self.translation), report_func)
            offset += 12
        if (flags & NodeFlags.ROTATION_ZERO) != 0:
            log('Rotation zero', report_func)
            self.rotation = np.identity(3, dtype=np.float32)
        elif (flags & NodeFlags.ROTATION_COMPRESSED) != 0:
            log('Rotation compressed', report_func)
            A = fixed_to_float(read16(data, offset))
            B = fixed_to_float(read16(data, offset + 2))
            pivot = (flags & NodePivotData.MASK) >> NodePivotData.SHIFT
            self.rotation = np.identity(3, dtype=np.float32)
            row = pivot / 3
            column = pivot % 3
            self.rotation[row, column] = -1.0 if (flags & NodeFlags.PIVOT_MINUS) else 1.0
            AIndex = NSBMDNode.pivot_table[row][column][0]
            BIndex = NSBMDNode.pivot_table[row][column][1]
            CIndex = NSBMDNode.pivot_table[row][column][2]
            DIndex = NSBMDNode.pivot_table[row][column][3]
            self.rotation[AIndex[0], AIndex[1]] = A
            self.rotation[BIndex[0], BIndex[1]] = B
            self.rotation[CIndex[0], CIndex[1]] = -B if (flags & NodeFlags.PIVOT_REVERSED_C) else B
            self.rotation[DIndex[0], DIndex[1]] = -A if (flags & NodeFlags.PIVOT_REVERSED_D) else A
            log('Rotation: ' + str(self.rotation), report_func)
            offset += 4
        else:
            self.rotation = np.identity(3, dtype=np.float32)
            self.rotation[0, 0] = fixed_to_float(read16(data, 2))
            self.rotation[0, 1] = fixed_to_float(read16(data, offset))
            self.rotation[0, 2] = fixed_to_float(read16(data, offset + 2))
            self.rotation[1, 0] = fixed_to_float(read16(data, offset + 4))
            self.rotation[1, 1] = fixed_to_float(read16(data, offset + 6))
            self.rotation[1, 2] = fixed_to_float(read16(data, offset + 8))
            self.rotation[2, 0] = fixed_to_float(read16(data, offset + 10))
            self.rotation[2, 1] = fixed_to_float(read16(data, offset + 12))
            self.rotation[2, 2] = fixed_to_float(read16(data, offset + 14))
            log('Rotation: ' + str(self.rotation), report_func)
            offset += 16
        if (flags & NodeFlags.SCALE_ONE) != 0:
            log('Scale one', report_func)
            self.scale = np.array([1, 1, 1], dtype=np.float32)
            self.inverseScale = np.array([1, 1, 1], dtype=np.float32)
        else:
            self.scale = np.array([fixed_to_float(read32(data, offset)), fixed_to_float(read32(data, offset + 4)), fixed_to_float(read32(data, offset + 8))], dtype=np.float32)
            self.inverseScale = np.array([fixed_to_float(read32(data, offset + 12)), fixed_to_float(read32(data, offset + 16)), fixed_to_float(read32(data, offset + 20))], dtype=np.float32)
            log('Scale: ' + str(self.scale), report_func)
            offset += 24
        return offset

class NSBMDModel():
    def __init__(self, name):
        self.name = name
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)

class NSBMD():
    def __init__(self, has_textures, model_offset, texture_offset):
        self.has_textures = has_textures
        self.model_offset = model_offset
        self.texture_offset = texture_offset
        self.models = []
    
    def add_model(self, model):
        self.models.append(model)


class NSBMDImporter():
    def __init__(self, filename, import_settings, report_func):
        self.filename = filename
        self.import_settings = import_settings
        self.report = report_func

    def read(self):
        if not isfile(self.filename):
            raise Exception('File not found')
        
        data = []
        with open(self.filename, 'rb') as f:
            data = memoryview(f.read())
        
        if data[0:4] != b'BMD0':
            raise Exception('Invalid file format')
        
        return self.parse(data)
    
    def parse(self, data):
        has_textures = read16(data, 0x0E) == 2
        model_offset = read32(data, 0x10)
        texture_offset = 0
        if has_textures:
            texture_offset = read32(data, 0x14)
        
        nsbmd = NSBMD(has_textures, model_offset, texture_offset)

        log('Model offset: %08X' % model_offset, self.report)
        if has_textures:
            log('Texture offset: %08X' % texture_offset, self.report)
        
        modelset_data = data[model_offset:]

        if modelset_data[0:4] != b'MDL0':
            raise Exception('Invalid file format')
        
        dictionary = parse_dictionary(modelset_data[8:])

        for key, value in dictionary.items():
            model = NSBMDModel(key)
            log('%s: %08X' % (key, value), self.report)
            model_data = modelset_data[value:]
            sbc_offset = read32(model_data, 0x04)
            log('SBC offset: %08X' % sbc_offset, self.report)
            materialset_offset = read32(model_data, 0x08)
            log('Materialset offset: %08X' % materialset_offset, self.report)
            shape_offset = read32(model_data, 0x0C)
            log('Shape offset: %08X' % shape_offset, self.report)
            envelope_matrix_offset = read32(model_data, 0x10)
            log('Envelope matrix offset: %08X' % envelope_matrix_offset, self.report)

            model.options = NSBMDOptions()
            model.options.scalingRule = ScalingRule(read8(model_data, 0x15))
            log('Scaling rule: %s' % model.options.scalingRule.name, self.report)
            model.options.textureMatrixMode = TextureMatrixMode(read8(model_data, 0x16))
            log('Texture matrix mode: %s' % model.options.textureMatrixMode.name, self.report)
            model.options.jointNumber = read8(model_data, 0x17)
            log('Joint number: %d' % model.options.jointNumber, self.report)
            model.options.materialNumber = read8(model_data, 0x18)
            log('Material number: %d' % model.options.materialNumber, self.report)
            model.options.shapeNumber = read8(model_data, 0x19)
            log('Shape number: %d' % model.options.shapeNumber, self.report)
            model.options.firstUnusedMatrixStackId = read8(model_data, 0x1A)
            log('First unused matrix stack ID: %d' % model.options.firstUnusedMatrixStackId, self.report)
            model.options.positionScale = fixed_to_float(read32(model_data, 0x1C))
            log('Position scale: %.12f' % model.options.positionScale, self.report)
            model.options.inversePositionScale = fixed_to_float(read32(model_data, 0x20))
            log('Inverse position scale: %.12f' % model.options.inversePositionScale, self.report)
            model.options.vertexNumber = read16(model_data, 0x24)
            log('Vertex number: %d' % model.options.vertexNumber, self.report)
            model.options.polygonNumber = read16(model_data, 0x26)
            log('Polygon number: %d' % model.options.polygonNumber, self.report)
            model.options.triangleNumber = read16(model_data, 0x28)
            log('Triangle number: %d' % model.options.triangleNumber, self.report)
            model.options.quadNumber = read16(model_data, 0x2A)
            log('Quad number: %d' % model.options.quadNumber, self.report)
            model.options.boxX = fixed_to_float(read16(model_data, 0x2C))
            log('Box X: %.12f' % model.options.boxX, self.report)
            model.options.boxY = fixed_to_float(read16(model_data, 0x2E))
            log('Box Y: %.12f' % model.options.boxY, self.report)
            model.options.boxZ = fixed_to_float(read16(model_data, 0x30))
            log('Box Z: %.12f' % model.options.boxZ, self.report)
            model.options.boxWidth = fixed_to_float(read16(model_data, 0x32))
            log('Box width: %.12f' % model.options.boxWidth, self.report)
            model.options.boxHeight = fixed_to_float(read16(model_data, 0x34))
            log('Box height: %.12f' % model.options.boxHeight, self.report)
            model.options.boxDepth = fixed_to_float(read16(model_data, 0x36))
            log('Box depth: %.12f' % model.options.boxDepth, self.report)
            model.options.boxPositionScale = fixed_to_float(read32(model_data, 0x38))
            log('Box position scale: %.12f' % model.options.boxPositionScale, self.report)
            model.options.inverseBoxPositionScale = fixed_to_float(read32(model_data, 0x3C))
            log('Inverse box position scale: %.12f' % model.options.inverseBoxPositionScale, self.report)

            nodeset_data = model_data[0x40:]
            node_dictionary = parse_dictionary(nodeset_data)
            offset = 0
            for node_key, node_value in node_dictionary.items():
                log('%s: %08X' % (node_key, node_value), self.report)
                node = NSBMDNode(node_key)
                node_data = nodeset_data[node_value:]
                node_flags = read16(node_data, 0x00)
                node_offset = node.parse_data(node_flags, self.report, node_data)
                offset = node_value + node_offset
                model.add_node(node)
            
            log('Offset: %08X' % (offset + 0x40), self.report)
            model.sbc = model_data[sbc_offset:materialset_offset].tobytes()
            log('SBC: %s' % model.sbc.hex(" "), self.report)

            materialset_data = model_data[materialset_offset:]
            offsetDictTextToMat = read16(materialset_data, 0x00)
            offsetDictPlttToMat = read16(materialset_data, 0x02)
            materialset_dictionary = parse_dictionary(materialset_data[4:])
            text_to_mat_dictionary = parse_dictionary(materialset_data[offsetDictTextToMat:])
            pltt_to_mat_dictionary = parse_dictionary(materialset_data[offsetDictPlttToMat:])

            # no offsets so this code is gonna be fucky
            matIdxDataEnd = 0xFFFFFFFF

            for material_key, material_value in materialset_dictionary.items():
                log('%s: %08X' % (material_key, material_value), self.report)
                if material_value < matIdxDataEnd:
                    matIdxDataEnd = material_value
            
            dict_size = read16(materialset_data[offsetDictPlttToMat:], 2)
            model.matIdxData = materialset_data[offsetDictPlttToMat + dict_size:matIdxDataEnd].tobytes() # no idea how this is used, but essential
            log('Material id data: %s' % model.matIdxData.hex(" "), self.report)
            
        #todo
        return nsbmd
