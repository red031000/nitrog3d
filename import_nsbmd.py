from enum import IntEnum, IntFlag
from os.path import isfile
from .utils import read8, read16, read32, read_str, log, debug, parse_dictionary, fixed_to_float, to_rgb, PolygonMode, CullMode
from .g3_commands import parse_dl
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

class NSBMDPaletteMaterialData():
    def __init__(self, name, material_id, bound):
        self.name = name
        self.materialId = material_id
        self.bound = bound

class NSBMDTextureMaterialData():
    def __init__(self, name, material_id, bound):
        self.name = name
        self.materialId = material_id
        self.bound = bound

class NSBMDMaterialPolygonAttributes():
    def __init__(self):
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
        pass

    def parse_attributes(self, attributes, report_func):
        light = attributes & 0xF
        self.lights[0] = (light & 0x1) != 0
        self.lights[1] = (light & 0x2) != 0
        self.lights[2] = (light & 0x4) != 0
        self.lights[3] = (light & 0x8) != 0
        log('Lights: %s' % str(self.lights), report_func)

        polyMode = (attributes >> 4) & 0x3
        self.polyMode = PolygonMode(polyMode)
        log('Polygon mode: %s' % self.polyMode.name, report_func)

        cullMode = (attributes >> 6) & 0x3
        self.cullMode = CullMode(cullMode)
        log('Cull mode: %s' % self.cullMode.name, report_func)

        polygonId = (attributes >> 24) & 0x3F
        self.polygonId = polygonId
        log('Polygon ID: %d' % self.polygonId, report_func)

        alpha = (attributes >> 16) & 0x1F
        self.alpha = alpha
        log('Alpha: %d' % self.alpha, report_func)

        self.xluDepthUpdate = (attributes >> 11) & 0x1 != 0
        log('XLU depth update: %s' % self.xluDepthUpdate, report_func)

        self.farClipping = (attributes >> 12) & 0x1 != 0
        log('Far clipping: %s' % self.farClipping, report_func)

        self.display1Dot = (attributes >> 13) & 0x1 != 0
        log('Display 1 dot polygons: %s' % self.display1Dot, report_func)

        self.depthTest = (attributes >> 14) & 0x1 != 0
        log('Depth test: %s' % self.depthTest, report_func)

        self.fog = (attributes >> 15) & 0x1 != 0
        log('Fog: %s' % self.fog, report_func)

class TextureFormat(IntEnum):
    NONE = 0
    A3I5 = 1
    PLTT4 = 2
    PLTT16 = 3
    PLTT256 = 4
    COMP4X4 = 5
    A5I3 = 6
    DIRECT = 7

class TextureConversionMode(IntEnum):
    NONE = 0
    TEXCOORD = 1
    NORMAL = 2
    VERTEX = 3

class TextureSSize(IntEnum):
    S8 = 0
    S16 = 1
    S32 = 2
    S64 = 3
    S128 = 4
    S256 = 5
    S512 = 6
    S1024 = 7

class TextureTSize(IntEnum):
    T8 = 0
    T16 = 1
    T32 = 2
    T64 = 3
    T128 = 4
    T256 = 5
    T512 = 6
    T1024 = 7

class TextureRepeat(IntEnum):
    NONE = 0
    S = 1
    T = 2
    ST = 3

class TextureFlip(IntEnum):
    NONE = 0
    S = 1
    T = 2
    ST = 3

class TexturePalette0Mode(IntEnum):
    USE = 0
    TRANSPARENT = 1

class NSBMDMaterialTextureImageParameters():
    def __init__(self):
        self.address = 0
        self.textureFormat = TextureFormat.NONE
        self.textureConversionMode = TextureConversionMode.NONE
        self.textureSSize = TextureSSize.S8
        self.textureTSize = TextureTSize.T8
        self.textureRepeat = TextureRepeat.NONE
        self.textureFlip = TextureFlip.NONE
        self.texturePalette0Mode = TexturePalette0Mode.USE

    def parse_parameters(self, parameters, report_func):
        self.address = (parameters & 0xFFFF) << 3
        log('Address: %d' % self.address, report_func)

        textureFormat = (parameters >> 26) & 0x7
        self.textureFormat = TextureFormat(textureFormat)
        log('Texture format: %s' % self.textureFormat.name, report_func)

        textureConversionMode = (parameters >> 30) & 0x3
        self.textureConversionMode = TextureConversionMode(textureConversionMode)
        log('Texture conversion mode: %s' % self.textureConversionMode.name, report_func)

        textureSSize = (parameters >> 20) & 0x7
        self.textureSSize = TextureSSize(textureSSize)
        log('Texture S size: %s' % self.textureSSize.name, report_func)

        textureTSize = (parameters >> 23) & 0x7
        self.textureTSize = TextureTSize(textureTSize)
        log('Texture T size: %s' % self.textureTSize.name, report_func)

        textureRepeat = (parameters >> 16) & 0x3
        self.textureRepeat = TextureRepeat(textureRepeat)
        log('Texture repeat: %s' % self.textureRepeat.name, report_func)

        textureFlip = (parameters >> 18) & 0x3
        self.textureFlip = TextureFlip(textureFlip)
        log('Texture flip: %s' % self.textureFlip.name, report_func)

        texturePalette0Mode = (parameters >> 29) & 0x1
        self.texturePalette0Mode = TexturePalette0Mode(texturePalette0Mode)
        log('Texture palette 0 mode: %s' % self.texturePalette0Mode.name, report_func)

class MaterialFlags(IntFlag):
    TEXTURE_MATRIX_USE = 0x0001
    SCALE_ONE = 0x0002
    ROTATION_ZERO = 0x0004
    TRANSLATION_ZERO = 0x0008
    WIDTH_HEIGHT_SAME = 0x0010
    WIREFRAME = 0x0020
    DIFFUSE = 0x0040
    AMBIENT = 0x0080
    VERTEX_COLOR = 0x0100
    SPECULAR = 0x0200
    EMISSION = 0x0400
    SHININESS = 0x0800
    TEXTURE_BASE_PALETTE = 0x1000
    EFFECT_MATRIX_USE = 0x2000

class NSBMDMaterialFlags():
    def __init__(self):
        self.textureMatrixUse = False
        self.scaleOne = False
        self.rotationZero = False
        self.translationZero = False
        self.widthHeightSame = False # not sure if this is needed because this is mostly set during execution
        self.wireframe = False
        self.diffuse = False
        self.ambient = False
        self.vertexColor = False
        self.specular = False
        self.emission = False
        self.shininess = False
        self.textureBasePalette = False
        self.effectMatrixUse = False
        pass

    def parse_flags(self, flags, report_func):
        self.textureMatrixUse = (flags & MaterialFlags.TEXTURE_MATRIX_USE) != 0
        log('Texture matrix use: %s' % self.textureMatrixUse, report_func)

        self.scaleOne = (flags & MaterialFlags.SCALE_ONE) != 0
        log('Scale one: %s' % self.scaleOne, report_func)

        self.rotationZero = (flags & MaterialFlags.ROTATION_ZERO) != 0
        log('Rotation zero: %s' % self.rotationZero, report_func)

        self.translationZero = (flags & MaterialFlags.TRANSLATION_ZERO) != 0
        log('Translation zero: %s' % self.translationZero, report_func)

        self.widthHeightSame = (flags & MaterialFlags.WIDTH_HEIGHT_SAME) != 0
        log('Width height same: %s' % self.widthHeightSame, report_func)

        self.wireframe = (flags & MaterialFlags.WIREFRAME) != 0
        log('Wireframe: %s' % self.wireframe, report_func)

        self.diffuse = (flags & MaterialFlags.DIFFUSE) != 0
        log('Diffuse: %s' % self.diffuse, report_func)

        self.ambient = (flags & MaterialFlags.AMBIENT) != 0
        log('Ambient: %s' % self.ambient, report_func)

        self.vertexColor = (flags & MaterialFlags.VERTEX_COLOR) != 0
        log('Vertex color: %s' % self.vertexColor, report_func)

        self.specular = (flags & MaterialFlags.SPECULAR) != 0
        log('Specular: %s' % self.specular, report_func)

        self.emission = (flags & MaterialFlags.EMISSION) != 0
        log('Emission: %s' % self.emission, report_func)

        self.shininess = (flags & MaterialFlags.SHININESS) != 0
        log('Shininess: %s' % self.shininess, report_func)

        self.textureBasePalette = (flags & MaterialFlags.TEXTURE_BASE_PALETTE) != 0
        log('Texture base palette: %s' % self.textureBasePalette, report_func)

        self.effectMatrixUse = (flags & MaterialFlags.EFFECT_MATRIX_USE) != 0
        log('Effect matrix use: %s' % self.effectMatrixUse, report_func)

class NSBMDMaterial():
    def __init__(self, name):
        self.name = name
        self.textureMatData = []
        self.paletteMatData = []

    def add_texture_mat_data(self, texture_mat_data):
        self.textureMatData.append(texture_mat_data)
    
    def add_palette_mat_data(self, palette_mat_data):
        self.paletteMatData.append(palette_mat_data)

class ShapeFlags(IntFlag):
    USE_NORMAL = 0x0001
    USE_COLOR = 0x0002
    USE_TEXCOORD = 0x0004
    USE_RESTOREMTX = 0x0008

class NSBMDShape():
    def __init__(self, name):
        self.name = name
        self.useNormal = False
        self.useColor = False
        self.useTexCoord = False
        self.useRestoreMtx = False
    
    def parse_flags(self, flags, report_func):
        self.useNormal = (flags & ShapeFlags.USE_NORMAL) != 0
        log('Use normal: %s' % self.useNormal, report_func)

        self.useColor = (flags & ShapeFlags.USE_COLOR) != 0
        log('Use color: %s' % self.useColor, report_func)

        self.useTexCoord = (flags & ShapeFlags.USE_TEXCOORD) != 0
        log('Use tex coord: %s' % self.useTexCoord, report_func)

        self.useRestoreMtx = (flags & ShapeFlags.USE_RESTOREMTX) != 0
        log('Use restore mtx: %s' % self.useRestoreMtx, report_func)

class NSBMDModel():
    def __init__(self, name):
        self.name = name
        self.nodes = []
        self.materials = []

    def add_node(self, node):
        self.nodes.append(node)
    
    def add_material(self, material):
        self.materials.append(material)

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
            
            dict_size = read16(materialset_data[offsetDictPlttToMat:], 0x2)
            model.matIdxData = materialset_data[offsetDictPlttToMat + dict_size:matIdxDataEnd].tobytes() # no idea how this is used, but essential
            log('Material id data: %s' % model.matIdxData.hex(" "), self.report)
            
            for material_key, material_value in materialset_dictionary.items():
                log('%s: %08X' % (material_key, material_value), self.report)
                material = NSBMDMaterial(material_key)
                material_data = materialset_data[material_value:]
                diffAmb = read32(material_data, 0x04)
                material.diffuse = to_rgb(diffAmb & 0x7FFF)
                log("Diffuse: R: %d G: %d B: %d" % material.diffuse, self.report)
                material.ambient = to_rgb((diffAmb >> 16) & 0x7FFF)
                log("Ambient: R: %d G: %d B: %d" % material.ambient, self.report)
                material.vertexColor = (diffAmb >> 15) & 0x01 != 0
                log("Vertex color: %s" % material.vertexColor, self.report)
                specEmi = read32(material_data, 0x08)
                material.specular = to_rgb(specEmi & 0x7FFF)
                log("Specular: R: %d G: %d B: %d" % material.specular, self.report)
                material.emission = to_rgb((specEmi >> 16) & 0x7FFF)
                log("Emission: R: %d G: %d B: %d" % material.emission, self.report)
                material.shininess = (specEmi >> 15) & 0x01 != 0
                log("Shininess: %s" % material.shininess, self.report)
                polygonAttrData = read32(material_data, 0x0C)
                polygonAttributes = NSBMDMaterialPolygonAttributes()
                polygonAttributes.parse_attributes(polygonAttrData, self.report)
                material.polygonAttributes = polygonAttributes
                textureImageParamData = read32(material_data, 0x14)
                textureImageParam = NSBMDMaterialTextureImageParameters()
                textureImageParam.parse_parameters(textureImageParamData, self.report)
                material.textureImageParameters = textureImageParam
                texturePaletteBase = read16(material_data, 0x1C)
                texturePaletteBase = texturePaletteBase << 3 if material.textureImageParameters.textureFormat == TextureFormat.PLTT4 else texturePaletteBase << 4
                material.texturePaletteBase = texturePaletteBase
                log('Texture palette base: %d' % texturePaletteBase, self.report)
                flagsData = read16(material_data, 0x1E)
                flags = NSBMDMaterialFlags()
                flags.parse_flags(flagsData, self.report)
                material.materialFlags = flags
                material.originWidth = read16(material_data, 0x20)
                log('Origin width: %d' % material.originWidth, self.report)
                material.originHeight = read16(material_data, 0x22)
                log('Origin height: %d' % material.originHeight, self.report)
                widthMagnitude = fixed_to_float(read32(material_data, 0x24))
                material.widthMagnitude = widthMagnitude
                log('Width magnitude: %.12f' % widthMagnitude, self.report)
                heightMagnitude = fixed_to_float(read32(material_data, 0x28))
                material.heightMagnitude = heightMagnitude
                log('Height magnitude: %.12f' % heightMagnitude, self.report)
                materialOffset = 0x2C
                if material.materialFlags.scaleOne:
                    material.scaleS = 1.0
                    material.scaleT = 1.0
                else:
                    material.scaleS = fixed_to_float(read32(material_data, materialOffset))
                    material.scaleT = fixed_to_float(read32(material_data, materialOffset + 4))
                    materialOffset += 8
                log('Scale S: %.12f' % material.scaleS, self.report)
                log('Scale T: %.12f' % material.scaleT, self.report)
                if material.materialFlags.rotationZero:
                    material.rotationSin = 0.0
                    material.rotationCos = 1.0
                else:
                    material.rotationSin = fixed_to_float(read32(material_data, materialOffset))
                    material.rotationCos = fixed_to_float(read32(material_data, materialOffset + 4))
                    materialOffset += 8
                log('Rotation sin: %.12f' % material.rotationSin, self.report)
                log('Rotation cos: %.12f' % material.rotationCos, self.report)
                if material.materialFlags.translationZero:
                    material.translationS = 0.0
                    material.translationT = 0.0
                else:
                    material.translationS = fixed_to_float(read32(material_data, materialOffset))
                    material.translationT = fixed_to_float(read32(material_data, materialOffset + 4))
                    materialOffset += 8
                log('Translation S: %.12f' % material.translationS, self.report)
                log('Translation T: %.12f' % material.translationT, self.report)
                if material.materialFlags.effectMatrixUse:
                    effectMatrix = []
                    for i in range(16):
                        effectMatrix.append(fixed_to_float(read32(material_data, materialOffset)))
                        materialOffset += 4
                    material.effectMatrix = np.array(effectMatrix).reshape((4, 4))
                else:
                    material.effectMatrix = None
                log('Effect matrix: %s' % material.effectMatrix, self.report)
                model.add_material(material)

            for text_mat_key, text_mat_value in text_to_mat_dictionary.items():
                text_mat_offset = text_mat_value & 0xFFFF
                text_mat_number = text_mat_value >> 16 & 0xFF
                text_mat_bound = text_mat_value >> 24 & 0xFF
                text_mat_data = materialset_data[text_mat_offset:]
                for i in range(text_mat_number):
                    material_id = read8(text_mat_data, i)
                    text_mat = NSBMDTextureMaterialData(text_mat_key, material_id, text_mat_bound)
                    model.materials[material_id].add_texture_mat_data(text_mat)
            
            for pltt_mat_key, pltt_mat_value in pltt_to_mat_dictionary.items():
                pltt_mat_offset = pltt_mat_value & 0xFFFF
                pltt_mat_number = pltt_mat_value >> 16 & 0xFF
                pltt_mat_bound = pltt_mat_value >> 24 & 0xFF
                pltt_mat_data = materialset_data[pltt_mat_offset:]
                for i in range(pltt_mat_number):
                    material_id = read8(pltt_mat_data, i)
                    pltt_mat = NSBMDPaletteMaterialData(pltt_mat_key, material_id, pltt_mat_bound)
                    model.materials[material_id].add_palette_mat_data(pltt_mat)

            shape_data = data[shape_offset:]
            shape_dictionary = parse_dictionary(shape_data)
            for shape_key, shape_value in shape_dictionary.items():
                log('%s: %08X' % (shape_key, shape_value), self.report)
                shape = NSBMDShape(shape_key)
                shape_item_data = shape_data[shape_value:]
                shape_flags = read32(shape_item_data, 0x04)
                shape.parse_flags(shape_flags, self.report)
                shape_dl_offset = read32(shape_item_data, 0x08)
                shape_dl_size = read32(shape_item_data, 0x0C)
                shape.dlData = parse_dl(shape_data[shape_dl_offset:], shape_dl_size, self.report)

        #todo
        return nsbmd
