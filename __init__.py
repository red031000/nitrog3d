import bpy
from bpy.props import StringProperty, BoolProperty, CollectionProperty
from bpy_extras.io_utils import ImportHelper
from .utils import log, debug
import os

bl_info = {
    "name": "Nitro G3D Importer/Exporter",
    "author": "red031000",
    "version": (0, 1, 0),
    "blender": (3, 6, 0),
    "location": "File > Import-Export",
    "description": "Import/Export 3D compiled files for Nitro",
    'support': 'COMMUNITY',
    "category": "Import-Export",
}

def reload_package(module_dict_main):
    import importlib
    from pathlib import Path

    def reload_package_recursive(current_dir, module_dict):
        for path in current_dir.iterdir():
            if "__init__" in str(path) or path.stem not in module_dict:
                continue

            if path.is_file() and path.suffix == ".py":
                importlib.reload(module_dict[path.stem])
            elif path.is_dir():
                reload_package_recursive(path, module_dict[path.stem].__dict__)

    reload_package_recursive(Path(__file__).parent, module_dict_main)


if "bpy" in locals():
    reload_package(locals())

class ImportNitro(bpy.types.Operator, ImportHelper):
    bl_idname = "import_scene.g3d"
    bl_label = "Import Nitro"
    bl_options = {'PRESET'}

    filter_glob: StringProperty(
        default="*.nsbmd",
        options={'HIDDEN'},
        )
    
    files: CollectionProperty(
        name="File Path",
        type=bpy.types.OperatorFileListElement,
    )
    
    #generate_log: BoolProperty(name="Generate Log", default=False)

    def execute(self, context):
        return self.process_import()
    
    def draw(self, context):
        pass
        #layout = self.layout

        #layout.use_property_split = True
        #layout.use_property_decorate = False

        #layout.prop(self, "generate_log")
    
    def process_import(self):
        import_settings = self.as_keywords()

        if self.files:
            ret = {'FINISHED'}
            dirname = os.path.dirname(self.filepath)
            for file in self.files:
                path = os.path.join(dirname, file.name)
                if self.try_import(path, import_settings) != {'FINISHED'}:
                    ret = {'CANCELLED'}
            return ret
        else:
            return self.try_import(self.filepath, import_settings)
    
    def try_import(self, filename, import_settings):
        try:
            if filename.lower().endswith('.nsbmd'):
                log("Valid file type", self.report)
                from .import_nsbmd import NSBMDImporter
                nsbmd_importer = NSBMDImporter(filename, import_settings, self.report)
                data = nsbmd_importer.read()
            else:
                raise Exception('Unsupported file type')
            #todo
            return {'FINISHED'}
        except Exception as e:
            self.report(type={'ERROR'}, message=str(e))
            return {'CANCELLED'}

def menu_func_import(self, context):
    self.layout.operator(ImportNitro.bl_idname, text="Nitro Compiled (.nsbmd)")

def register():
    bpy.utils.register_class(ImportNitro)
    bpy.types.TOPBAR_MT_file_import.append(menu_func_import)

def unregister():
    bpy.utils.unregister_class(ImportNitro)
    bpy.types.TOPBAR_MT_file_import.remove(menu_func_import)

if __name__ == "__main__":
    register()
