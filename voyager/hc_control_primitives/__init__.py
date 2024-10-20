import pkg_resources
import os
import voyager.utils as U


def load_control_primitives(primitive_names=None):
    package_path = pkg_resources.resource_filename("voyager", "")
    if primitive_names is None:
        primitive_names = [
            primitives[:-3]
            for primitives in os.listdir(f"{package_path}/hc_control_primitives")
            if primitives.endswith(".py")
        ]
    primitives = [
        U.load_text(f"{package_path}/hc_control_primitives/{primitive_name}.py")
        for primitive_name in primitive_names
    ]
    return primitives
