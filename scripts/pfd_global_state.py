import os.path
import stat
from collections import OrderedDict

from modules import shared, scripts, sd_models
from modules.paths import models_path
from scripts.pfd_logging import pfd_logger

models_exts = [".pt", ".pth", ".ckpt", ".safetensors"]
models_dir = os.path.join(models_path, "SeeCoder")
models_dir_old = os.path.join(scripts.basedir(), "models")
models_paths = OrderedDict() # "My_Lora(abcd1234)" -> C:/path/to/model.safetensors
models_names = {}  # "my_lora" -> "My_Lora(abcd1234)"
default_conf = os.path.join("models", "seecoder-v1-0.yaml")
script_dir = scripts.basedir()

os.makedirs(models_dir, exist_ok=True)

def traverse_all_files(curr_path, model_list):
    f_list = [(os.path.join(curr_path, entry.name), entry.stat())
              for entry in os.scandir(curr_path)]
    for f_info in f_list:
        fname, fstat = f_info
        if os.path.splitext(fname)[1] in models_exts:
            model_list.append(f_info)
        elif stat.S_ISDIR(fstat.st_mode):
            model_list = traverse_all_files(fname, model_list)
    return model_list

def get_all_models(sort_by, filter_by, path):
    res = OrderedDict()
    fileinfos = traverse_all_files(path, [])
    filter_by = filter_by.strip(" ")
    if len(filter_by) != 0:
        fileinfos = [x for x in fileinfos if filter_by.lower()
                     in os.path.basename(x[0]).lower()]
    if sort_by == "name":
        fileinfos = sorted(fileinfos, key=lambda x: os.path.basename(x[0]))
    elif sort_by == "date":
        fileinfos = sorted(fileinfos, key=lambda x: -x[1].st_mtime)
    elif sort_by == "path name":
        fileinfos = sorted(fileinfos)

    for finfo in fileinfos:
        filename = finfo[0]
        name = os.path.splitext(os.path.basename(filename))[0]
        # Prevent a hypothetical "None.pt" from being listed.
        if name != "None":
            res[name + f" [{sd_models.model_hash(filename)}]"] = filename

    return res

def update_models():
    models_paths.clear()
    ext_dirs = (shared.opts.data.get("seecoder_path", None), getattr(shared.cmd_opts, 'seecoder_dir', None))
    extra_lora_paths = (extra_lora_path for extra_lora_path in ext_dirs
                if extra_lora_path is not None and os.path.exists(extra_lora_path))
    paths = [models_dir, models_dir_old, *extra_lora_paths]

    for path in paths:
        sort_by = shared.opts.data.get(
            "seecoder_sort_models_by", "name")
        filter_by = shared.opts.data.get("seecoder_name_filter", "")
        found = get_all_models(sort_by, filter_by, path)
        models_paths.update({**found, **models_paths})

    models_paths_copy = OrderedDict(models_paths)
    models_paths.clear()
    models_paths.update({**{"None": None}, **models_paths_copy})

    models_names.clear()
    for name_and_hash, filename in models_paths.items():
        if filename is None:
            continue
        name = os.path.splitext(os.path.basename(filename))[0].lower()
        models_names[name] = name_and_hash
