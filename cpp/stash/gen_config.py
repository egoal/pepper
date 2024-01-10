import yaml
import re
import math # for simple eval

header_template = '''
//! this file is generated, never do you modify this. see config_gen.py

#ifndef ALGO_CONFIG_H_
#define ALGO_CONFIG_H_

#include <string>

namespace cloud_mapping::algo_ul::config {{

struct Config final {{
  Config(const Config&) = delete;
  Config& operator=(const Config&) = delete;

  //* >>> config goes from here.
{}
  //* <<<

  static const Config& Instance(const std::string& filepath = "") {{
    static Config ac(filepath);
    return ac;
  }}

 private:
  Config(const std::string& filepath);
}};

}} // namespace cloud_mapping::algo_ul::config

// shorthand.
#define ALGO_CONFIG cloud_mapping::algo_ul::config::Config::Instance()

#endif
'''

source_template = '''
#include "algo_config.h"

#include <experimental/filesystem>

#include "glog/logging.h"
#include "yaml-cpp/yaml.h"

namespace cloud_mapping::algo_ul::config {{

Config::Config(const std::string& config_path) {{
  namespace fs = std::experimental::filesystem;

  if (!fs::exists(config_path)) {{
    LOG(INFO) << "config path \\"" << config_path
              << "\\" does not exist, use default values instead.";
    return;
  }}

  YAML::Node root = YAML::LoadFile(config_path);
  LOG(INFO) << "Parsing config \\"" << config_path << "\\" succeeded.";

  // read values from config files

{}
  
}}
  
}}

'''


def load_yaml(yamlfile)-> dict:
    try:
        with open(yamlfile, "r") as fin:
            return yaml.safe_load(fin)
    except yaml.YAMLError as exc:
        print(exc)


def cpp_value_str(value):
    if type(value) == bool:
        return str.lower(str(value))
    return str(value)


def alter_string(key, value):
    assert type(value)== str 
    if value.startswith(';'):
        return key, eval(value[1:])
    raise RuntimeError(f"usage of string is banned for now, only equations alowed. on <{key} : {value}>")

primtypes = {int: "int", float: "double", bool: "bool", }


def gen_header(key, value):
    t = type(value)

    if t== str:
        key, value = alter_string(key, value)
        return gen_header(key, value)

    if t in primtypes:
        return f"{primtypes[t]} {key}{{{cpp_value_str(value)}}};"

    if t != dict:
        raise RuntimeError(f"missing primtype: {t} of value <{key} : {value}>")

    code = "struct {\n"
    for k, v in value.items():
        code += gen_header(k, v) + "\n"
    code += f"}} {key};"
    return code


block_count = 0


def gen_blockname():
    global block_count
    block_count += 1
    return f"__var{block_count}"


def gen_source(key, value, blockname="root", context=None):
    newblock = gen_blockname()

    code = f'if (auto {newblock} = {blockname}["{key}"]) {{\n'

    t = type(value)
    if t== str:
        key, value = alter_string(key, value)
        return gen_source(key, value, blockname, context)

    if t in primtypes:
        varname = key if context is None else (context + key)
        code += f"{varname} = {newblock}.as<{primtypes[t]}>();\n"
    else:
        for k, v in value.items():
            newcontext = (context or "") + key + "."
            code += gen_source(k, v, newblock, newcontext)

    code += "}\n"

    return code


def main(yamlfile, hfile, cppfile):    
    data = load_yaml(yamlfile)

    header_body = ''
    source_body = ''

    for k, v in data.items():
        header_body += gen_header(k, v) + "\n\n"
        source_body += gen_source(k, v) + "\n\n"

    header_code = str.format(header_template, header_body)
    source_code = str.format(source_template, source_body)

    with open(hfile, 'w') as fh, open(cppfile, "w") as fcpp:
        fh.write(header_code)
        fcpp.write(source_code)

    print(f"{yamlfile} -> {hfile}, {cppfile}")


if __name__=="__main__":
    yamlfile = 'algo_config.yaml'
    hfile = 'algo_config.h'
    cppfile = 'algo_config.cpp'

    main(yamlfile, hfile, cppfile)
