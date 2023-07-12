import numpy as np

def read8(data, offset):
    return data[offset]

def read16(data, offset):
    return data[offset] | (data[offset + 1] << 8)

def read32(data, offset):
    return data[offset] | (data[offset + 1] << 8) | (data[offset + 2] << 16) | (data[offset + 3] << 24)

def read_str(data, offset):
    end = offset
    while data[end] != 0:
        end += 1
    return data[offset:end].tobytes().decode('ascii')

def read_dict_string(data, offset):
    end = offset
    while data[end] != 0:
        end += 1
        if (end - offset) == 16:
            break
    return data[offset:end].tobytes().decode('ascii')

def log(string, report_func):
    report_func(type={"INFO"}, message=string)

def error(string, report_func):
    report_func(type={"ERROR"}, message=string)

def debug(string, report_func):
    report_func(type={"DEBUG"}, message=string)

def parse_dictionary(data):
    num_entries = read8(data, 0x01)
    data_offset = read16(data, 0x06)

    data_size = read16(data, data_offset + 0x00)
    name_offset = read16(data, data_offset + 0x02)

    dictionary = {}
    for i in range(num_entries):
        name = read_dict_string(data, data_offset + name_offset + i * 0x10)
        value = 0
        if data_size == 1:
            value = read8(data, data_offset + 0x04 + i * 0x01)
        elif data_size == 2:
            value = read16(data, data_offset + 0x04 + i * 0x02)
        elif data_size == 4:
            value = read32(data, data_offset + 0x04 + i * 0x04)
        dictionary[name] = value
    
    return dictionary

def fixed_to_float(value):
    return float(value / 4096.0)

def to_rgb(color):
    return (color & 0x1F, (color >> 5) & 0x1F, (color >> 10) & 0x1F)

def np_fixed_to_float(value):
    return (value / 4096.0).astype('float')

def vec10_to_vec(value):
    return np_fixed_to_float(np.array([value & 0x3FF, (value >> 10) & 0x3FF, (value >> 20) & 0x3FF]))