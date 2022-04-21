CPLX = 2
BYTE_BITS = 8
SAMPLE_BITS = 10
N_POLS = 2

# dsim_host_data = "10.100.44.1"
# dsim_port_data = 7148
dsim_host_katcp =  "qgpu01.sdpdyn.kat.ac.za"
dsim_port_katcp = 7147

srcs = [
    [
        ("239.103.0.64", 7148),
        ("239.103.0.65", 7148),
        ("239.103.0.66", 7148),
        ("239.103.0.67", 7148),
        ("239.103.0.68", 7148),
        ("239.103.0.69", 7148),
        ("239.103.0.70", 7148),
        ("239.103.0.71", 7148),
    ],
    [
        ("239.103.0.72", 7148),
        ("239.103.0.73", 7148),
        ("239.103.0.74", 7148),
        ("239.103.0.75", 7148),
        ("239.103.0.76", 7148),
        ("239.103.0.77", 7148),
        ("239.103.0.78", 7148),
        ("239.103.0.79", 7148),
    ]
]

# tests = ['cw', 'noise']
# noise_scales = [0.25]
# noise_scales = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]
# cw_scales = [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625, 0.001953125, 0.000976562]
# frequencies_to_check = [10e6, 100e6, 200e6, 300e6, 400e6, 500e6, 600e6, 700e6, 800e6, 850e6]

value_sets = [  ('cw', 1.0, 100e6, 0), 
                ('cw', 0.5, 100e6, 0), 
                ('cw', 0.25, 100e6, 0),
                ('cw', 0.125, 100e6, 0),
                ('cw', 0.0625, 100e6, 0),
                ('cw', 0.03125, 100e6, 0),
                ('cw', 0.015625, 100e6, 0),
                ('cw', 0.0078125, 100e6, 0),
                ('cw', 0.00390625, 100e6, 0),
                ('cw', 0.001953125, 100e6, 0),
                ('cw', 0.000976562, 100e6, 0),
                ('noise', 0.0, 100e6, 1.0), 
                ('noise', 0.0, 100e6, 0.5), 
                ('noise', 0.0, 100e6, 0.25),
                ('noise', 0.0, 100e6, 0.125),
                ('noise', 0.0, 100e6, 0.0625),
                ('noise', 0.0, 100e6, 0.03125),
                ('noise', 0.0, 100e6, 0.015625),
                ('noise', 0.0, 100e6, 0.0078125),
                ('noise', 0.0, 100e6, 0.00390625),
                ('freq_range', 0.5, 10e6, 0),
                ('freq_range', 0.5, 100e6, 0),
                ('freq_range', 0.5, 200e6, 0),
                ('freq_range', 0.5, 300e6, 0),
                ('freq_range', 0.5, 400e6, 0),
                ('freq_range', 0.5, 500e6, 0),
                ('freq_range', 0.5, 600e6, 0),
                ('freq_range', 0.5, 700e6, 0),
                ('freq_range', 0.5, 800e6, 0),
                ('freq_range', 0.5, 850e6, 0),
                ]


# Allan variance parameters
n = 3000

# Histogram resolution
hist_res = 33
