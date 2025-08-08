## Clock: 50MHz
set_property PACKAGE_PIN E3 [get_ports clk]
set_property IOSTANDARD LVCMOS33 [get_ports clk]
create_clock -period 20.000 -name sys_clk -waveform {0 5} [get_ports clk]

## Reset
set_property PACKAGE_PIN D4 [get_ports rst_n]
set_property IOSTANDARD LVCMOS33 [get_ports rst_n]

