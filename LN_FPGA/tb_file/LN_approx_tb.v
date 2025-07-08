`timescale 1ns / 1ps

module LN_approx_tb;

    reg clk, rst;
    reg [255:0] x_in_flat;
    wire [15:0] mean_out;

    LN_approx uut (
        .clk(clk),
        .rst(rst),
        .x_in_flat(x_in_flat),
        .mean_out(mean_out)
    );

    // clock
    always #5 clk = ~clk;

    // Q8.8 to float
    function real q8_8_to_real(input [15:0] val);
        begin
            q8_8_to_real = $itor(val) / 256.0;
        end
    endfunction

    integer i;
    reg [15:0] x_vals [0:15];

    initial begin
        clk = 0; rst = 1;
        #10 rst = 0;

        // 입력: 모두 1.0 = 256
        for (i = 0; i < 16; i = i + 1)
            x_vals[i] = 16'd256;

        // Pack into flat input
        for (i = 0; i < 16; i = i + 1)
            x_in_flat[i*16 +: 16] = x_vals[i];

        #100;
        $display("Mean = %f (expected 1.0)", q8_8_to_real(mean_out));
        $finish;
    end

endmodule
