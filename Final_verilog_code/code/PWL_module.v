module PWL_module (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        in_valid,
    output wire        in_ready,
    input  wire [15:0] x_in,
    output wire [15:0] y_out,
    output wire        out_valid,
    input  wire        out_ready
);

    assign in_ready = ~out_valid | out_ready;
    
    wire [15:0] slope_sqrt, intercept_sqrt;
    wire        sqrt_lut_valid, sqrt_eval_valid;
    wire [15:0] sqrt_x;

    wire [15:0] slope_rsqrt, intercept_rsqrt;
    wire        rsqrt_lut_valid;

    // Stage 1: sqrt LUT
    LUT_sqrt lut_sqrt_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(in_valid),
        .x_in(x_in),
        .slope_out(slope_sqrt),
        .intercept_out(intercept_sqrt),
        .out_valid(sqrt_lut_valid)
    );

    // Stage 1: sqrt PWL eval
    PWL_eval #(16) sqrt_eval_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(sqrt_lut_valid),
        .x_in(x_in),
        .slope(slope_sqrt),
        .intercept(intercept_sqrt),
        .y_out(sqrt_x),
        .out_valid(sqrt_eval_valid)
    );

    // Stage 2: rsqrt LUT
    LUT_rsqrt lut_rsqrt_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(sqrt_eval_valid),
        .x_in(sqrt_x),
        .slope_out(slope_rsqrt),
        .intercept_out(intercept_rsqrt),
        .out_valid(rsqrt_lut_valid)
    );

    // Stage 2: rsqrt PWL eval
    PWL_eval #(16) rsqrt_eval_inst (
        .clk(clk),
        .rst_n(rst_n),
        .in_valid(rsqrt_lut_valid),
        .x_in(sqrt_x),
        .slope(slope_rsqrt),
        .intercept(intercept_rsqrt),
        .y_out(y_out),
        .out_valid(out_valid)
    );

endmodule
