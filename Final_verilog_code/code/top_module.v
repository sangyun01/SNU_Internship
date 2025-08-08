module top_module (
    input  wire clk,
    input  wire rst_n
);

    wire              in_valid;
    wire              in_ready;
    wire              out_valid;
    wire              out_ready;
    wire [64*16-1:0]  a_in;
    wire [64*16-1:0]  norm_out;

    LN_module u_ln (
        .clk(clk),              // 1bit, input
        .rst_n(rst_n),          // 1bit, input
        .in_valid(in_valid),    // 1bit, input
        .in_ready(in_ready),    // 1bit, output
        .out_valid(out_valid),  // 1bit, output
        .out_ready(out_ready),  // 1bit, input
        .a_in(a_in),            // 64*16bit => 1024, input
        .norm_out(norm_out)     // 64*16bit => 1024, output
    );

    ila_LN u_ila (
        .clk(clk),          // 1bit, input
        .probe0(rst_n),     // 1bit, input
        .probe1(in_valid),  // 1bit, input
        .probe2(in_ready),  // 1bit, input
        .probe3(out_valid), // 1bit, input
        .probe4(out_ready), // 1bit, input
        .probe5(a_in),      // 64*16bit => 1024, input
        .probe6(norm_out)   // 64*16bit => 1024, input
    );
    
    LN_FSM u_fsm (
    .clk(clk),              // 1bit, input
    .rst_n(rst_n),          // 1bit, input
    .in_valid(in_valid),    // 1bit, input
    .in_ready(in_ready),    // 1bit, input
    .a_in(a_in),            // 64*16bit => 1024, output
    .out_valid(out_valid),  // 1bit, output
    .out_ready(out_ready)   // 1bit, output
);


endmodule
