module recip_sqrt_reg #(
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [WIDTH-1:0] in_val,
    output wire [WIDTH-1:0] out_val
);

    floating_point_rsqrt u_fp_rsqrt (
        .aclk(clk),
        .s_axis_a_tvalid(en),
        .s_axis_a_tdata(in_val),
        .m_axis_result_tvalid(),     // 무시
        .m_axis_result_tready(1'b1), // 항상 1
        .m_axis_result_tdata(out_val)
    );

endmodule
