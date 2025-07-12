module divide_reg #(
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [WIDTH-1:0] a,
    output wire [WIDTH-1:0] result
);

    // 고정 분모 1.0 (float16: 16'h3C00)
    localparam [WIDTH-1:0] ONE = 16'h4010;

    floating_point_div u_fp_div (
        .aclk(clk),
        .s_axis_a_tvalid(en),
        .s_axis_a_tdata(a),
        .s_axis_b_tvalid(en),
        .s_axis_b_tdata(ONE),
        .m_axis_result_tvalid(),     // 무시
        .m_axis_result_tready(1'b1), // 항상 1
        .m_axis_result_tdata(result)
    );

endmodule
