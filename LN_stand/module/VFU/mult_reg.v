module mult_reg #(
    parameter N = 4,
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [N*WIDTH-1:0] a_vec,
    input  wire [N*WIDTH-1:0] b_vec,
    output wire [N*WIDTH-1:0] result_vec
);

genvar i;
generate
    for (i = 0; i < N; i = i + 1) begin : mult_array
        wire en_i;
        assign en_i = en;

        floating_point_mult u_fp_mult (
            .aclk(clk),
            .s_axis_a_tvalid(en_i),
            .s_axis_a_tdata(a_vec[i*WIDTH +: WIDTH]),
            .s_axis_b_tvalid(en_i),
            .s_axis_b_tdata(b_vec[i*WIDTH +: WIDTH]),
            .m_axis_result_tvalid(),         // 무시
            .m_axis_result_tready(1'b1),     // 필수!
            .m_axis_result_tdata(result_vec[i*WIDTH +: WIDTH])
        );
    end
endgenerate

endmodule
