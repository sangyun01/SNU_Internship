module adder_tree #(
    parameter N = 4,
    parameter WIDTH = 16
)(
    input  wire clk,
    input  wire rst,
    input  wire en,
    input  wire [N*WIDTH-1:0] in_vec,
    output wire [WIDTH-1:0] sum_out
);

    // 입력 분리
    wire [WIDTH-1:0] in_data [0:N-1];
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : input_split
            assign in_data[i] = in_vec[i*WIDTH +: WIDTH];
        end
    endgenerate

    // 내부 트리 버스 정의 (최대 N/2 레벨)
    wire [WIDTH-1:0] level_1 [0:(N/2)-1];
    wire [WIDTH-1:0] level_2;

    // 첫 번째 레벨 (N=4 기준: 4개 → 2개)
    generate
        for (i = 0; i < N/2; i = i + 1) begin : level1_add
            floating_point_addsub u_fp_add_L1 (
                .aclk(clk),
                .s_axis_a_tvalid(en),
                .s_axis_a_tdata(in_data[2*i]),
                .s_axis_b_tvalid(en),
                .s_axis_b_tdata(in_data[2*i+1]),
                .s_axis_operation_tvalid(en),
                .s_axis_operation_tdata(2'b00), // ADD
                .m_axis_result_tvalid(), // 무시
                .m_axis_result_tready(1'b1),
                .m_axis_result_tdata(level_1[i])
            );
        end
    endgenerate

    // 두 번째 레벨 (2개 → 1개)
    floating_point_addsub u_fp_add_L2 (
        .aclk(clk),
        .s_axis_a_tvalid(en),
        .s_axis_a_tdata(level_1[0]),
        .s_axis_b_tvalid(en),
        .s_axis_b_tdata(level_1[1]),
        .s_axis_operation_tvalid(en),
        .s_axis_operation_tdata(2'b00),
        .m_axis_result_tvalid(),
        .m_axis_result_tready(1'b1),
        .m_axis_result_tdata(level_2)
    );

    assign sum_out = level_2;

endmodule
