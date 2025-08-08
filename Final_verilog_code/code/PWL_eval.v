module PWL_eval #(parameter N = 16)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    input  wire signed [N-1:0] x_in,        // Q8.8 입력값
    input  wire signed [N-1:0] slope,       // Q8.8 slope
    input  wire signed [N-1:0] intercept,   // Q8.8 intercept
    output wire signed [N-1:0] y_out,       // Q8.8 결과값
    output wire               out_valid
);

    // 1. 곱셈 (Q8.8 x Q8.8 = Q16.16)
    wire signed [2*N-1:0] mult_out;
    
    mult_gen_mult mult_inst (
        .CLK(clk),
        .A  (slope),
        .B  (x_in),
        .P  (mult_out)
    );

    // 2. 곱셈 결과 스케일 조정 (Q16.16 → Q8.8)
    wire signed [N-1:0] scaled_result = mult_out[23:8];

    // 3. 덧셈 (scaled_result + intercept)
    wire signed [N-1:0] sum_out;

    c_addsub_add add_inst (
        .CLK(clk),
        .A  (scaled_result),
        .B  (intercept),
        .CE(1'b1),
        .S  (sum_out)
    );

    // 4. valid 파이프라인 (mult: 4 + add: 1 = 총 5 cycle)
    reg [4:0] valid_pipe;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) valid_pipe <= 5'b0;
        else        valid_pipe <= {valid_pipe[3:0], in_valid};
    end

    assign y_out     = sum_out;
    assign out_valid = valid_pipe[4];

endmodule
