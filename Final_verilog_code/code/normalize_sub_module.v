module normalize_sub_module #(
    parameter N = 64
)(
    input  wire              clk,
    input  wire              rst_n,
    input  wire              in_valid,
    output wire              in_ready,
    input  wire [N*16-1:0]   x_in,
    input  wire [15:0]       mu,
    output wire [N*16-1:0]   diff_out,
    output reg               out_valid,
    input  wire              out_ready
);

    assign in_ready = ~out_valid | out_ready;

    wire [15:0] x     [0:N-1];
    wire [15:0] diff  [0:N-1];

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : SUB_GEN
            assign x[i] = x_in[i*16 +: 16];

            c_addsub_sub sub_inst (
                .A(x[i]),
                .B(mu),
                .CLK(clk),
                .CE  (1'b1),
                .S(diff[i])
            );

            assign diff_out[i*16 +: 16] = diff[i];
        end
    endgenerate

    // 1 clock delay cycle to store reg
    reg [0:0] delay_cnt;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            delay_cnt <= 0;
            out_valid <= 0;
        end else if (in_valid && in_ready) begin
            delay_cnt <= 1;
            out_valid <= 0;
        end else if (delay_cnt != 0) begin
            delay_cnt <= delay_cnt - 1;
            out_valid <= 1;  // out_valid is 1 after 1 clk delayed
        end else if (out_valid && out_ready) begin
            out_valid <= 0;
        end else begin
            out_valid <= 0;
        end
    end


endmodule
